'''
Simulated data object compatible with Prometheus.
'''

import numpy as np
import scipy.linalg as sl
from tqdm import tqdm
from scipy.signal.windows import tukey
import jax.numpy as jnp

import utilities as utils
from pulsar import Pulsar
from pta import PTA
from prometheus.utilities import renorm


class SimulatedData:

    def __init__(self, pta: PTA, name, float32=True, det_window_ext_factor=2.0, nfreqs_det=60, ecorr=False):
        self.pta = pta
        self.name = name
        self.det_window_ext_factor = det_window_ext_factor
        self.nfreqs_det = nfreqs_det
        self.ecorr = ecorr
        self.params_inj = pta.params_inj

        if float32:
            self.float_dtype = np.float32
            self.single_precision = True
        else:
            self.float_dtype = np.float64
            self.single_precision = False
        
        # objects needed for likelihood
        self.FNrs, self.FNFs = self.pta.get_likelihood_objects(jax=True, single_precision=self.single_precision)

        # attributes of `prometheus.data.Data` object

        # general PTA attributes
        self.nfreqs = self.pta.Nf
        self.psr_names = [self.pta.pulsars[i].name for i in range(self.pta.npsrs)]
        self.npsrs = len(self.psr_names)
        self.Tspan = self.pta.Tspan
        self.ncomponents = self.pta.Na
        self.freqs_unique = jnp.array(self.pta.freqs, dtype=self.float_dtype)
        self.freqs = jnp.repeat(self.freqs_unique, 2)

        # pulsar sky locations
        self.psr_phi = jnp.array([self.pta.pulsars[i].phi for i in range(self.pta.npsrs)], dtype=self.float_dtype)
        self.psr_theta = jnp.array([self.pta.pulsars[i].theta for i in range(self.pta.npsrs)], dtype=self.float_dtype)
        self.psrpos = jnp.array([self.pta.pulsars[i].psrpos for i in range(self.pta.npsrs)], dtype=self.float_dtype)

        # pulsar data dictionary
        self.per_psr_data_dict = self.build_per_psr_data_dict()
        
        # pulsar measured distances and uncertainty
        self.psr_dists_measured = jnp.array([self.per_psr_data_dict[name]['pdist']
                                             for name in self.psr_names])[:, 0]
        self.psr_dists_std = jnp.array([self.per_psr_data_dict[name]['pdist']
                                       for name in self.psr_names])[:, 1]
        self.psr_dist_method = np.array([self.per_psr_data_dict[name]['psr_dist_method']
                                          for name in self.psr_names])
        
        # constants needed for stochastic part of posterior
        self.Sigma_0_inv_jc = jnp.stack([self.per_psr_data_dict[psrname]['Sigma_inv']/renorm**2
                                         for psrname in self.psr_names])
        self.Sigma_0_inv_j = jnp.array(sl.block_diag(*[(self.per_psr_data_dict[psrname]['Sigma_inv']/renorm**2).astype(np.float32)
                                                       for psrname in self.psr_names]))
        self.a_hat_j = jnp.array(np.concatenate([self.per_psr_data_dict[psrname]['a_hat']
                                                 for psrname in self.psr_names]).astype(np.float32))

        self.phiinv_0_j = jnp.array(np.concatenate([self.per_psr_data_dict[psrname]['phiinv']/renorm**2
                                                    for psrname in self.psr_names]).astype(np.float32))
        self.phiinv_logdet_0_j = jnp.sum(np.log(self.phiinv_0_j*renorm**2))
        self.Sigma_0_logdet_j = jnp.array(np.sum([self.per_psr_data_dict[psrname]['logdet']\
                                                  for psrname in self.psr_names]))
        self.Si0_a_hat_j = jnp.dot(self.Sigma_0_inv_j, self.a_hat_j) * renorm

        # Also make phiinv_0 cubed. For vmap, need it in both forms:
        # - Npsr x (nfreqs x nfreqs) and
        # - Nfreqs x (Npsr x Npsr)
        self.phiinv_0_vecs_j = jnp.stack([self.per_psr_data_dict[psrname]['phiinv']/renorm**2
                                          for psrname in self.psr_names])   # npsrs x nfreqs
        self.phiinv_0_cube_pf = jnp.zeros((self.phiinv_0_vecs_j.shape[0], self.phiinv_0_vecs_j.shape[1], self.phiinv_0_vecs_j.shape[1]))
        self.phiinv_0_cube_fp = jnp.zeros((self.phiinv_0_vecs_j.shape[1], self.phiinv_0_vecs_j.shape[0], self.phiinv_0_vecs_j.shape[0]))
        self.ii_diag_pf = jnp.arange(self.phiinv_0_vecs_j.shape[1])
        self.ii_diag_fp = jnp.arange(self.phiinv_0_vecs_j.shape[0])
        self.phiinv_0_cube_pf = self.phiinv_0_cube_pf.at[:, self.ii_diag_pf, self.ii_diag_pf].set(self.phiinv_0_vecs_j)
        self.phiinv_0_cube_fp = self.phiinv_0_cube_fp.at[:, self.ii_diag_fp, self.ii_diag_fp].set(self.phiinv_0_vecs_j.T)
        self.a_hat_2d_pf = jnp.stack(([self.per_psr_data_dict[psrname]['a_hat'] * renorm
                                       for psrname in self.psr_names]))  # npsrs x nfreqs
        self.Si0_a_hat_j_pf = jnp.stack([np.dot(self.per_psr_data_dict[psrname]['Sigma_inv']/renorm**2,
                                        self.per_psr_data_dict[psrname]['a_hat'] * renorm)
                                        for psrname in self.psr_names]) # equivalent to TNr, but may be marginalized over WN if desired

        # constants needed for deterministic part of likelihood
        self.num_coeff_det = self.per_psr_data_dict[self.psr_names[0]]['num_coeff_det']
        self.sparse_toas_det = jnp.array([self.per_psr_data_dict[name]['sparse_toas_det']
                                         for name in self.psr_names])
        self.Nsparse = self.sparse_toas_det.shape[1]
        self.freqs_forFFT = jnp.array([self.per_psr_data_dict[name]['freqs_forFFT']
                                       for name in self.psr_names])
        self.Tspan_ext = jnp.array([self.per_psr_data_dict[name]['Tspan_ext']
                                    for name in self.psr_names])[0]
        self.Tukey_det = jnp.array(tukey(self.Nsparse, alpha=(self.Tspan_ext - self.Tspan)/self.Tspan_ext))
        self.TDNTDs = jnp.array([self.per_psr_data_dict[name]['TDNTD']
                                 for name in self.psr_names]) / (renorm**2.)
        self.TNTDs = jnp.array([self.per_psr_data_dict[name]['TNTD']
                                for name in self.psr_names]) / (renorm**2.)
        self.TDNrs = jnp.array([self.per_psr_data_dict[name]['TDNr']
                                for name in self.psr_names]) / renorm

    def build_per_psr_data_dict(self):

        # save necessary data in dictionary
        data_dict = dict()

        # first and last toas across PTA
        tmins = [p.toas.min() for p in self.pta.pulsars]
        tmaxs = [p.toas.max() for p in self.pta.pulsars]
        Tspan = np.max(tmaxs) - np.min(tmins)

        with tqdm(range(self.npsrs), desc='building pulsar models') as pbar:
            for i in pbar:
                psr = self.pta.pulsars[i]
                pbar.set_postfix_str(f'running {psr.name}')

                # reference power law parameters to regularize covariance matrices
                log10_Arn = -12
                gamma = 4.33

                # number of frequency bins for pulsar noise model
                nfrequencies = self.nfreqs

                # arrays needed for posterior evaluation
                FNF = self.FNFs[i]
                FNr = self.FNrs[i]
                phiinv = utils.power_law(log10_Arn, gamma, self.freqs_unique)
                Sigma_inv = FNF + np.diag(phiinv)
                Li = sl.cholesky(Sigma_inv, lower=True)
                Sigma = sl.cho_solve((Li, True), np.identity(len(Li)))
                a_hat = sl.cho_solve((Li, True), FNr)
                logdet = -2 * np.sum(np.log(np.diag(Li)))

                # save alternative basis used for deterministic signals
                # window extension for deterministic FFT (avoids Gibbs phenomena)
                window_ext = self.pta.Tspan * self.det_window_ext_factor
                Tspan_ext = self.pta.Tspan + 2. * window_ext
                Nf_det = self.nfreqs_det
                num_coeff_det = 2 * Nf_det

                # sparse TOAs for CW FFT
                toas = psr.toas
                sparse_toas_det = np.linspace(np.min(tmins) - window_ext, np.max(tmaxs) + window_ext,
                                              num_coeff_det + 2, endpoint=False)
                Nsparse = sparse_toas_det.shape[0]
                freqs_forFFT = np.fft.fftfreq(Nsparse, Tspan_ext / Nsparse)

                # Fourier design matrix for CW
                F_D = np.zeros((toas.shape[0], num_coeff_det))
                for j in range(Nf_det):
                    F_D[:, 2 * j] = np.sin(2. * np.pi * freqs_forFFT[j + 1] * toas)
                    F_D[:, 2 * j + 1] = np.cos(2. * np.pi * freqs_forFFT[j + 1] * toas)

                # arrays needed for posterior evaluation with CW in model
                F = self.pta.Fs[i]
                Ntinv = psr.Ntinv
                TDNTD = F_D.T @ Ntinv @ F_D
                TNTD = F.T @ Ntinv @ F_D
                TDNr = F_D.T @ Ntinv @ psr.residuals
                
                psr_dist_method = 'simulated'
                psr_dist_and_uncertainty = (psr.dist_kpc, psr.dist_kpc_std)
                            
                # store pulsar data and associated objects in dictionary
                data_dict[psr.name] = dict(
                    phi = psr.phi,
                    theta = np.arccos(psr.costheta),
                    Tspan = Tspan,
                    log10_Arn = log10_Arn,
                    gamma = gamma,
                    nfrequencies = nfrequencies,
                    ncomponents = 2 * nfrequencies,
                    Li = Li,
                    Sigma = Sigma,
                    Sigma_inv = Sigma_inv,
                    phiinv = phiinv,
                    a_hat = a_hat,
                    logdet = logdet,
                    TNr = FNr,
                    TNT = FNF,
                    pdist = psr_dist_and_uncertainty,
                    psr_dist_method = psr_dist_method,
                    # toas = psr.toas,
                    # residuals = psr.residuals,
                    # F = pta_psr.get_basis()[0],
                    # FD = F_D,
                    num_coeff_det = num_coeff_det,
                    TDNTD = TDNTD,
                    TNTD = TNTD,
                    TDNr = TDNr,
                    sparse_toas_det = sparse_toas_det,
                    freqs_forFFT = freqs_forFFT,
                    Tspan_ext = Tspan_ext,
                )

        return data_dict

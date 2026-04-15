'''
Build a PTA out of constituent pulsars with simulation methods.
'''


import numpy as np
import utilities as utils
from pulsar import Pulsar


class PTA:

    def __init__(self, pulsars: list[Pulsar], Nf=30):
        
        # pulsars
        self.pulsars = pulsars
        self.npsrs = len(pulsars)
        self.psrpos = np.array([psr.psrpos for psr in pulsars])
        self.psrdists = np.array([psr.dist_kpc for psr in pulsars])
        self.psrdists_std = np.array([psr.dist_kpc_std for psr in pulsars])

        # timing / frequency attributes
        self.toa_mins = np.array([psr.toas.min() for psr in pulsars])
        self.toa_maxs = np.array([psr.toas.max() for psr in pulsars])
        self.Tspan = np.max(self.toa_maxs) - np.min(self.toa_mins)
        self.Tspan_yrs = self.Tspan / utils.year
        self.Nf = Nf
        self.Na = 2 * self.Nf
        self.freqs = np.arange(1, Nf + 1) / self.Tspan

        # Fourier design matrices
        Fs = []
        for i in range(self.npsrs):
            F = np.zeros((self.pulsars[i].Ntoas, self.Na))
            F[:, ::2] = np.sin(2. * np.pi * self.freqs[None, :] * self.pulsars[i].toas[:, None])
            F[:, 1::2] = np.cos(2. * np.pi * self.freqs[None, :] * self.pulsars[i].toas[:, None])
            Fs.append(F)
        self.Fs = Fs

        # Hellings-Downs correlation
        cosgamma = np.clip(np.dot(self.psrpos, self.psrpos.T), -1, 1)
        xp = 0.5 * (1 - cosgamma)
        old_settings = np.seterr(all='ignore')
        logxp = 1.5 * xp * np.log(xp)
        np.fill_diagonal(logxp, 0)
        np.seterr(**old_settings)
        self.hdmat = logxp - 0.25 * xp + 0.5 + 0.5 * np.diag(np.ones(self.npsrs))

        # store injected parameter values
        self.params_inj = {}

    # output vectors / matrices used in likelihood evalutions
    def get_likelihood_objects(self, scale=1.0, jax=False, single_precision=True):
        
        # Fourier objects used in likelihood evaluations
        FNrs = []
        FNFs = []
        for ii, pulsar in enumerate(self.pulsars):
            FNr = self.Fs[ii].T @ pulsar.Ntinv @ pulsar.projected_residuals()
            FNF = self.Fs[ii].T @ pulsar.Ntinv @ self.Fs[ii]
            FNrs.append(FNr)
            FNFs.append(FNF)
        
        # rescale terms if using different units
        FNrs = np.array(FNrs) / scale
        FNFs = np.array(FNFs) / scale**2
        
        # convert to jax arrays if desired
        if jax:
            import jax.numpy as jnp
            FNrs = jnp.array(FNrs, dtype=jnp.float32 if single_precision else jnp.float64)
            FNFs = jnp.array(FNFs, dtype=jnp.float32 if single_precision else jnp.float64)
        
        return FNrs, FNFs
    
    # intrinsic pulsar red noise
    def add_irn_delay(self, log10_As, gammas, seed=0):
        np.random.seed(seed)
        a_irn_inj = []
        for ii, pulsar in enumerate(self.pulsars):
            log10_A = log10_As[ii]
            gamma = gammas[ii]
            phi_diag = utils.power_law(log10_A, gamma, self.freqs)
            z = np.random.normal(size=self.Na)
            a_psr_irn_inj = np.sqrt(phi_diag) * z
            irn_delay = self.Fs[ii] @ a_psr_irn_inj
            pulsar.add_delay(irn_delay)
            a_irn_inj.append(a_psr_irn_inj)
        self.params_inj['a_irn'] = np.array(a_irn_inj)
        self.params_inj['rn_pl'] = np.array([[log10_As[ii], gammas[ii]] for ii in range(self.npsrs)])

    # stochastic GWB with Hellings-Downs correlations
    def add_gwb_delay(self, log10_A_gwb, gamma_gwb, seed=0):
        np.random.seed(seed)
        phi_diag = utils.power_law(log10_A_gwb, gamma_gwb, self.freqs)
        phi_flat = np.kron(self.hdmat, np.diag(phi_diag))
        z = np.random.normal(size=(self.npsrs * self.Na))
        a_psr_gwb_inj = np.linalg.cholesky(phi_flat, upper=False) @ z
        a_psr_gwb_inj = a_psr_gwb_inj.reshape((self.npsrs, self.Na))
        for ii, pulsar in enumerate(self.pulsars):
            gwb_delay = self.Fs[ii] @ a_psr_gwb_inj[ii]
            pulsar.add_delay(gwb_delay)
        self.params_inj['a_gwb'] = a_psr_gwb_inj
        self.params_inj['gwb_pl'] = np.array([log10_A_gwb, gamma_gwb])

    # i.i.d. zero-mean Gaussian white noise
    def add_white_noise(self, seed=0):
        for pulsar in self.pulsars:
            pulsar.add_white_noise(seed=seed)
    
    # combined IRN + GWB injection
    def add_irn_gwb_delay(self, log10_As, gammas, log10_A_gwb, gamma_gwb, seed=0):
        np.random.seed(seed)
        phi_diag_irn = np.array([utils.power_law(log10_As[ii], gammas[ii], self.freqs) for ii in range(self.npsrs)]).flatten()
        phi_gwb = np.kron(self.hdmat, np.diag(utils.power_law(log10_A_gwb, gamma_gwb, self.freqs)))
        phi_diag = phi_gwb + np.diag(phi_diag_irn)
        z = np.random.normal(size=(self.npsrs * self.Na))
        a_inj = np.linalg.cholesky(phi_diag, upper=False) @ z
        a_inj = a_inj.reshape((self.npsrs, self.Na))
        for ii, pulsar in enumerate(self.pulsars):
            delay = self.Fs[ii] @ a_inj[ii]
            pulsar.add_delay(delay)
        self.params_inj['a'] = a_inj
        self.params_inj['gwb_pl'] = np.array([log10_A_gwb, gamma_gwb])
        self.params_inj['rn_pl'] = np.array([[log10_As[ii], gammas[ii]] for ii in range(self.npsrs)])

    # color base_draws with HD-correlated power law spectrum
    def add_non_gaussian_gwb_delay(self, base_draws, base_variance, log10_A_gwb, gamma_gwb, seed=0):
        np.random.seed(seed)
        phi_gwb = np.kron(self.hdmat, np.diag(utils.power_law(log10_A_gwb, gamma_gwb, self.freqs)))
        L = np.linalg.cholesky(phi_gwb, upper=False) * np.sqrt(1. / base_variance)
        a_inj = L @ base_draws.flatten()
        a_inj = a_inj.reshape((self.npsrs, self.Na))
        for ii, pulsar in enumerate(self.pulsars):
            delay = self.Fs[ii] @ a_inj[ii]
            pulsar.add_delay(delay)
        self.params_inj['a_gwb'] = a_inj
        self.params_inj['gwb_pl'] = np.array([log10_A_gwb, gamma_gwb])


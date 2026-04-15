'''
A simulated pulsar object.
'''

import numpy as np
import utilities as utils


class Pulsar:

    def __init__(self, name, costheta, phi, dist_kpc, dist_kpc_std, toas, toa_errors, timing_design_matrix=None):
        self.name = name
        self.costheta = costheta
        self.phi = phi
        self.dist_kpc = dist_kpc
        self.dist_kpc_std = dist_kpc_std
        self.toas = toas
        self.toa_errors = toa_errors
        self.timing_design_matrix = timing_design_matrix

        # if no timing design matrix, assume quadratic spindown model
        if self.timing_design_matrix is None:
            self.timing_design_matrix = np.zeros((len(toas), 3))
            self.timing_design_matrix[:, 0] = 1.0
            self.timing_design_matrix[:, 1] = toas
            self.timing_design_matrix[:, 2] = toas**2

        # timing attributes
        self.mjds = self.toas / utils.day
        self.psr_Tspan = np.max(self.toas) - np.min(self.toas)
        self.psr_Tspan_yrs = self.psr_Tspan / utils.year

        # white noise covariance matrix
        self.Ntoas = len(toas)
        self.N = np.diag(toa_errors**2)
        self.Ninv = np.diag(1.0 / toa_errors**2)

        # projection orthogonal to timing model
        self.R = np.eye(self.Ntoas) - self.timing_design_matrix @ \
            np.linalg.inv(self.timing_design_matrix.T @ self.timing_design_matrix) @ \
                self.timing_design_matrix.T

        # white noise covariance matrix projected orthogonal to timing model
        self.U = np.linalg.svd(self.timing_design_matrix)[0]
        self.G = self.U[:, self.timing_design_matrix.shape[1]:]
        self.Ntinv = self.G @ np.linalg.inv(self.G.T @ self.N @ self.G) @ self.G.T

        # pulsar sky location Cartesian unit vector
        self.theta = np.arccos(self.costheta)
        self.psrpos = np.array([np.sin(self.theta) * np.cos(self.phi),
                                np.sin(self.theta) * np.sin(self.phi),
                                np.cos(self.theta)])
        
        # initially assume timing model is perfect
        self.residuals = np.zeros_like(toas)

    
    def add_delay(self, delay):
        self.residuals += delay

    def add_white_noise(self, seed=0):
        np.random.seed(seed)
        white_noise = np.random.normal(loc=0., scale=1.) * self.toa_errors
        self.residuals += white_noise

    def projected_residuals(self):
        return self.R @ self.residuals
    


def simulate_toas(first_mjd, Tspan_yr, monthly_observations=1, random_offsets_in_days=2, seed=0):
    np.random.seed(seed)
    first_toa = first_mjd * utils.day
    last_toa = first_toa + Tspan_yr * utils.year
    ntoas = int(Tspan_yr * utils.months_in_year * monthly_observations)
    evenly_spaced_toas = np.linspace(first_toa, last_toa, ntoas)
    random_offsets = np.random.normal(loc=0., scale=1., size=ntoas) * random_offsets_in_days * utils.day
    toas = evenly_spaced_toas + random_offsets
    return toas




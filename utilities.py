'''
Constants and useful functions.
'''

import numpy as np

# time constants
day = 24. * 3600.
year = 365.25 * day
months_in_year = 12
fyear = 1. / year


# power law spectral model for Gaussian processes
def power_law(log10_A, gamma, freqs):
    df = np.diff(np.concatenate((np.array([0]), freqs)))
    log10_phi_diag = 2*log10_A - np.log10(12.0 * np.pi**2) +  (gamma - 3)*np.log10(fyear) + \
        (-gamma)*np.log10(np.repeat(freqs, 2)) + np.log10(np.repeat(df, 2))
    phi_diag = 10**log10_phi_diag
    return phi_diag

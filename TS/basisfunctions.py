import numpy as np
import math
from numba import njit, prange
from njit_options import opts


@njit( **opts() )
def basisfunctions_mflsq( n, lat0, lon0, lats, lons, times, NHarm, period=365.25 ) -> np.ndarray:
    """
    Input: 
        lat0, lon0: float64, float64
        lats, lons, times: arrays, {float64} each of size n
        period: float64, period of temporal component
    Return:
        basisfunctions: array{float64} of shape ( n, 6+(NHarm*2) )
    """
    numparams = NHarm*2 + 6
    basisfunctions = np.ones( shape=(n, numparams), dtype=np.float64 )
    TwoPi_Period = 2 * math.pi / period
    for i in prange(n):
        lat = lats[i]-lat0
        lon = lons[i]-lon0
        if lon > 180.0:
            lon = lon - 360.0
        if lon <= -180.0:
            lon = lon + 360.0
        # https://docs.dask.org/en/stable/delayed-best-practices.html#don-t-mutate-inputs
        basisfunctions[i, 0] = lat
        basisfunctions[i, 1] = lon
        basisfunctions[i, 2] = lat**2
        basisfunctions[i, 3] = lon**2
        basisfunctions[i, 4] = lat*lon
        if NHarm > 0:
            time = times[i]
            for harmonic, p in zip(range(1, NHarm + 1), range(6, numparams , 2)):
                phase = TwoPi_Period * harmonic * time
                basisfunctions[i,p] = math.sin(phase)
                basisfunctions[i,p+1] = math.cos(phase)
    return basisfunctions


@njit( fastmath=True, parallel=True, nogil=True, cache=True, error_model = "numpy" )
def basisfunctions_mean( times, NHarm, period=365.25 ) -> np.ndarray:
    """
    Input:
        times: array{float64} of size n
        period: float64, period of temporal component
    Return:
        basisfunctions: array{float64} of shape ( n, numparams )
    """
    numparams = NHarm*2 + 6
    n = times.size
    basisfunctions = np.zeros( shape=(n, numparams), dtype=np.float64 )
    TwoPi_Period = 2 * math.pi / period
    for i in prange(n):
        basisfunctions[i, 5] = 1.0
        if NHarm > 0:
            time = times[i]
            for harmonic, p in zip(range(1, NHarm + 1), range(6, numparams , 2)):
                phase = TwoPi_Period * harmonic * time
                basisfunctions[i,p] = math.sin(phase)
                basisfunctions[i,p+1] = math.cos(phase)
    return basisfunctions
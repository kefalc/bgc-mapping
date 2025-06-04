import numpy as np
import math
from numba import njit, prange
from njit_options import opts

##Update to include options for number of functions
@njit( **opts() ) 
def basisfunctions_mflsq( n, lat0, lon0, lats, lons, temp, sal, times, NHarm, period=365.25 ) -> np.ndarray:
    """
    Input: 
        lat0, lon0: float64, float64
        lats, lons, times: arrays, {float64} each of size n
        period: float64, period of temporal component
    Return:
        basisfunctions: array{float64} of shape ( n, 8+(NHarm*2) )
    """
    
    """ Need to make this into a loop to change numparams """
    
    numparams = NHarm*2 + 8
    basisfunctions = np.ones( shape=(n, numparams), dtype=np.float64 )
    TwoPi_Period = 2 * math.pi / period
    for i in prange(n):
        lat = lats[i]-lat0
        lon = lons[i]-lon0
        
        if lon > 180.0:
            lon = lon - 360.0
        if lon <= -180.0:
            lon = lon + 360.0
        
        ##Generalize function so that this becomes a loop
        # https://docs.dask.org/en/stable/delayed-best-practices.html#don-t-mutate-inputs
        basisfunctions[i, 0] = lat
        basisfunctions[i, 1] = lon
        basisfunctions[i, 2] = lat**2
        basisfunctions[i, 3] = lon**2
        basisfunctions[i, 4] = lat*lon
        basisfunctions[i, 5] = temp[i] #.item()
        basisfunctions[i, 6] = sal[i] #.item()
        # basisfunctions[i, 7] = temp**2
        # basisfunctions[i, 8] = sal**2
        # basisfunctions[i, 9] = temp*sal
        if NHarm > 0:
            time = times[i]
            for harmonic, p in zip(range(1, NHarm + 1), range(8, numparams , 2)):
                phase = TwoPi_Period * harmonic * time
                basisfunctions[i,p] = math.sin(phase)
                basisfunctions[i,p+1] = math.cos(phase)
    return basisfunctions


@njit( fastmath=True, parallel=True, nogil=True, cache=True, error_model = "numpy" )
def basisfunctions_mean( times, mean_temp, mean_salt, NHarm, period=365.25 ) -> np.ndarray:
    """
    Input:
        times: array{float64} of size n
        period: float64, period of temporal component
    Return:
        basisfunctions: array{float64} of shape ( n, numparams )
        
        Add input for mean temp and salinity for columns 5 and 6
    """
    
    """ Need to make this into a loop to change numparams """
    numparams = NHarm*2 + 8
    n = times.size
    basisfunctions = np.zeros( shape=(n, numparams), dtype=np.float64 )
    TwoPi_Period = 2 * math.pi / period
    for i in prange(n):
        basisfunctions[i,5] = mean_temp[i]
        basisfunctions[i,6] = mean_salt[i]
        basisfunctions[i, 7] = 1.0
        if NHarm > 0:
            time = times[i]
            for harmonic, p in zip(range(1, NHarm + 1), range(8, numparams , 2)):
                phase = TwoPi_Period * harmonic * time
                basisfunctions[i,p] = math.sin(phase)
                basisfunctions[i,p+1] = math.cos(phase)
    return basisfunctions

#change sum in window later
def basisfunctions_alt( sum_in_window, time_iw, temp_iw, sal_iw, NHarm, period=365.25 ) -> np.ndarray:
    """
    Alternative using np.zeros, instead of subtracting to produce zeros.
    """
    
    """ Need to make this into a loop to change numparams """
    numparams = NHarm*2 + 8
    basisfunctions = np.zeros( shape=(sum_in_window, numparams), dtype=np.float64 )
    TwoPi_Period = 2 * math.pi / period
    for i in prange(sum_in_window):
        basisfunctions[i,5] = temp_iw[i]
        basisfunctions[i,6] = sal_iw[i]
        basisfunctions[i, 7] = 1.0
        if NHarm > 0:
            time = time_iw[i]
            for harmonic, p in zip(range(1, NHarm + 1), range(8, numparams , 2)):
                phase = TwoPi_Period * harmonic * time
                basisfunctions[i,  p] = math.sin(phase)
                basisfunctions[i,p+1] = math.cos(phase)
    return basisfunctions
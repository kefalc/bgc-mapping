import numpy as np
from numba import njit
from njit_options import opts
import map_anom as ma
""" 
dask.delayed and client.submit limiting number of tasks to ~100_000 due to overhead per delayed object. 
When time, check if dask.bag-API is useful here.
"""

def batch_map_anom( 
    lats, lons, maptimes, window_size, timewindow_size, covparams, data, time, lat, lon
) -> np.ndarray:
    batchsize = lats.size
    mapped_anom = np.empty(shape=(batchsize,maptimes.size,2,))
    for i in range(batchsize):
        mapped_anom[i,:,:] = ma.map_anom(
            lats[i], lons[i], maptimes, window_size, timewindow_size, covparams[i], data, time, lat, lon
        )
    return mapped_anom

#not needed for BGC
def batch_map_anom_2vars( 
    lats, lons, maptimes, window_size, timewindow_size, covparams_1, covparams_2, anom1, anom2, time, lat, lon
) -> (np.ndarray, np.ndarray):
    """ F.ex.:
    anom1: temperature-anomalies, 
    anom2: salinity-anomalies
    """
    ii1 = ~np.isnan(anom1)
    ii2 = ~np.isnan(anom2)
    mapped_anom_1 = batch_map_anom(
        lats, lons, maptimes, window_size, timewindow_size, covparams_1, anom1[ii1], time[ii1], lat[ii1], lon[ii1] 
    )
    mapped_anom_2 = batch_map_anom(
        lats, lons, maptimes, window_size, timewindow_size, covparams_2, anom2[ii2], time[ii2], lat[ii2], lon[ii2] 
    )
    return np.concatenate((mapped_anom_1, mapped_anom_2), axis=2)
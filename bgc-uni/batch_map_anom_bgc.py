import numpy as np
from numba import njit
from njit_options_bgc import opts
import map_anom_bgc as ma
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

def batch_map_anom_multi( lats, lons, maptimes, window_size, timewindow_size, bgccovparams, tscovparams, bgcanom, tsanoms_bgc, tsanoms, bgctime, bgclat, bgclon, tstime, tslat, tslon
) -> np.ndarray:
    batchsize = lats.size
    mapped_anom = np.empty(shape=(batchsize,maptimes.size,2,))
    for i in range(batchsize):
        mapped_anom[i,:,:] = ma.map_anom_multi(
            lats[i], lons[i], maptimes, window_size, timewindow_size, bgccovparams[i], tscovparams[i], bgcanom, tsanoms_bgc, tsanoms, bgctime, bgclat, bgclon, tstime, tslat, tslon
        )
    return mapped_anom
    return None

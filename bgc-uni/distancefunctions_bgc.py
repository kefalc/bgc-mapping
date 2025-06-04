from numba import njit, prange
import numpy as np
import math
from njit_options_bgc import opts

@njit(**opts())
def in_spatial_window(lat0, lon0, lat, lon, window_size):
    """
    Calculate the  distance between two points on the Earth's surface with haversine formula.
    Returns boolean-array for indexing.
    # R_earth * 2 = 6.371e6 * 2 = 12742000.0
    """
    phi1 = math.radians(lat0)
    lambda1 = math.radians(lon0)
    coslat1 = math.cos(phi1)
    n = lat.size
    in_window = np.empty((n,), dtype=np.bool_)    
    for i in prange(n):
        phi2 = math.radians(lat[i])
        a = math.sin((phi2 - phi1)/2.0)**2
        b = coslat1 * math.cos(phi2) * math.sin((lambda1 - math.radians(lon[i]))/2.0)**2
        in_window[i] = (12742000.0 * math.asin( math.sqrt(a+b) )) < window_size
    return in_window

@njit(**opts())
def in_timewindow(time0, time, timewindow_size):
    """
    Returns boolean-array for indexing
    """
    n = time.size
    in_window = np.empty(shape=(n,), dtype=np.bool_)
    for i in prange(n):
        in_window[i] = abs(time0-time[i]) <= timewindow_size
    return in_window

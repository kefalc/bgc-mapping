from numba import njit, prange
from njit_options import opts
import numpy as np
import math

@njit(**opts())
def haversine_distance( lat0, lon0, lat, lon ) -> np.ndarray:
    """
    Calculate the  distance between two points on the Earth's surface with haversine formula.
    formula: https://en.wikipedia.org/wiki/Haversine_formula
    using arcsin, which could suffer from precision loss when distance is small
    """
    # R_earth * 2 = 6.371e6  * 2 = 12742000.0
    phi1 = math.radians(lat0)
    lambda1 = math.radians(lon0)
    coslat1 = math.cos(phi1)
    n = lat.size
    distances = np.empty((n,))
    for i in prange(n):
        phi2 = math.radians(lat[i])
        lambda2 = math.radians(lon[i])
        a = math.sin((phi2 - phi1)/2.0)**2
        b = coslat1 * math.cos(phi2) * math.sin((lambda1 - lambda2)/2.0)**2
        distances[i] = 12742000.0 * math.asin( math.sqrt(a+b) )
    return distances

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
        if (12742000.0 * math.asin( math.sqrt(a+b) )) < window_size:
            in_window[i] = True
        else:
            in_window[i] = False
    return in_window

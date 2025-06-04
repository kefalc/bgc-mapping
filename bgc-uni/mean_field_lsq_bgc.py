import numpy as np
from numba import njit
from njit_options_bgc import opts
from distancefunctions_bgc import *
from basisfunctions_bgc import *


def mean_field_lsq( lat0, lon0, data, lat, lon, time, window_size, NHarm ) -> np.ndarray:
    """Computes parameters for the gridpoint lat0, lon0."""
    
    distances = haversine_distance( lat0, lon0, lat, lon )
    in_window = np.where(distances < window_size)[0]
    data_iw = data[in_window].astype(np.float64)
    lat_iw = lat[in_window]
    lon_iw = lon[in_window]
    time_iw = time[in_window]
    
    num_in_window = in_window.size
    if num_in_window >= 100:
        Mn = basisfunctions_mflsq( num_in_window, lat0, lon0, lat_iw, lon_iw, time_iw, NHarm )
        beta_hat = least_squares_solution(Mn, data_iw)
    
    else: 
        beta_hat = np.full( shape = ( NHarm*2 + 6 ), fill_value = np.nan )
    
    return beta_hat


@njit( **opts() )
def least_squares_solution(Mn, data) -> np.ndarray:
    """ numpy.linalg.lstsq """
    return np.linalg.lstsq(Mn, data)[0]
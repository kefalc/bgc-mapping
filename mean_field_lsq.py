import numpy as np
from numba import njit
from njit_options import opts
from distancefunctions import haversine_distance
from basisfunctions import basisfunctions_mflsq

def batch_mean_field_lsq(lats, lons, data, time, lat, lon, window_size, NHarm, Ndata_min) -> np.ndarray:
    # Loop through batch of coordinates
    batchsize = lats.size
    beta_hats = np.empty(shape=( batchsize, NHarm*2 + 6 ), dtype=np.double)
    for i in range(batchsize):
        beta_hats[i,:] = mean_field_lsq( lats[i], lons[i], data, lat, lon, time, window_size, NHarm, Ndata_min )
    return beta_hats


def batch_mean_field_lsq_2vars( 
    lats, lons, data1, data2, time, lat, lon, window_size1, window_size2, 
    Ndata_min1, Ndata_min2, NHarm ) -> np.ndarray:
    """ 
    F.ex.:
    data1: temperature 
    data2: salinity
    """
    # Loop through batch of coordinates
    batchsize = lats.size
    beta_hats = np.empty(shape=( batchsize, 2, NHarm*2 + 6 ), dtype=np.double)
    for i in range(batchsize):
        ii1 = ~np.isnan(data1)
        ii2 = ~np.isnan(data2)
        beta_hats[i,0,:] = mean_field_lsq( lats[i], lons[i], data1[ii1], lat[ii1], lon[ii1], time[ii1], window_size1, NHarm, Ndata_min1 )
        beta_hats[i,1,:] = mean_field_lsq( lats[i], lons[i], data2[ii2], lat[ii2], lon[ii2], time[ii2], window_size2, NHarm, Ndata_min2 )

    return beta_hats


def mean_field_lsq( lat0, lon0, data, lat, lon, time, window_size, NHarm, 
                   Ndata_min=100 ) -> np.ndarray:
    """Computes parameters for the gridpoint lat0, lon0."""
    
    distances = haversine_distance( lat0, lon0, lat, lon )
    in_window = np.where(distances < window_size)[0]
    data_iw = data[in_window]
    lat_iw = lat[in_window]
    lon_iw = lon[in_window]
    time_iw = time[in_window]
    
    num_in_window = in_window.size
    if num_in_window >= Ndata_min:
        Mn = basisfunctions_mflsq( num_in_window, lat0, lon0, lat_iw, lon_iw, time_iw, NHarm )
        beta_hat = least_squares_solution(Mn, data_iw)
    else: 
        beta_hat = np.full( shape=( NHarm*2 + 6 ), fill_value=np.nan )
    
    return beta_hat


def mean_field_lsq_2vars( lat0, lon0, data1, data2, lat, lon, time, window_size1, window_size2, NHarm, 
                         Ndata_min1, Ndata_min2 ) -> ( np.ndarray, np.ndarray ):
    """ 
    Have the same worker compute both parameters.
    For example:
    data1: temperature 
    data2: salinity
    """
    
    ii1 = ~np.isnan(data1)
    ii2 = ~np.isnan(data2)
    beta_hat_1 = mean_field_lsq( lat0, lon0, data1[ii1], lat[ii1], lon[ii1], time[ii1], window_size1, NHarm, Ndata_min1 )
    beta_hat_2 = mean_field_lsq( lat0, lon0, data2[ii2], lat[ii2], lon[ii2], time[ii2], window_size2, NHarm, Ndata_min2 )
    
    return beta_hat_1, beta_hat_2


@njit( **opts() )
def least_squares_solution(Mn, data) -> np.ndarray:
    """ numpy.linalg.lstsq """
    return np.linalg.lstsq(Mn, data)[0]
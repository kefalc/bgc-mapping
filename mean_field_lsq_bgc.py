import numpy as np
from numba import njit
from njit_options import opts
from distancefunctions import haversine_distance
from basisfunctions_bgc import basisfunctions_mflsq

##update to be able to change 8
def batch_mean_field_lsq(lats, lons, data, time, lat, lon, temp, sal, window_size, NHarm, Ndata_min) -> np.ndarray:
    # Loop through batch of coordinates
    batchsize = lats.size
    beta_hats = np.empty(shape=( batchsize, NHarm*2 + 8 ), dtype=np.double)
    for i in range(batchsize):
        beta_hats[i,:] = mean_field_lsq( lats[i], lons[i], data, lat, lon, temp, sal, time, window_size, NHarm, Ndata_min )
    return beta_hats

##update to be able to change 8
def mean_field_lsq( lat0, lon0, data, lat, lon, temp, sal, time, window_size, NHarm, 
                   Ndata_min=100 ) -> np.ndarray:
    """Computes parameters for the gridpoint lat0, lon0."""
    
    distances = haversine_distance( lat0, lon0, lat, lon )
    in_window = np.where(distances < window_size)[0]
    data_iw = data[in_window]
    lat_iw = lat[in_window]
    lon_iw = lon[in_window]
    temp_iw = temp[in_window]
    sal_iw = sal[in_window]
    time_iw = time[in_window]
    
    #remove few nan measurements so least squares works
    nfind = ~np.isnan(data_iw)
    lat_iw = lat_iw[nfind]
    lon_iw = lon_iw[nfind]
    temp_iw = temp_iw[nfind]
    sal_iw = sal_iw[nfind]
    time_iw = time_iw[nfind]
    in_window = in_window[nfind]
    data_iw = data_iw[nfind]
    
    nfind = ~np.isnan(temp_iw)
    lat_iw = lat_iw[nfind]
    lon_iw = lon_iw[nfind]
    temp_iw = temp_iw[nfind]
    sal_iw = sal_iw[nfind]
    time_iw = time_iw[nfind]
    in_window = in_window[nfind]
    data_iw = data_iw[nfind]
    
    nfind = ~np.isnan(sal_iw)
    lat_iw = lat_iw[nfind]
    lon_iw = lon_iw[nfind]
    temp_iw = temp_iw[nfind]
    sal_iw = sal_iw[nfind]
    time_iw = time_iw[nfind]
    in_window = in_window[nfind]
    data_iw = data_iw[nfind]
    
    num_in_window = in_window.size
    if num_in_window >= Ndata_min:
        Mn = basisfunctions_mflsq( num_in_window, lat0, lon0, lat_iw, lon_iw, temp_iw, sal_iw, time_iw, NHarm )
        beta_hat = least_squares_solution(Mn, data_iw)
    else: 
        beta_hat = np.full( shape=( NHarm*2 + 8), fill_value = np.nan )
    
    return beta_hat
    


@njit( **opts() )
def least_squares_solution(Mn, data) -> np.ndarray:
    """ numpy.linalg.lstsq """
    return np.linalg.lstsq(Mn, data)[0]
import math
import numpy as np
from numba import njit, prange
from njit_options_bgc import opts
from time import time as Time
from scipy.optimize import minimize
from distancefunctions_bgc import *
from covariancefunctions_bgc import *
import negative_loglikelihood_bgc as nll

def optimize_cov_parameters_Nblocks_uni(lat0, lon0, data, lat, lon, time, window_size, dateN, log_x0, Ndata_min, Ndata_lim, numparams, Nblocks, time_i, time_f ) -> (np.ndarray, float, int, str, int, int, int, int, np.ndarray, float): 
    """Estimates parameters for the gridpoint (lat0, lon0)."""
    in_window = in_spatial_window( lat0, lon0, lat, lon, window_size )
    data_iw = data[in_window]
    data_size = data_iw.size
    t_start = Time()
    if data_size >= Ndata_min:
        
        if data_size > Ndata_lim:
            Nblocks = Nblocks * 2
            dateN = np.arange(time_i, time_f+1e-10, (time_f - time_i)/Nblocks)
        fun = get_negLogLik_function( lat[in_window], lon[in_window], time[in_window], data_iw, dateN, Nblocks )
        result = minimize(fun, log_x0, method='L-BFGS-B')
    
        results = np.exp(result.x), result.fun, result.success, result.message, result.status, result.nfev, result.nit, result.njev, np.exp(log_x0), Time()-t_start, Nblocks, data_size 
        del fun
    else:
        results = -1e20*np.ones((numparams)), -1e20, 9, 'not enough data', -999,-999,-999,-999, np.exp(log_x0), Time()-t_start, Nblocks, data_size
    del data, lat, lon, time, data_iw, in_window, time_i, time_f
    return results 

def optimize_cov_parameters_Nblocks_multi( 
    lat0, lon0, data, ts_data, lat, lon, time, window_size, dateN, log_x0, Ndata_min, Ndata_lim, numparams, Nblocks, 
    time_i, time_f, ts_covparam_values, multivar_flag ) -> (np.ndarray, float, int, str, int, int, int, int, np.ndarray, float, float, float): 
    """Estimates parameters for the gridpoint (lat0, lon0)."""
    in_window = in_spatial_window( lat0, lon0, lat, lon, window_size )
    data_iw = data[in_window]
    ts_iw = ts_data[in_window]
    data_size = data_iw.size
    t_start = Time()
    if data_size >= Ndata_min:
       
        if data_size > Ndata_lim:
            Nblocks = Nblocks * 2
            dateN = np.arange(time_i, time_f+1e-10, (time_f - time_i)/Nblocks )
        if multivar_flag==2:
            fun = get_negLogLik_mv_function( ts_covparam_values, lat[in_window], lon[in_window], time[in_window], data_iw, ts_iw, dateN, Nblocks)
            result = minimize(fun, log_x0, method='L-BFGS-B')
        else: raise ValueError("Nothing here yet.")

        
        result_temp = np.full( shape=result.x.shape, fill_value=np.nan )
        #print('result x is ', result.x.shape)
        
        result_temp[0:5] = np.exp(result.x[0:5])
        result_temp[5] = np.tanh(result.x[5])

        #result temp was np.exp(result.x)
        results = result_temp, result.fun, result.success, result.message, result.status, result.nfev, result.nit, result.njev, np.exp(log_x0),  Time()-t_start, Nblocks, data_size
        del fun
    else:
        results = -1e20*np.ones((numparams)), -1e20, 9, 'not enough data', -999,-999,-999,-999, np.exp(log_x0), Time()-t_start, Nblocks, data_size
    del data, lat, lon, time, data_iw, in_window, time_i, time_f
    return results 


def get_negLogLik_function( lat_iw, lon_iw, time_iw, data_iw, dateN, Nblocks ):
    """ Just for readability of 'optimize_cov_parameters_Nblocks()' """
    def fun(params): return nll.negLogLik_MaternSpaceTime_Nblocks_nu32( params, lat_iw, lon_iw, time_iw, data_iw, dateN, Nblocks )
    return fun

def get_negLogLik_mv_function( ts_covparam_values, lat_iw, lon_iw, time_iw, data_iw, ts_iw, dateN, Nblocks ):
    """ Just for readability of 'optimize_cov_parameters_Nblocks()' """
    def fun(params): return nll.negLogLik_MaternSpaceTime_Nblocks_nu32_multi( params, ts_covparam_values, lat_iw, lon_iw, time_iw, data_iw, ts_iw, dateN, Nblocks )
    return fun
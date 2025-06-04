import math
import numpy as np
from numba import njit, prange
from njit_options import opts
from time import time as Time
from scipy.optimize import minimize
from distancefunctions import *
from covariancefunctions import *
import negative_loglikelihood as nll

def optimize_cov_parameters_Nblocks( 
    lat0, lon0, data, lat, lon, time, window_size, dateN, log_x0, Ndata_min, Ndata_lim, numparams, Nblocks, 
    time_i, time_f, nu ) -> (np.ndarray, float, int, str, int, int, int, int, np.ndarray, float):
    """Estimates parameters for the gridpoint (lat0, lon0)."""
    in_window = in_spatial_window( lat0, lon0, lat, lon, window_size )
    data_iw = data[in_window]
    data_size = data_iw.size
    t_start = Time()
    if data_size >= Ndata_min:
        # Alternatively: let Nblocks and log_x0 be a function of data in window
        if data_size > Ndata_lim:
            Nblocks = Nblocks * 2
            dateN = np.arange(
                time_i, time_f+1e-10, (time_f - time_i)/Nblocks 
            )
        fun = get_negLogLik_function( nu, lat[in_window], lon[in_window], time[in_window], data_iw, dateN, Nblocks )
        result = minimize(fun, log_x0, method='L-BFGS-B') #this ends up being params (log_x0)
        results = np.exp(result.x), result.fun, result.success, result.message, result.status, result.nfev, result.nit, result.njev, np.exp(log_x0), Time()-t_start
        del fun
    else:
        results = -1e20*np.ones((numparams)), -1e20, 9, 'not enough data', -999,-999,-999,-999, np.exp(log_x0), Time()-t_start
    del data, lat, lon, time, data_iw, in_window, time_i, time_f
    return results

def optimize_cov_parameters_Nblocks_2vars(
    lat0, lon0, anom1, anom2, lat, lon, time, window_size, dateN, log_x0, Ndata_min, Ndata_lim, numparams, Nblocks, 
    time_i, time_f, nu) -> (
    np.ndarray, float, int, str, int, int, int, int, 
    np.ndarray, float, int, str, int, int, int, int,
    np.ndarray, float, float ):
    """Estimates parameters for 2 variables, for the gridpoint (lat0, lon0)."""
    ii1 = ~np.isnan(anom1)
    ii2 = ~np.isnan(anom2)
    x1, fun1, success1, message1, status1, nfev1, nit1, njev1, log_x0_1, time1 = optimize_cov_parameters_Nblocks(
        lat0, lon0, anom1[ii1], lat[ii1], lon[ii1], time[ii1], window_size, dateN, log_x0[:,0], 
        Ndata_min, Ndata_lim, numparams, Nblocks, time_i, time_f, nu )
    x2, fun2, success2, message2, status2, nfev2, nit2, njev2, log_x0_2, time2 = optimize_cov_parameters_Nblocks( 
        lat0, lon0, anom2[ii2], lat[ii2], lon[ii2], time[ii2], window_size, dateN, log_x0[:,1], 
        Ndata_min, Ndata_lim, numparams, Nblocks, time_i, time_f, nu )
    return x1, fun1, success1, message1, status1, nfev1, nit1, njev1, x2, fun2, success2, message2, status2, nfev2, nit2, njev2, np.exp(log_x0), time1, time2 
    
#can probably simplify this    
def get_negLogLik( nu, lat_iw, lon_iw, time_iw, data_iw, dateN, Nblocks ):
    """ Just for readability of 'optimize_cov_parameters_Nblocks()' """
    if nu == 0.5:
        def fun(params): return negLogLik_MaternSpaceTime_Nblocks_nu12( params, lat_iw, lon_iw, time_iw, data_iw, dateN, Nblocks )
    elif nu == 1.5:
        def fun(params): return negLogLik_MaternSpaceTime_Nblocks_nu32( params, lat_iw, lon_iw, time_iw, data_iw, dateN, Nblocks )
    elif nu == 2.5:
        def fun(params): return negLogLik_MaternSpaceTime_Nblocks_nu52( params, lat_iw, lon_iw, time_iw, data_iw, dateN, Nblocks )
    elif nu == math.inf:
        def fun(params): return negLogLik_MaternSpaceTime_Nblocks_nuInf( params, lat_iw, lon_iw, time_iw, data_iw, dateN, Nblocks )
    elif nu == None:
        def fun(params): return negLogLik_MaternSpaceTime_GenExp_Nblocks( params, lat_iw, lon_iw, time_iw, data_iw, dateN, Nblocks )
    else: 
        raise ValueError('No code for "nu='+str(nu)+'"')
    return fun

#can probably delete most of these and skip this step completely.    
def get_negLogLik_function( nu, lat_iw, lon_iw, time_iw, data_iw, dateN, Nblocks ):
    """ Just for readability of 'optimize_cov_parameters_Nblocks()' """
    if nu == 0.5:
        def fun(params): return nll.negLogLik_MaternSpaceTime_Nblocks_nu12( params, lat_iw, lon_iw, time_iw, data_iw, dateN, Nblocks )
    elif nu == 1.5:
        def fun(params): return nll.negLogLik_MaternSpaceTime_Nblocks_nu32( params, lat_iw, lon_iw, time_iw, data_iw, dateN, Nblocks )
    elif nu == 2.5:
        def fun(params): return nll.negLogLik_MaternSpaceTime_Nblocks_nu52( params, lat_iw, lon_iw, time_iw, data_iw, dateN, Nblocks )
    elif nu == math.inf:
        def fun(params): return nll.negLogLik_MaternSpaceTime_Nblocks_nuInf( params, lat_iw, lon_iw, time_iw, data_iw, dateN, Nblocks )
    elif nu == None:
        def fun(params): return nll.negLogLik_MaternSpaceTime_GenExp_Nblocks( params, lat_iw, lon_iw, time_iw, data_iw, dateN, Nblocks )
    else: 
        raise ValueError('No function for "nu='+str(nu)+'"')
    return fun
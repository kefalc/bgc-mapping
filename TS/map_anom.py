import numpy as np
import numpy.matlib as ml
import math
import covariancefunctions as cov
import distancefunctions as dist
import solvers as solve
import dotproduct as dot
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from numba import njit, prange
from njit_options import opts, not_cache


def map_anom(
    lat0, lon0, maptimes, window_size, timewindow_size, covparam, data, time, lat, lon
) -> np.ndarray:
    """
    lat0, lon0:  single coordinates.
    maptimes:    range of time-coordinates (days since 1970-01-01)
    # Rearth = 6.371e6
    # 2Omega * Rearth = 2*2*math.pi/86400.0 * Rearth = 926.6243887046561
    """
    n_times = maptimes.size
    mapped_anom = np.empty(shape=(n_times,2,))
    if lat0 == 0.0:
        mapped_anom[:,0] = 0.0
        mapped_anom[:,1] = covparam[0]
        return mapped_anom

    # Data in spatial window
    in_sw = dist.in_spatial_window( lat0, lon0, lat, lon, window_size )
    anom_sw = data[in_sw]
    if anom_sw.size > 0:
        
        thetas = covparam[0]
        thetax = covparam[1]
        thetay = covparam[2]
        thetat = covparam[3]
        sigma_squared = covparam[4]**2
        
        time_sw, lat_sw, lon_sw = time[in_sw], lat[in_sw], lon[in_sw]
        f0R = math.sin(math.radians(lat0)) * 926.6243887046561
        
        for ntime in range(n_times):
            time0 = maptimes[ntime]
            in_tw = dist.in_timewindow( time0, time_sw, timewindow_size)            
            anom_stw = anom_sw[in_tw]
            Ndata = anom_stw.size
            
            if Ndata == 0:
                mapped_anom[ntime,0] = 0.0
                mapped_anom[ntime,1] = thetas
                
            elif Ndata > 0:
                ##might need to change the function here
                Ktheta, data_and_covPt = cov.Matern_SpaceTimeCovariance_Nblocks_Ktheta_covPt_scalardata_nu32(
                    Ndata, lat0, lon0, time0, lat_sw[in_tw], lon_sw[in_tw], time_sw[in_tw], 
                    thetas, thetax, thetay, thetat, sigma_squared, anom_stw
                )
                L, INFO = solve.dpotrf( Ktheta, Ndata ) # Overwrites Ktheta
                if INFO > 0: # check for spd
                    raise ValueError('INFO = '+str(INFO))
                Cd, Ced = dot.covPt_scalardata( 
                    Ndata, 
                    data_and_covPt[1,:], 
                    solve.dpotrs_b2( L, data_and_covPt.copy(), Ndata )
                )
                mapped_anom[ntime, 0] = Cd
                mapped_anom[ntime, 1] = thetas - Ced

    else:
        mapped_anom[:,0] = 0.0
        mapped_anom[:,1] = covparam[0]
           
    return mapped_anom

@njit(**not_cache())
def map_anom_seasonal_multi(
    lat0, lon0, maptimes, window_size, timewindow_size, covparam, uanoms, vanoms, time, lat, lon
)-> np.ndarray:
    """    
    lat0, lon0: single coordinates,   (float, float) [degrees North], [degrees East]
    """

    maptimes_size = maptimes.size
    timeseries = np.empty((maptimes_size, 2, 2 ))
    
    if lat0 == 0.0: 
        timeseries[:,0,0] = 0.0
        timeseries[:,0,1] = np.nan
        timeseries[:,1,0] = 0.0
        timeseries[:,1,1] = np.nan
        return timeseries
    
    thetas = covparam[0]
    thetax_squared = covparam[1]**2
    thetay_squared = covparam[2]**2
    thetat_squared = covparam[3]**2
    uu_sigma_squared = covparam[4]**2
    vv_sigma_squared = covparam[5]**2
    
    rlat0 = math.radians(lat0)
    f0 = TwoOmega * math.sin(rlat0)
    coslat0 = math.cos(rlat0)
    prior_var_u = 3.0 * thetas / thetay_squared / f0**2 * unitconversion_2 + uu_sigma_squared
    prior_var_v = 3.0 * thetas / thetax_squared / f0**2 * unitconversion_2 / coslat0**2 + vv_sigma_squared
    
    # Data in spatial-window
    in_sw = dist.in_spatial_window( lat0, lon0, lat, lon, window_size )
    uanom_sw = uanoms[in_sw]
    if uanom_sw.size > 0:
    
        vanom_sw = vanoms[in_sw]
        lat_sw, lon_sw, time_sw = lat[in_sw], lon[in_sw], time[in_sw]
        
        for ntime in range(maptimes_size):
            time0 = maptimes[ntime]
            in_tw = dist.in_timewindow(time0, time_sw, timewindow_size)
            uanom_stw = uanom_sw[in_tw]
            Ndata = uanom_stw.size
            if Ndata == 0:
                timeseries[ntime,0,0] = 0.0
                timeseries[ntime,0,1] = prior_var_u
                timeseries[ntime,1,0] = 0.0
                timeseries[ntime,1,1] = prior_var_v
                
            elif Ndata > 0:
                Ktheta, b = cov.Matern_nu72_Ktheta_k_uu_uv_vu_vv_geoageo(
                    Ndata, lat0, lon0, time0, f0, coslat0, 
                    lat_sw[in_tw], lon_sw[in_tw], time_sw[in_tw], uanom_stw, vanom_sw[in_tw],
                    thetas, thetax_squared, thetay_squared, thetat_squared, uu_sigma_squared, vv_sigma_squared
                )
                TwoN = Ndata*2                                
                L, INFO = solve.dpotrf(Ktheta, TwoN) # Overwrites Ktheta
                if INFO != 0:
                    timeseries[:,:,:] = np.nan
                    return timeseries
                x, xINFO = solve.dpotrs(L, b.copy(), 3, TwoN) # Overwrites b
                if xINFO != 0:
                    timeseries[:,:,:] = np.nan
                    return timeseries
                pred_u, pred_v, red_var_u, red_var_v = dot.pred_uv(TwoN, b[1:,:], x)
                timeseries[ntime,0,0] = pred_u
                timeseries[ntime,0,1] = prior_var_u - red_var_u
                timeseries[ntime,1,0] = pred_v
                timeseries[ntime,1,1] = prior_var_v - red_var_v

    else:
        timeseries[:,0,0] = 0.0
        timeseries[:,0,1] = prior_var_u
        timeseries[:,1,0] = 0.0
        timeseries[:,1,1] = prior_var_v
        
    return timeseries

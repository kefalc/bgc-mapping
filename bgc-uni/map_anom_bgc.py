import numpy as np
import numpy.matlib as ml
import math
import covariancefunctions_bgc as cov
import distancefunctions_bgc as dist
import solvers_bgc as solve
import dotproduct_bgc as dot
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from numba import njit, prange
from njit_options_bgc import opts, not_cache


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
        #f0R = math.sin(math.radians(lat0)) * 926.6243887046561
        
        for ntime in range(n_times):
            time0 = maptimes[ntime]
            in_tw = dist.in_timewindow( time0, time_sw, timewindow_size)            
            anom_stw = anom_sw[in_tw]
            Ndata = anom_stw.size
            
            if Ndata == 0:
                mapped_anom[ntime,0] = 0.0
                mapped_anom[ntime,1] = thetas
                
            elif Ndata > 0:
                
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

def map_anom_multi(lat0, lon0, maptimes, window_size, timewindow_size, bgccovparam, tscovparam, bgcanoms, tsanoms_bgc, tsanoms, bgctime, bgclat, bgclon, tstime, tslat, tslon )-> np.ndarray:

    """    
    lat0, lon0: single coordinates,   (float, float) [degrees North], [degrees East]
    """
    maptimes_size = maptimes.size
    timeseries = np.empty((maptimes_size, 2 ))

    if lat0 == 0.0: 
        timeseries[:,0] = 0.0
        timeseries[:,1] = np.nan
        return timeseries
        
    ##for TS
    thetasTS = tscovparam[0]
    thetaxTS = tscovparam[1]
    thetayTS = tscovparam[2]
    thetatTS = tscovparam[3]
    sigmaTS_squared = tscovparam[4]**2
    
    ##for BGC
    thetasB = bgccovparam[0]
    thetaxB = bgccovparam[1]
    thetayB = bgccovparam[2]
    thetatB = bgccovparam[3]
    sigmaB_squared = bgccovparam[4]**2
    beta = bgccovparam[5]

    ############## define prior_var_bgc
    #prior_var_bgc = thetasB/(thetaxB*thetayB*thetatB)*math.sqrt(math.pi / 2)
    prior_var_bgc = thetasB
    var_factor = 1 / (thetaxB*thetayB*thetatB)

    # Data in spatial-window - TS
    in_sw = dist.in_spatial_window( lat0, lon0, bgclat, bgclon, window_size )
    bgcanom_sw = bgcanoms[in_sw]
    in_sw_ts = dist.in_spatial_window( lat0, lon0, tslat, tslon, window_size )
    tsanom_sw = tsanoms[in_sw_ts]
    if tsanom_sw.size > 0:
        tsanom_bgc_sw = tsanoms_bgc[in_sw]
        bgclat_sw = bgclat[in_sw]
        bgclon_sw = bgclon[in_sw]
        bgctime_sw = bgctime[in_sw]
        tslat_sw, tslon_sw, tstime_sw = tslat[in_sw_ts], tslon[in_sw_ts], tstime[in_sw_ts]

        for ntime in range(maptimes_size):
            time0 = maptimes[ntime]
            in_tw_ts = dist.in_timewindow(time0, tstime_sw, timewindow_size)
            in_tw_bgc = dist.in_timewindow(time0, bgctime_sw, timewindow_size)
            bgcanom_stw = bgcanom_sw[in_tw_bgc]
            tsanom_bgc_stw = tsanom_bgc_sw[in_tw_bgc]
            tsanom_stw = tsanom_sw[in_tw_ts]
            nbgc = bgcanom_stw.size
            nts = tsanom_stw.size
            Ndata = 2*nbgc + nts
            if Ndata == 0:
                timeseries[ntime,0] = 0.0
                timeseries[ntime,1] = prior_var_bgc * var_factor
            elif Ndata > 0:
                Ktheta, b = cov.Matern_SpaceTimeCovariance_Nblocks_Ktheta_covPt_scalardata_nu32_multi( nbgc, nts, lat0, lon0, time0, bgclat_sw[in_tw_bgc], bgclon_sw[in_tw_bgc], bgctime_sw[in_tw_bgc], bgcanom_stw, tsanom_bgc_stw, tslat_sw[in_tw_ts], tslon_sw[in_tw_ts], tstime_sw[in_tw_ts], tsanom_stw, thetasTS, thetaxTS, thetayTS, thetatTS, sigmaTS_squared, thetasB, thetaxB, thetayB, thetatB, sigmaB_squared, beta)                             

                L, INFO = solve.dpotrf(Ktheta, Ndata) # Overwrites Ktheta
                if INFO != 0:
                    timeseries[:,:] = np.nan
                    return timeseries

                # x, xINFO = solve.dpotrs(L, b.copy(), 2, Ndata) # Overwrites b
                # if xINFO != 0:
                #     timeseries[:,:] = np.nan
                #     return timeseries
                #pred_bgc, pred_var_bgc = dot.pred_multi(Ndata, b[1:,:], x)
                pred_bgc, pred_var_bgc = dot.covPt_scalardata(Ndata, b[1,:], solve.dpotrs_b2( L, b.copy(), Ndata ))
                timeseries[ntime,0] = pred_bgc
                timeseries[ntime,1] = (prior_var_bgc - pred_var_bgc)*var_factor
                
    else:
        timeseries[:,0] = 0.0
        timeseries[:,1] = prior_var_bgc
        
    return timeseries
import numpy as np
from time import time as Time
import convenient_bgc as cv
from basisfunctions_bgc import *
from mean_field_lsq_bgc import *
from covparam_window_averaging_bgc import *
from maximum_likelihood_bgc import *
from savefunctions_bgc import *
from batch_map_anom_bgc import *

#update to bgc covparam, will need to add inputs for t&S.
#get rid of nu related inputs since it is not changing
def bgc_covparam( plevel, datafield, bgcresidualfile, tsresidualfile, maskfile, tscovparamfile, outfile, window_size, Nblocks, time_i, time_f, client, multivar_flag, param_flag, NWorkers=None, Ndata_min=100, Ndata_lim=math.inf) -> None:
    if multivar_flag == 1: ##UNIVARIATE CASE
        # Load data, mask
        bgcres, bgclat, bgclon, bgctime, b = cv.get_residualdata_bgc(bgcresidualfile)
            ##update this to get rid of nans
        lats, lons, mask, latg, long = cv.get_compressed_mask(maskfile)
    
        #Initial Parameters
        log_x0 = cv.get_initial_covparams(datafield, mask)
        numparams = log_x0.shape[2] # Number of parameters to estimate
        dateN = cv.get_dateN(time_i, time_f, Nblocks)
    
    
        #Scatter Data to workers
        workerlist = cv.get_workerlist(client, NWorkers)
        bgcres_sc = client.scatter(bgcres, broadcast=False) #, workers=workerlist)
        bgclat_sc = client.scatter(bgclat, broadcast=False) #, workers=workerlist)
        bgclon_sc = client.scatter(bgclon, broadcast=False) #, workers=workerlist)
        bgctime_sc = client.scatter(bgctime, broadcast=False) #, workers=workerlist)

        print(str(b.pressure[:].sum())+' dbar pressurelevel (pressureindex '+str(plevel)+'):', end=' ')
        list_of_submitted_tasks = []
        t_start = Time()
        iM, jM = np.where(mask.mask)
        for i, j, lat0, lon0 in zip(iM, jM, lats, lons):
            submitted = client.submit(optimize_cov_parameters_Nblocks_uni,
                lat0, lon0, bgcres_sc, bgclat_sc, bgclon_sc, bgctime_sc, 
                window_size, dateN, log_x0[i, j, :], Ndata_min, Ndata_lim, numparams, Nblocks, 
                time_i, time_f,
                pure = False,
                workers = workerlist 
            )
            list_of_submitted_tasks.append( submitted )
        covparams = client.gather(list_of_submitted_tasks)
        save_covparams_bgc( covparams, outfile, plevel, b, mask, latg, long, window_size, Nblocks, t_start, numparams)
        print('Covariance-parameters saved to outfile.')
        del bgcres_sc, bgctime_sc, bgclat_sc, bgclon_sc, covparams, list_of_submitted_tasks, submitted

    if multivar_flag == 2: ##MULTIVARIATE CASE W/O NOISE
         # Load data, mask
        bgcres, bgclat, bgclon, bgctime, b = cv.get_residualdata_bgc(bgcresidualfile)
        lats, lons, mask, latg, long = cv.get_compressed_mask(maskfile)
        ##input TS residual files and TS covparams.
        rest, ress, tslat, tslon, tstime, a = cv.get_residualdata_2vars(tsresidualfile)
        clat, clon, tscovparam_values, c = cv.get_covparams_2vars(tscovparamfile, param_flag)
    
        #Initial Parameters
        if param_flag == 1: #Temperature
            initB = np.corrcoef(bgcres,rest)
            initB = initB[0,1] #might need to be .tolist()
        elif param_flag == 2: #Salinity
            initB = np.corrcoef(bgcres,ress)
            initB = initB[0,1]
        else: raise ValueError("Incorrect flag.")
        
        log_x0 = cv.get_initial_covparams_multi(datafield, multivar_flag, initB, mask)
        numparams = log_x0.shape[2] # Number of parameters to estimate
        dateN = cv.get_dateN(time_i, time_f, Nblocks)
    
    
        #Scatter Data to workers
        workerlist = cv.get_workerlist(client, NWorkers)
        bgcres_sc = client.scatter(bgcres, broadcast=False) #, workers=workerlist)
        bgclat_sc = client.scatter(bgclat, broadcast=False) #, workers=workerlist)
        bgclon_sc = client.scatter(bgclon, broadcast=False) #, workers=workerlist)
        bgctime_sc = client.scatter(bgctime, broadcast=False) #, workers=workerlist)

        ##Scatter T&S depending on paramflag
        tslat_sc = client.scatter(tslat, broadcast=False)
        tslon_sc = client.scatter(tslon, broadcast=False)
        tstime_sc = client.scatter(tstime, broadcast=False)
        ts_covparam_sc = client.scatter(tscovparam_values, broadcast=False)
        if param_flag == 1: #Temperature
            res_ts_sc = client.scatter(rest, broadcast=False)
        elif param_flag == 2: #Salinity
            res_ts_sc = client.scatter(ress, broadcast=False)
        else: raise ValueError("Incorrect flag.")
            
        
        print(str(b.pressure[:].sum())+' dbar pressurelevel (pressureindex '+str(plevel)+'):', end=' ')
        list_of_submitted_tasks = []
        t_start = Time()
        iM, jM = np.where(mask.mask)
        for i, j, lat0, lon0 in zip(iM, jM, lats, lons):
            submitted = client.submit(optimize_cov_parameters_Nblocks_multi,
                lat0, lon0, bgcres_sc, res_ts_sc, bgclat_sc, bgclon_sc, bgctime_sc, 
                window_size, dateN, log_x0[i, j, :], Ndata_min, Ndata_lim, numparams, Nblocks,  
                time_i, time_f, tscovparam_values[i,j,:].values.tolist(), multivar_flag,
                pure = False,
                workers = workerlist 
            )
            list_of_submitted_tasks.append( submitted )
        covparams = client.gather(list_of_submitted_tasks)
        save_covparams_bgc( covparams, outfile, plevel, b, mask, latg, long, window_size, Nblocks, t_start, numparams)
        print('Covariance-parameters saved to outfile.')
        del bgcres_sc, bgctime_sc, bgclat_sc, bgclon_sc, covparams, list_of_submitted_tasks, submitted
    
#    else: raise ValueError("Nothing here yet.")

def grid_anom_seasonal_bgc( 
    plevel, covparamfile, residualfile, maskfile, window_size, timewindow_size, time_i, time_f, outfile, client, 
    interval_days=5, batchsize=10, NWorkers=None, 
) -> None:

    maptimes = np.arange(time_i+interval_days/2.0, time_f, interval_days)
    lats, lons, mask, latg, long = cv.get_compressed_mask(maskfile)
    anom1, lat, lon, time, a = cv.get_residualdata_bgc(residualfile)
    clat, clon, covparam_values, c = cv.get_covparams_bgc(covparamfile)
    covparam_1_interpolated = cv.get_interpolated_parameters( covparam_values[:,:,:,0], clat, clon, lats, lons )
   # covparam_2_interpolated = cv.get_interpolated_parameters( covparam_values[:,:,:,1], clat, clon, lats, lons )

    # Distribute data to workers
    workerlist = cv.get_workerlist(client, NWorkers)
    anom1_sc = client.scatter(anom1, broadcast=False) #, workers=workerlist)
    #anom2_sc = client.scatter(anom2, broadcast=False) #, workers=workerlist)
    time_sc = client.scatter(time, broadcast=False) #, workers=workerlist) 
    lat_sc = client.scatter(lat, broadcast=False) # , workers=workerlist) 
    lon_sc = client.scatter(lon, broadcast=False) # , workers=workerlist) 
    
    list_of_submitted_tasks = []
    for k in range(0, lats.size, batchsize):
        submitted = client.submit( batch_map_anom,
            lats[k:k+batchsize], lons[k:k+batchsize], maptimes, window_size, timewindow_size, 
            covparam_1_interpolated[k:k+batchsize,:], 
            anom1_sc, time_sc, lat_sc, lon_sc,
            pure = False,
            workers = workerlist
        )
        list_of_submitted_tasks.append( submitted )
    save_grid_anom_seasonal_bgc(
        cv.get_from_batches(client.gather(list_of_submitted_tasks), batchsize),
        c, a, mask, latg, long, maptimes, outfile, plevel, window_size, timewindow_size, interval_days
    )
    print(' Gridded small scale estimates on '+str(interval_days)+'-day intervals saved to outfile.')
    del submitted, list_of_submitted_tasks, anom1_sc, time_sc, lat_sc, lon_sc


def grid_anom_seasonal_multi(
    plevel, param_flag, bgccovparamfile, tscovparamfile, bgcresidualfile, tsresidualfile, tsonly_residualfile, maskfile, window_size, timewindow_size, time_i, time_f, outfile, client, 
    interval_days=5, batchsize=10, NWorkers=None, residual_largeval=np.inf, print_text=True
) -> None:
    """
    Batches must align with chunks when storing results to region. Chose to chunk by longitude here.
    rechunk() can be applied after computation if necessary.
    """
    maptimes = np.arange( time_i+interval_days/2, time_f, interval_days )
    lats, lons, mask, latg, long = cv.get_compressed_mask(maskfile)
    iM, jM = np.where(mask)
    clat, clon, covparam_values, c = cv.get_covparams_bgc(bgccovparamfile)
    #bgccovparam_interpolated = np.full(shape=(latg.size, long.size, covparam_values.shape[2]), fill_value=np.nan)
    bgccovparam_interpolated = cv.get_interpolated_parameters(covparam_values[:,:,:,0], clat, clon, lats, lons)
    workerlist = cv.get_workerlist(client, NWorkers)

    bgcanom, lat, lon, time, b = cv.get_residualdata_bgc(bgcresidualfile)
    rest, ress, tslat, tslon, tstime, a = cv.get_residualdata_2vars(tsresidualfile)
    rest_only, ress_only, tslat_only, tslon_only, tstime_only, a2 = cv.get_residualdata_2vars(tsonly_residualfile)

    clat, clon, tscovparam_values, c = cv.get_covparams_2vars(tscovparamfile, param_flag)
    #tscovparam_interpolated = np.full(shape=(latg.size, long.size, tscovparam_values.shape[2]), fill_value=np.nan)
    tscovparam_interpolated = cv.get_interpolated_parameters(tscovparam_values, clat, clon, lats, lons)
    
    bgcanom_sc = client.scatter(bgcanom)
    if param_flag == 1:
        tsanoms_bgc_sc = client.scatter(rest)
        tsanoms_sc = client.scatter(rest_only)
    else:
        tsanoms_bgc_sc = client.scatter(ress)
        tsanoms_sc = client.scatter(ress_only)
    bgctime_sc = client.scatter(time)
    bgclat_sc = client.scatter(lat)
    bgclon_sc = client.scatter(lon)
    tslat_sc = client.scatter(tslat_only)
    tslon_sc = client.scatter(tslon_only)
    tstime_sc = client.scatter(tstime_only)
    
    list_of_submitted_tasks = []
    for k in range(0, lats.size, batchsize):
        submitted = client.submit( batch_map_anom_multi,
            lats[k:k+batchsize], lons[k:k+batchsize], maptimes, window_size, timewindow_size, 
            bgccovparam_interpolated[k:k+batchsize,:], tscovparam_interpolated[k:k+batchsize,:],
            bgcanom_sc, tsanoms_bgc_sc, tsanoms_sc, bgctime_sc, bgclat_sc, bgclon_sc, tstime_sc, tslat_sc, tslon_sc,
            pure = False,
            workers = workerlist
        )
        
        list_of_submitted_tasks.append( submitted )
    save_grid_anom_seasonal_bgc(
        cv.get_from_batches(client.gather(list_of_submitted_tasks), batchsize),
        c, a, mask, latg, long, maptimes, outfile, plevel, window_size, timewindow_size, interval_days
    )
    print(' Gridded small scale estimates on '+str(interval_days)+'-day intervals saved to outfile.')
    del submitted, list_of_submitted_tasks, bgctime_sc, bgclat_sc, bgclon_sc
  
    #sleep(4.5) # Give client time to delete futures on all workers.
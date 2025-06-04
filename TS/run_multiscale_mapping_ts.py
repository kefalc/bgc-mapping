import numpy as np
from time import time as Time
import convenient as cv
from basisfunctions import *
from mean_field_lsq import *
from covparam_window_averaging import *
from maximum_likelihood import *
from savefunctions import *
from batch_map_anom import *


def ts_covparam( plevel, residualfile, maskfile, outfile, window_size, Nblocks, time_i, time_f, client, 
    nu=3/2, initcovparamfile=None, ic=None, NWorkers=None, Ndata_min=100, Ndata_lim=math.inf, Opt_nu=False ) -> None:
    
    # Load data, mask
    res1, res2, lat, lon, time, a = cv.get_residualdata_2vars(residualfile) #specify T&S here?
    lats, lons, mask, latg, long = cv.get_compressed_mask(maskfile)
    
    # Initial parameters
    log_x0, ic = cv.get_initial_covparams(initcovparamfile, 'TS', mask)
    numparams = log_x0.shape[2] # Number of parameters to estimate
    dateN = cv.get_dateN(time_i, time_f, Nblocks)
    
    # Scatter data to workers
    workerlist = cv.get_workerlist(client, NWorkers)
    res1_sc = client.scatter(res1, broadcast=False) #, workers=workerlist)  
    res2_sc = client.scatter(res2, broadcast=False) #, workers=workerlist)  
    time_sc = client.scatter(time, broadcast=False) #, workers=workerlist)
    lat_sc = client.scatter(lat, broadcast=False) #, workers=workerlist)
    lon_sc = client.scatter(lon, broadcast=False) #, workers=workerlist)
    
    print(str(a.pressure[:].sum())+' dbar pressurelevel (pressureindex '+str(plevel)+'):', end=' ')
    list_of_submitted_tasks = []
    t_start = Time()
    iM, jM = np.where(mask.mask)
    for i, j, lat0, lon0 in zip(iM, jM, lats, lons):
        submitted = client.submit(optimize_cov_parameters_Nblocks_2vars,
            lat0, lon0, res1_sc, res2_sc, lat_sc, lon_sc, time_sc, 
            window_size, dateN, log_x0[i, j, :, :], Ndata_min, Ndata_lim, numparams, Nblocks,
            time_i, time_f, nu,
            pure = False,
            workers = workerlist 
        )
        list_of_submitted_tasks.append( submitted )
    covparams = client.gather(list_of_submitted_tasks)
    save_covparams_2vars( covparams, outfile, plevel, a, mask, latg, long, window_size, Nblocks, nu, Time()-t_start, ic, numparams)
    print('Covariance-parameters saved to outfile.')
    del res1_sc, res2_sc, time_sc, lat_sc, lon_sc, covparams, list_of_submitted_tasks, submitted
    # """ 
    # When running run_grid_anom_seasonal_TS() directly after run_TS_covparam(), 
    # waiting a few seconds helps avoid CancelledError for some reason.
    # """
    # sleep(10.0)
    

def grid_anom_seasonal_ts( 
    plevel, covparamfile, residualfile, maskfile, window_size, timewindow_size, time_i, time_f, outfile, client, 
    interval_days=5, batchsize=10, NWorkers=None, 
) -> None:

    maptimes = np.arange(time_i+interval_days/2.0, time_f, interval_days)
    lats, lons, mask, latg, long = cv.get_compressed_mask(maskfile)
    res1, res2, lat, lon, time, a = cv.get_residualdata_2vars(residualfile)
    clat, clon, covparam_values, c = cv.get_covparams_2vars(covparamfile)
    covparam_1_interpolated = cv.get_interpolated_parameters( covparam_values[:,:,:,0], clat, clon, lats, lons )
    covparam_2_interpolated = cv.get_interpolated_parameters( covparam_values[:,:,:,1], clat, clon, lats, lons )

    # Distribute data to workers
    workerlist = cv.get_workerlist(client, NWorkers)
    res1_sc = client.scatter(res1, broadcast=False) #, workers=workerlist)
    res2_sc = client.scatter(res2, broadcast=False) #, workers=workerlist)
    time_sc = client.scatter(time, broadcast=False) #, workers=workerlist) 
    lat_sc = client.scatter(lat, broadcast=False) # , workers=workerlist) 
    lon_sc = client.scatter(lon, broadcast=False) # , workers=workerlist) 
    
    list_of_submitted_tasks = []
    for k in range(0, lats.size, batchsize):
        submitted = client.submit( batch_map_anom_2vars,
            lats[k:k+batchsize], lons[k:k+batchsize], maptimes, window_size, timewindow_size, 
            covparam_1_interpolated[k:k+batchsize,:], covparam_2_interpolated[k:k+batchsize,:], 
            res1_sc, res2_sc, time_sc, lat_sc, lon_sc,
            pure = False,
            workers = workerlist
        )
        list_of_submitted_tasks.append( submitted )
    save_grid_anom_seasonal_2vars(
        cv.get_from_batches(client.gather(list_of_submitted_tasks), batchsize),
        c, a, mask, latg, long, maptimes, outfile, plevel, window_size, timewindow_size, interval_days
    )
    print(' Gridded small scale estimates on '+str(interval_days)+'-day intervals saved to outfile.')
    del submitted, list_of_submitted_tasks, res1_sc, res2_sc, time_sc, lat_sc, lon_sc
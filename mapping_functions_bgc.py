import numpy as np
import math
import xarray as xr
from time import time as Time, sleep
import convenient_bgc as cv
import bgc_savefunctions as BGCsave
import bgc_savefunctions as save 
import basisfunctions_bgc as bf
import mean_field_lsq_bgc as mfl
import distancefunctions as dist

    
def bgc_betaparam( 
    datafield, plevel, bgcdatafile, tsmeanfile, outfile, maskfile, window_size1, NHarm, client, 
    NWorkers=None, Ndata_min1=100, batchsize=7 ) -> None:
    
    # Load data, mask
    BGClat, BGClon, BGCtime, temp, salt, bgc_param, BGCd = cv.get_BGCdata(bgcdatafile, datafield, plevel)
    
    TSlat, TSlon, TStime, mean_temp, res_temp, mean_salt, res_salt, m = cv.get_meandata(tsmeanfile)
    
    lats, lons, mask, latg, long = cv.get_compressed_mask(maskfile)
    
    pressure = np.int64(BGCd.pressure.values)
    
    # Scatter data to workers
    workerlist = cv.get_workerlist(client, NWorkers)
    bgc_param_sc = client.scatter(bgc_param, broadcast=False) #, workers=workerlist)
    time_sc = client.scatter(BGCtime, broadcast=False) #, workers=workerlist)  
    lat_sc = client.scatter(BGClat, broadcast=False) #, workers=workerlist)  
    lon_sc = client.scatter(BGClon, broadcast=False) #, workers=workerlist)  
    temp_sc = client.scatter(mean_temp, broadcast=False) #,workers=workerlist)
    sal_sc = client.scatter(mean_salt, broadcast=False) #,workers=workerlist)
    
    print(str(pressure)+' dbar pressurelevel (pressureindex '+str(plevel)+'):', end=' ')
    list_of_submitted_tasks = []
    t_start = Time()
    for k in range( 0, lats.size, batchsize ):
        submitted = client.submit( mfl.batch_mean_field_lsq,                                   
            lats[k:k+batchsize], lons[k:k+batchsize],
            bgc_param_sc, time_sc, lat_sc, lon_sc, temp_sc, sal_sc,
            window_size1, NHarm, Ndata_min1,
            pure = False,
            workers = workerlist
        )
        list_of_submitted_tasks.append( submitted )
    save.save_betaparams( 
        cv.get_from_batches(client.gather(list_of_submitted_tasks), batchsize),
        outfile, bgcdatafile, plevel, pressure, mask, latg, long, window_size1, NHarm, Ndata_min1, Time() - t_start
    )
    print('Betaparameters saved to outfile.')
    del submitted, list_of_submitted_tasks, temp_sc, sal_sc, bgc_param_sc, time_sc, lat_sc, lon_sc
    
    
def interpolate_to_observations_bgc(datafield, bgcdatafile, betaparamfile, tsmeanfile, outfile, plevel ) -> None:
    """
    Interpolates large scale parameters to BGC parameter profile-coordinates and computes 
    their mean and residuals (anomalies) on those coordinates. Saves to outfile.
    """
    # Load data and betaparameters
    TSlat, TSlon, TStime, mean_temp, res_temp, mean_salt, res_salt, m = cv.get_meandata(tsmeanfile)
    
    BGClat, BGClon, BGCtime, bgctemp, bgcsal, bgc, d = cv.get_BGCdata(bgcdatafile, datafield, plevel)

    blat, blon, betaparams_bgc, b = cv.get_betaparams_bgc(betaparamfile)
    
    beta_bgc_interpolated = cv.get_interpolated_parameters( betaparams_bgc, blat, blon, BGClat, BGClon )
  
    # Calculating mean and residuals (anomalies)
    NHarm = b.attrs["number_of_harmonics"]
    Mn = bf.basisfunctions_mean( BGCtime, mean_temp, mean_salt, NHarm )
    
    ##Calculate for BGC
    mean_bgc = np.sum(Mn * beta_bgc_interpolated, axis=1 )
    residuals_bgc = bgc - mean_bgc
    
    BGCsave.save_BGC_mean_res( datafield, plevel, mean_bgc, residuals_bgc, BGCtime, BGClat, BGClon, b, d, NHarm, outfile )
    print(' Mean and residuals calculated and saved to outfile.')
    del  betaparams_bgc, Mn, TStime, TSlat, TSlon, mean_temp, mean_salt, res_temp, res_salt, bgcsal, bgctemp, bgc
    
    
def grid_mean_seasonal_BGC( 
    plevel, datafield, maskfile, betaparamfile, tsmeanfile, outfile, datafile, time_i, time_f, window_size,
    interval_days=5.0, Ndata_min=1) -> None:
    
    lats, lons, mask, latg, long = cv.get_compressed_mask(maskfile)
    blat, blon, betaparam_1, b = cv.get_betaparams_bgc(betaparamfile)
    BGClat, BGClon, _, _, _, _, d = cv.get_BGCdata(datafile, datafield, plevel)
    TSlat, TSlon, TStime, mean_temp, mean_salt, grid = cv.get_meangrid_TS(tsmeanfile)
    
    NHarm = b.attrs["number_of_harmonics"]
    pressure = b.pressure[:]
    maptimes =  np.arange(time_i+interval_days/2.0, time_f, interval_days)
    n_times = maptimes.size
    
    #Interpolation to coordinates
    beta_1_interpolated = cv.get_interpolated_parameters( betaparam_1, blat, blon, lats, lons )
    mean_bgc = np.full(shape=(latg.size, long.size, n_times, 1), fill_value=np.nan)
    
    iM,jM = np.where(mask)
    for n, i, j in zip(range(lats.size), iM, jM):
        
        mtemp = np.squeeze(mean_temp[i, j, :, 0])
        msalt = np.squeeze(mean_salt[i, j, :, 0])

        Mn = bf.basisfunctions_alt(n_times, maptimes, mtemp, msalt, NHarm)
        mean_bgc[i,j,:,0] = np.sum(Mn * beta_1_interpolated[n,:], axis=1)

    BGCsave.save_grid_mean_seasonal_bgc( datafield, mean_bgc, maptimes, latg, long, pressure, NHarm, b, d, outfile )
    print(' Gridded estimated '+str(int(interval_days))+'-day mean saved to outfile.')
    del beta_1_interpolated, betaparam_1, blat, blon, lats, lons, mask, iM, jM
    
    
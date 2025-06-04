import numpy as np
import math
import xarray as xr
from time import time as Time, sleep
import convenient as cv
import TS_savefunctions as TSsave
import basisfunctions as bf
import mean_field_lsq as mfl
import distancefunctions as dist


def ts_betaparam( 
    plevel, datafile, outfile, maskfile, window_size1, window_size2, NHarm, client, 
    NWorkers=None, Ndata_min1=100, Ndata_min2=100, batchsize=7 ) -> None:
    
    # Load data, mask
    TSlat, TSlon, TStime, temp, salt, TSd = cv.get_TSdata(datafile, plevel)
    
    lats, lons, mask, latg, long = cv.get_compressed_mask(maskfile)
    pressure = np.int64(TSd.pressure.values) 
    
    # Scatter data to workers
    workerlist = cv.get_workerlist(client, NWorkers)
    temp_sc = client.scatter(temp, broadcast=False) #, workers=workerlist) 
    salt_sc = client.scatter(salt, broadcast=False) #, workers=workerlist) 
    time_sc = client.scatter(TStime, broadcast=False) #, workers=workerlist) 
    lat_sc = client.scatter(TSlat, broadcast=False) #, workers=workerlist)  
    lon_sc = client.scatter(TSlon, broadcast=False) #, workers=workerlist)  
    
    print(str(pressure)+' dbar pressurelevel (pressureindex '+str(plevel)+'):', end=' ')
    list_of_submitted_tasks = []
    t_start = Time()
    for k in range( 0, lats.size, batchsize ):
        submitted = client.submit( mfl.batch_mean_field_lsq_2vars,                                  
            lats[k:k+batchsize], lons[k:k+batchsize],
            temp_sc, salt_sc, time_sc, lat_sc, lon_sc,
            window_size1, window_size2, Ndata_min1, Ndata_min2, NHarm,
            pure = False,
            workers = workerlist
        )
        list_of_submitted_tasks.append( submitted )
    TSsave.save_betaparams_2vars( 
        cv.get_from_batches(client.gather(list_of_submitted_tasks), batchsize),
        outfile, plevel, pressure, mask, latg, long, window_size1, window_size2, NHarm, Time() - t_start
    )
    print('Betaparameters saved to outfile.')
    del submitted, list_of_submitted_tasks, temp_sc, salt_sc, time_sc, lat_sc, lon_sc
    
    
def interpolate_to_observations_TS( datafile, betaparamfile, outfile, plevel ) -> None:
    """
    Interpolates large scale parameters to temperature and salinity profile-coordinates and computes 
    their mean and residuals on those coordinates. Saves to outfile.
    """
    # Load data and betaparameters
    TSlat, TSlon, TStime, temp, salt, d = cv.get_TSdata(datafile, plevel)
    blat, blon, betaparams_temp, betaparams_sal, b = cv.get_betaparams_2vars(betaparamfile)
    beta_temp_interpolated = cv.get_interpolated_parameters( betaparams_temp, blat, blon, TSlat, TSlon )
    beta_sal_interpolated = cv.get_interpolated_parameters( betaparams_sal, blat, blon, TSlat, TSlon )
    
    # Calculating mean and residuals
    NHarm = b.attrs["number_of_harmonics"]
    Mn = bf.basisfunctions_mean( TStime, NHarm  )
    mean_temp = np.sum(Mn * beta_temp_interpolated, axis=1)
    residuals_temp = temp - mean_temp
    mean_sal = np.sum(Mn * beta_sal_interpolated, axis=1 )
    residuals_sal = salt - mean_sal
    
    TSsave.save_TS_mean_res( plevel, mean_temp, mean_sal, residuals_temp, residuals_sal, TStime, TSlat, TSlon, b, d, NHarm, outfile )
    print(' Mean and residuals calculated and saved to outfile.')
    del  betaparams_temp, betaparams_sal, beta_temp_interpolated, beta_sal_interpolated, Mn, TStime, TSlat, TSlon, mean_temp, mean_sal, residuals_temp, residuals_sal
    
def grid_mean_seasonal_ts( 
    plevel, maskfile, betaparamfile, outfile, datafile, time_i, time_f, window_size,
    interval_days=5.0, Ndata_min=1) -> None:
    
    lats, lons, mask, latg, long = cv.get_compressed_mask(maskfile)
    blat, blon, betaparam_1, betaparam_2, b = cv.get_betaparams_2vars(betaparamfile)
    TSlat, TSlon, _, _, _, d = cv.get_TSdata(datafile, plevel)
    NHarm = b.attrs["number_of_harmonics"]
    pressure = b.pressure[:]
    maptimes =  np.arange(time_i+interval_days/2.0, time_f, interval_days)
    n_times = maptimes.size
    
    #Interpolation to coordinates
    beta_1_interpolated = cv.get_interpolated_parameters( betaparam_1, blat, blon, lats, lons )
    beta_2_interpolated = cv.get_interpolated_parameters( betaparam_2, blat, blon, lats, lons )
    mean_temp = np.full(shape=(latg.size, long.size, n_times, 1), fill_value=np.nan)
    mean_sal = np.full(shape=(latg.size, long.size, n_times, 1), fill_value=np.nan)
    
    iM,jM = np.where(mask)
    for n, i, j in zip(range(lats.size), iM, jM):
        
#     for n, i, j, lat0, lon0 in zip(range(lats.size), iM, jM, lats, lons):
        
#         """ 
#         Until a mask is created, temporarily add checks for latitude, and for amount of data in window, in order to reduce runtime.
#         """
        
#         if lat0<-19.0:
#             in_window = dist.in_spatial_window(lat0, lon0, TSlat, TSlon, window_size)
#             if in_window.size >= Ndata_min: 

        # lats = np.full(shape=(n_times), fill_value=lat0)
        # lons = np.full(shape=(n_times), fill_value=lon0)

        Mn = bf.basisfunctions_alt(n_times, maptimes, NHarm)
        mean_temp[i,j,:,0] = np.sum(Mn * beta_1_interpolated[n,:], axis=1)
        mean_sal[i,j,:,0] = np.sum(Mn * beta_2_interpolated[n,:], axis=1)

    TSsave.save_grid_mean_seasonal_TS( mean_temp, mean_sal, maptimes, latg, long, pressure, NHarm, b, d, outfile )
    print(' Gridded estimated '+str(int(interval_days))+'-day mean saved to outfile.')
    del beta_1_interpolated, beta_2_interpolated, betaparam_1, betaparam_2, blat, blon, lats, lons, mask, iM, jM
    
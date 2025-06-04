from convenient_bgc import *
from distancefunctions_bgc import *
from savefunctions_bgc import *


def run_covparam_window_average(covparamfile, maskfile, mask_outfile, outfile, correct_intervals, client, 
                                window_size=100e3, limit=1.5e6, scale=1.1, enough_data=5, batchsize=15) -> None:
    """ Creates and saves a mask based on optimized parameters with "wrong" 
    local minimum to mask_outfile. Iterates over this mask and switches out "wrong" parameters with an average 
    of nearby "correct" local minimum parameters, and saves to outfile."""
    
    # Create mask based on optimized parameters falling outside of "correct"-interval
    c = xr.open_zarr(get_gcs().get_mapper(covparamfile), consolidated=True)
    numparams = c.parameter.size
    optimized_covparameters = c.covparam.isel( pressure = 0 )
    parameter0 = optimized_covparameters.sel( parameter = 0 ).load()
    masked_grid = parameter0.where( (parameter0>=correct_intervals[0][0]) & (parameter0<=correct_intervals[0][1]) ).values
    for p in range(1, numparams):
        parameter = optimized_covparameters.sel( parameter = p ).load()
        masked_param = parameter.where( (parameter>=correct_intervals[p][0]) & (parameter<=correct_intervals[p][1]) ).values
        masked_grid = masked_grid + masked_param
    _, _, mask, latg, long = get_compressed_mask( maskfile )
    correct_mask = np.nan_to_num( masked_grid * 0 + 1, nan = 0 ).astype(int)
    wrong_mask = ( 1 - correct_mask ) * mask.mask
    save_wronglocalminimum_mask( wrong_mask, latg, long, mask_outfile, c ) # Storing for access in case needed.
    lonM, latM = np.meshgrid( long, latg )
    correct_lats = ma.MaskedArray(latM, 1-correct_mask).compressed() # "1-mask" since ma.MaskedArray use masked (unmasked) as False (True).
    correct_lons = ma.MaskedArray(lonM, 1-correct_mask).compressed() # "1-mask" since ma.MaskedArray use masked (unmasked) as False (True).
    correct_index = correct_mask.astype(bool)
    
    # Scatter data to workers on cluster
    new_initial_covparams = optimized_covparameters.values.copy() # Array to fill up with windowaveraged values where "wrong".
    correct_parameters_sc = client.scatter(new_initial_covparams[correct_index,:], broadcast=False)#, workers=workerlist) 
    correct_lats_sc = client.scatter(correct_lats, broadcast=False)#, workers=workerlist) 
    correct_lons_sc = client.scatter(correct_lons, broadcast=False)#, workers=workerlist)

    # Switch out parameters on "wrong"-local-minimum-gridpoints, with average of nearby "correct"-local-minimum-gridpoints.
    wrong_lats, wrong_lons, _, _, _ = get_compressed_mask( mask_outfile )
    list_of_submitted_tasks = []
    for k in range(0, wrong_lats.size, batchsize):
        submitted = client.submit( compute_window_average, 
            wrong_lats[k:k+batchsize], wrong_lons[k:k+batchsize], 
            correct_parameters_sc, correct_lats_sc, correct_lons_sc, 
            numparams, window_size, limit, scale, enough_data,
            pure = False
        )
        list_of_submitted_tasks.append(submitted)
    window_averaged = get_from_batches( client.gather(list_of_submitted_tasks), batchsize, alternative=True)
    new_initial_covparams[wrong_mask.astype(bool), :] = window_averaged
    save_initial_covparams( new_initial_covparams, outfile, correct_intervals, window_size, limit, scale, enough_data,  c )
    print('Window-averaged covariance-parameters saved to outfile.')
    del new_initial_covparams, correct_lats_sc, correct_lons_sc, correct_parameters_sc
    
    
def compute_window_average( wrong_lats, wrong_lons, correct_data, correct_lats, correct_lons, numparams, window_size, limit, scale, enough_data ) -> float:
    """ For correct-local-minima-values in window around "wrong"-local-minima gridpoints, return mean. 
    If not enough data in window, increase window-size"""
    
    # Loop through batch of "wrong" coordinates
    n = wrong_lats.size
    window_averages = np.full(shape=(n, numparams), fill_value=np.nan)
    for i, lat0, lon0 in zip( range(n), wrong_lats, wrong_lons ):
        distances = haversine_distance( lat0, lon0, correct_lats, correct_lons )
        in_window = np.where(distances < window_size)[0]
        data_iw = correct_data[in_window,:]
        increased_window_size = window_size
        while (~np.isnan(data_iw[:,0])).sum() < enough_data:
            increased_window_size = increased_window_size * scale
            if increased_window_size > limit: 
                raise ValueError(
                    'Limit: '+str(
                        limit)+' reached, increased_window_size: '+str(
                        increased_window_size)+' is too large.\n(Number of valid data in previous window: '+str((~np.isnan(data_iw[:,0])).sum())+').')
            in_window = np.where( distances < increased_window_size )[0]
            data_iw = correct_data[in_window,:]
        window_averages[i,:] = np.nanmean(data_iw, axis=0)
    return window_averages
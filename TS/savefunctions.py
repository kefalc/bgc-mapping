from convenient import *

def save_grid_anom_seasonal_2vars( from_compute, c, a, mask, latg, long, maptimes, outfile, plevel, Nblocks, window_size, interval_days ) -> None:
    
    """
    When time: 
    Save one variable at a time, using the original savefunctions 
    "save_grid_anom_seasonal", and delete this function. 
    """
    
    gridshape=(latg.size, long.size, maptimes.size, 2, 1)
    data_anom = np.full(shape=gridshape, fill_value=np.nan)
    data_mse = np.full(shape=gridshape, fill_value=np.nan)
    iM,jM = np.where(mask)[0], np.where(mask)[1]
    
    for n, i, j in zip( range(iM.size), iM, jM ):
        data_anom[i,j,:,0,0] = from_compute[n,:,0]
        data_mse[i,j,:,0,0]  = from_compute[n,:,1]
        data_anom[i,j,:,1,0] = from_compute[n,:,2]
        data_mse[i,j,:,1,0]  = from_compute[n,:,3]
        
    # create xr.dataset and save to outfile here:
    ds = xr.Dataset(
        data_vars = dict(
            anom = ( ['latitude', 'longitude', 'time', 'var', 'pressure'], data_anom ),
            mse = ( ['latitude', 'longitude', 'time', 'var', 'pressure'], data_mse ),
        ),
        coords=dict(
            time = ( ['time'], maptimes ),
            latitude = ( ['latitude'], latg ),
            longitude = ( ['longitude'], long ),
            var = ( ['var'], np.arange(2) ),
            pressure = (['pressure'], c.pressure[:].values )
        ),
        attrs=dict(
            description = 'Gridded estimated anomalies and errors on '+str(interval_days)+'-day grid',
            pressureindex = c.attrs["pressureindex"],
            window_size = window_size,
            Nblocks = Nblocks,
            covparamfile = c.attrs["filename"],
            anomfile = a.attrs["filename"],
            filename=outfile,
            maptime_interval_days = interval_days,
            created_date = str( datetime.now() )
        )
    )
    ds.time.attrs["standard_name"] = 'time'
    ds.time.attrs["units"] = 'days since 1970-01-01 00:00:00'
    ds.latitude.attrs = c.latitude.attrs
    ds.longitude.attrs= c.longitude.attrs
    ds.pressure.attrs = c.pressure.attrs

    # Save dataset
    ds.to_netcdf(outfile)
    del from_compute, c, a, mask, latg, long, maptimes, outfile, plevel, Nblocks, window_size, ds
    
    ##will just need 1var for BGC
def save_covparams_2vars( from_gather, outfile, plevel, a, mask, latg, long, window_size, Nblocks, nu, total_time, ic, numparams ) -> None:
    
    results = np.array(from_gather, dtype=object)
    
    # Preallocating arrays
    gridshape = (mask.shape[0], mask.shape[1], 2)
    opt_covparam = np.full(shape= (latg.size, long.size, numparams,2), fill_value=np.nan)
    init_covparam = np.full(shape= (latg.size, long.size, numparams,2), fill_value=np.nan)
    opt_fval = np.full(shape=gridshape, fill_value=np.nan)
    opt_success = np.full(shape=gridshape, fill_value=np.nan)
    opt_message = np.chararray(shape=gridshape, itemsize=61) 
    opt_status = np.full(shape=gridshape, fill_value=np.nan)
    opt_nfev = np.full(shape=gridshape, fill_value=np.nan)
    opt_nit = np.full(shape=gridshape, fill_value=np.nan)
    opt_njev = np.full(shape=gridshape, fill_value=np.nan)
    opt_time = np.full(shape=gridshape, fill_value=np.nan)
    
    # Allocating parameters to gridpoints using mask
    iM, jM = np.where(mask)
    init_covparam[iM,jM,:,:] = np.array([item[16] for item in results])  # x0, initial parameters
    opt_covparam[iM,jM,:,0] = np.array([item[0] for item in results])    # x, optimized parameters
    opt_covparam[iM,jM,:,1] = np.array([item[8] for item in results])
    opt_fval[iM,jM,0] = np.array([item[1] for item in results])          # functionvalue
    opt_fval[iM,jM,1] = np.array([item[9] for item in results])
    opt_success[iM,jM,0] = np.array([item[2] for item in results])       # boolean 1,0, or bad startingpoint 6
    opt_success[iM,jM,1] = np.array([item[10] for item in results])
    opt_message[iM,jM,0] = np.array([item[3] for item in results])       # Describing cause of termination
    opt_message[iM,jM,1] = np.array([item[11] for item in results])
    opt_status[iM,jM,0]  = np.array([item[4] for item in results])       # see scipy.optimize.minimize
    opt_status[iM,jM,1]  = np.array([item[12] for item in results])
    opt_nfev[iM,jM,0] = np.array([item[5] for item in results])          # number of function-evaluations
    opt_nfev[iM,jM,1] = np.array([item[13] for item in results])         
    opt_nit[iM,jM,0] = np.array([item[6] for item in results])           # number of iterations
    opt_nit[iM,jM,1] = np.array([item[14] for item in results])          
    opt_njev[iM,jM,0] = np.array([item[7] for item in results])          # number of jacobian-evaluations
    opt_njev[iM,jM,1] = np.array([item[15] for item in results])         #
    opt_time[iM,jM,0] = np.array([item[17] for item in results])         # number of seconds each minimization took
    opt_time[iM,jM,1] = np.array([item[18] for item in results])         
    
    # Create dataset of optimization-results
    covparams = xr.Dataset(
        data_vars = dict(
            initial_covparam = ( ['latitude', 'longitude', 'parameter', 'var', 'pressure'], np.expand_dims(init_covparam, axis=-1) ),
            covparam = ( ['latitude', 'longitude', 'parameter', 'var', 'pressure'], np.expand_dims(opt_covparam, axis=-1) ),
            fval = (['latitude', 'longitude', 'var', 'pressure'], np.expand_dims(opt_fval, axis=-1) ),
            success = (['latitude', 'longitude', 'var', 'pressure'], np.expand_dims(opt_success, axis=-1) ),
            message = (['latitude', 'longitude','var', 'pressure'], np.expand_dims(opt_message, axis=-1) ),
            status = (['latitude', 'longitude', 'var', 'pressure'], np.expand_dims(opt_status, axis=-1) ),
            nfev = (['latitude', 'longitude', 'var', 'pressure'], np.expand_dims(opt_nfev, axis=-1) ),
            nit = (['latitude', 'longitude', 'var', 'pressure'], np.expand_dims(opt_nit, axis=-1) ),
            njev = (['latitude', 'longitude', 'var', 'pressure'], np.expand_dims(opt_njev, axis=-1) ),
            time_optimization = (['latitude', 'longitude', 'var', 'pressure'], np.expand_dims(opt_time, axis=-1) ),
        ),
        coords=dict(
            latitude = ( ['latitude'], latg ),
            longitude = ( ['longitude'], long ),
            parameter = ( ['parameter'], np.arange(numparams) ),
            var = ( ['var'], np.arange(2) ),
            pressure = (["pressure"], a.pressure[:].values )
        ),
        attrs=dict(
            description = 'For description, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html \n',
            pressureindex = plevel,
            window_size = window_size,
            Nblocks = Nblocks,
            nu = nu,
            total_time = total_time,
            anomfile = a.attrs["filename"],
            initial_covparamsfile = ic.attrs["filename"] if ic else 'None',
            filename = outfile,
            created_date = str( datetime.now() )
        )
    )
    covparams.latitude.attrs["standard_name"] = 'latitude'
    covparams.latitude.attrs["units"] = 'degrees_north'
    covparams.longitude.attrs["standard_name"] = 'longitude'
    covparams.longitude.attrs["units"] = 'degrees_east'
    covparams.pressure.attrs["standard_name"] = 'pressure'
    covparams.pressure.attrs["units"] = 'decibar'
    covparams.time_optimization.attrs["standard_name"] = 'number of seconds each optimization took'
    
    # Save dataset
    covparams.to_netcdf(outfile)
    del from_gather, results, outfile, plevel, a, ic, mask, latg, long, window_size, Nblocks, covparams, nu
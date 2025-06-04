from convenient_bgc import *
from time import time as Time

#will just need one variable
# def save_grid_anom_seasonal_bgc( from_compute, c, a, mask, latg, long, maptimes, outfile, plevel, Nblocks, window_size, interval_days ) -> None:
    
#     """
#     When time: 
#     Save one variable at a time, using the original savefunctions 
#     "save_grid_anom_seasonal", and delete this function. 
#     """
    
#     gridshape=(latg.size, long.size, maptimes.size, 2, 1)
#     data_anom = np.full(shape=gridshape, fill_value=np.nan)
#     data_mse = np.full(shape=gridshape, fill_value=np.nan)
#     iM,jM = np.where(mask)[0], np.where(mask)[1]
    
#     for n, i, j in zip( range(iM.size), iM, jM ):
#         data_anom[i,j,:,0,0] = from_compute[n,:,0]
#         data_mse[i,j,:,0,0]  = from_compute[n,:,1]
#         data_anom[i,j,:,1,0] = from_compute[n,:,2]
#         data_mse[i,j,:,1,0]  = from_compute[n,:,3]
        
#     # create xr.dataset and save to outfile here:
#     ds = xr.Dataset(
#         data_vars = dict(
#             anom = ( ['latitude', 'longitude', 'time', 'var', 'pressure'], data_anom ),
#             mse = ( ['latitude', 'longitude', 'time', 'var', 'pressure'], data_mse ),
#         ),
#         coords=dict(
#             time = ( ['time'], maptimes ),
#             latitude = ( ['latitude'], latg ),
#             longitude = ( ['longitude'], long ),
#             var = ( ['var'], np.arange(2) ),
#             pressure = (['pressure'], c.pressure[:].values )
#         ),
#         attrs=dict(
#             description = 'Gridded estimated anomalies and errors on '+str(interval_days)+'-day grid',
#             pressureindex = c.attrs["pressureindex"],
#             window_size = window_size,
#             Nblocks = Nblocks,
#             covparamfile = c.attrs["filename"],
#             anomfile = a.attrs["filename"],
#             filename=outfile,
#             maptime_interval_days = interval_days,
#             created_date = str( datetime.now() )
#         )
#     )
#     ds.time.attrs["standard_name"] = 'time'
#     ds.time.attrs["units"] = 'days since 1970-01-01 00:00:00'
#     ds.latitude.attrs = c.latitude.attrs
#     ds.longitude.attrs= c.longitude.attrs
#     ds.pressure.attrs = c.pressure.attrs

#     # Save dataset
#     ds.to_netcdf(outfile)
#     del from_compute, c, a, mask, latg, long, maptimes, outfile, plevel, Nblocks, window_size, ds

def save_grid_anom_seasonal_bgc( from_compute, c, a, mask, latg, long, maptimes, outfile, plevel, Nblocks, window_size, interval_days ) -> None:
    
    gridshape=(latg.size, long.size, maptimes.size, 1)
    data_anom = np.full(shape=gridshape, fill_value=np.nan)
    data_mse = np.full(shape=gridshape, fill_value=np.nan)
    iM,jM = np.where(mask)
    
    for n, i, j in zip( range(iM.size), iM, jM ):
        data_anom[i,j,:,0] = from_compute[n,:,0]
        data_mse[i,j,:,0]  = from_compute[n,:,1]
        
    # create xr.dataset and save to outfile here:
    ds = xr.Dataset(
        data_vars = dict(
            anom = ( ['latitude', 'longitude', 'time', 'pressure'], data_anom ),
            mse = ( ['latitude', 'longitude', 'time', 'pressure'], data_mse ),
        ),
        coords=dict(
            time = ( ['time'], maptimes ),
            latitude = ( ['latitude'], latg ),
            longitude = ( ['longitude'], long ),
            pressure = (['pressure'], c.pressure[:].values )
        ),
        attrs=dict(
            description = 'Gridded estimated anomalies and errors on '+str(interval_days)+'day grid',
            pressureindex = c.attrs["pressureindex"],
            window_size = window_size,
            Nblocks = Nblocks,
            covparamfile = c.attrs["filename"],
            anomfile = a.attrs["filename"],
            # covparameter_attributes = dict(c.attrs),
            # anomdata_attributes = dict(a.attrs),
            filename=outfile,
            maptime_interval_days = interval_days,
            created_date = str( datetime.now() )
        )
    )
    ds.anom.attrs["standard_name"] = 'Estimated  anomalies'
   # ds.anom.attrs["units"] = b.residuals_bgc.attrs["units"]
    ds.time.attrs["standard_name"] = 'time'
    ds.time.attrs["units"] = 'days since 1970-01-01 00:00:00'
    ds.latitude.attrs = c.latitude.attrs
    ds.longitude.attrs= c.longitude.attrs
    ds.pressure.attrs = c.pressure.attrs

    # Save dataset
    ds.to_netcdf(outfile)
    del from_compute, c, a, mask, latg, long, maptimes, outfile, plevel, Nblocks, window_size, ds
    
def save_covparams_bgc( from_gather, outfile, plevel, b, mask, latg, long, window_size, Nblocks, t_start, numparams ) -> None:
    
    results = np.array( from_gather, dtype=object )

    # Preallocating arrays
    gridshape = (*mask.shape,1)
    opt_covparam = np.full(shape=(latg.size, long.size, numparams,1), fill_value=np.nan)
    init_covparam = np.full(shape=(latg.size, long.size, numparams,1), fill_value=np.nan)
    opt_fval = np.full(shape=gridshape, fill_value=np.nan)
    opt_success = np.full(shape=gridshape, fill_value=np.nan)
    opt_message = np.chararray(shape=gridshape, itemsize=50) 
    opt_status = np.full(shape=gridshape, fill_value=np.nan)
    opt_nfev = np.full(shape=gridshape, fill_value=np.nan)
    opt_nit = np.full(shape=gridshape, fill_value=np.nan)
    opt_njev = np.full(shape=gridshape, fill_value=np.nan)
    var_Nblocks = np.full(shape=gridshape, fill_value=np.nan)
    Ndata_in_window = np.full(shape=gridshape, fill_value=np.nan)
    opt_time = np.full(shape=gridshape, fill_value=np.nan)
    
    # Allocating parameters to gridpoints using mask
    try:
        iM, jM = np.where(mask)
        init_covparam[iM,jM,:,0] = np.array([item[8] for item in results]) # x0, initial parameters
        opt_covparam[iM,jM,:,0] = np.array([item[0] for item in results])      # x, optimized parameters
        opt_fval[iM,jM,0] = np.array([item[1] for item in results])            # functionvalue
        opt_success[iM,jM,0] = np.array([item[2] for item in results])         # boolean 1,0, or bad startingpoint 6
        opt_message[iM,jM,0] = np.array([item[3] for item in results])         # Describing cause of termination
        opt_status[iM,jM,0] = np.array([item[4] for item in results])          # see scipy.optimize.minimize
        opt_nfev[iM,jM,0] = np.array([item[5] for item in results])            # number of function-evaluations
        opt_nit[iM,jM,0] = np.array([item[6] for item in results])             # number of iterations
        opt_njev[iM,jM,0] = np.array([item[7] for item in results])            # number of jacobian-evaluations
        var_Nblocks[iM,jM,0] = np.array([item[9] for item in results])         # number of blocks local window data was divided into
        Ndata_in_window[iM,jM,0] = np.array([item[10] for item in results])     # number of data-coordinates in local window
        opt_time[iM,jM,0] = np.array([item[11] for item in results])           # number of seconds each minimization took
    except IndexError:
        print(results.shape)
        print(results[0])
    
    # Create dataset of optimization-results
    try:
        datafile = b.attrs["bgcresidualfile"]
    except KeyError:
        datafile = ""
    covparams = xr.Dataset(
        data_vars = dict(
            initial_covparam = ( ['latitude', 'longitude', 'parameter', 'pressure'], init_covparam ),
            covparam = ( ['latitude', 'longitude', 'parameter', 'pressure'], opt_covparam ),
            fval = (['latitude', 'longitude', 'pressure'], opt_fval ),
            success = (['latitude', 'longitude', 'pressure'], opt_success ),
            message = (['latitude', 'longitude', 'pressure'], opt_message ),
            status = (['latitude', 'longitude', 'pressure'], opt_status ),
            nfev = (['latitude', 'longitude', 'pressure'], opt_nfev ),
            nit = (['latitude', 'longitude', 'pressure'], opt_nit ),
            njev = (['latitude', 'longitude', 'pressure'], opt_njev ),
            Nblocks = (['latitude', 'longitude', 'pressure'], var_Nblocks),
            Ndata_in_window = (['latitude', 'longitude', 'pressure'], Ndata_in_window ),
            time_optimization = (['latitude', 'longitude', 'pressure'], opt_time ),
        ),
        coords=dict(
            latitude = ( ['latitude'], latg ),
            longitude = ( ['longitude'], long ),
            parameter = ( ['parameter'], np.arange(numparams) ),
            pressure = (["pressure"], b.pressure.data[:] )
        ),
        attrs=dict(
            description = 'For description, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html \n',
            pressureindex = plevel,
            window_size = window_size,
            Nblocks = Nblocks,
            nu = 3/2,
            total_time = Time()-t_start,
            residualfile = b.attrs["filename"],
            datafile = datafile,
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
    covparams.Nblocks.attrs["standard_name"] = 'number of blocks local window data was divided into'
    covparams.Ndata_in_window.attrs["standard_name"] = 'number of data in local window'

    covparams.to_netcdf(outfile)
from convenient import *

def save_betaparams_2vars( from_compute, outfile, plevel, pressure, mask, latg, long, window_size1, window_size2, NHarm, total_time ) -> None:
    results = np.array(from_compute)

    # Preallocating array 
    numparams = NHarm*2 + 6
    betaparam_grid_temp = np.full(shape=(latg.size,long.size,numparams, 1), fill_value=np.nan)
    betaparam_grid_sal = np.full(shape=(latg.size,long.size,numparams, 1), fill_value=np.nan)

    # Allocating parameters to gridpoints using mask
    for param in range(numparams):
        tmp1 = np.full(( latg.size, long.size ), np.nan)
        tmp2 = np.full(( latg.size, long.size ), np.nan)
        tmp1[mask.mask] = results[:,0,param]
        tmp2[mask.mask] = results[:,1,param]
        betaparam_grid_temp[:,:,param, 0] = tmp1
        betaparam_grid_sal[:,:,param, 0] = tmp2
    
    # Create dataset of parameters
    beta_ds = xr.Dataset(
        data_vars = dict(
            betaparam_temp = ( ["latitude", "longitude", "parameter", "pressure"], betaparam_grid_temp ),
            betaparam_sal = ( ["latitude", "longitude", "parameter", "pressure"], betaparam_grid_sal )
        ),
        coords=dict(
            latitude = ( ["latitude"], np.squeeze(latg.data) ),
            longitude = ( ["longitude"], np.squeeze(long.data) ),
            parameter = ( ["parameter"], np.arange(numparams) ),
            pressure = (["pressure"], np.array([pressure]) )
        ),
        attrs=dict(
            description = 'Gridded betaparameters for the '+str(pressure)+' decibar pressurelevel.',
            pressureindex = plevel,
            number_of_harmonics = NHarm,
            window_size1 = window_size1,
            window_size2 = window_size2,
            total_time = total_time,
            filename=outfile,
            created_date = str( datetime.now() )
        )
    )
    beta_ds.latitude.attrs["standard_name"] = 'latitude'
    beta_ds.latitude.attrs["units"] = 'degrees_north'
    beta_ds.longitude.attrs["standard_name"] = 'longitude'
    beta_ds.longitude.attrs["units"] = 'degrees_east'
    beta_ds.betaparam_temp.attrs["standard_name"] = 'temperature'
    beta_ds.betaparam_temp.attrs["units"] = 'coefficient_of_fit'
    beta_ds.betaparam_sal.attrs["standard_name"] = 'salinity'
    beta_ds.betaparam_sal.attrs["units"] = 'coefficient_of_fit'
    beta_ds.pressure.attrs["standard_name"] = 'pressure'
    beta_ds.pressure.attrs["units"] = 'decibar'

    beta_ds.to_netcdf(outfile)

def save_TS_mean_res( plevel, mean_temp, mean_sal, residuals_temp, residuals_sal, time, lat, lon, b, d, NHarm, outfile ) -> None:
    
    # Mean and residuals for pressurelevel is saved to output-path:
    ds = xr.Dataset(
        data_vars = dict(
            mean_temp = ( ["i", "pressure"], mean_temp.reshape(mean_temp.size, 1) ), #need to update these to t&s
            mean_sal = ( ["i", "pressure"], mean_sal.reshape(mean_sal.size, 1) ),
            residuals_temp = ( ["i", "pressure"], residuals_temp.reshape(residuals_temp.size, 1) ),
            residuals_sal = ( ["i", "pressure"], residuals_sal.reshape(residuals_sal.size, 1) )
        ),
        coords=dict(
            time = ( ["i"], time.data),
            latitude = ( ["i"], lat.data ),
            longitude = ( ["i"], lon.data ),
            pressure = (["pressure"], b.pressure[:].data )
        ),
        attrs=dict(
            description = 'Estimated mean temperature and salinity, and residuals by subtracting from observations',
            pressureindex = plevel,
            number_of_harmonics = NHarm,
            window_size1 = b.attrs["window_size1"],
            window_size2 = b.attrs["window_size2"],
            filename=outfile,
            created_date = str( datetime.now() )
        )
    )
    ds.time.attrs["standard_name"] = 'time'
    ds.time.attrs["units"] = 'days since 1970-01-01 00:00:00'
    ds.latitude.attrs = d.lat.attrs
    ds.longitude.attrs = d.lon.attrs
    ds.pressure.attrs["standard_name"] = 'pressure'
    ds.pressure.attrs["units"] = 'decibar'
    ds.mean_temp.attrs["standard_name"] = 'Estimated temperature mean'
    ds.mean_sal.attrs["standard_name"] = 'Estimated salinity mean'
    ds.mean_temp.attrs["units"] = d.temperature.attrs["units"]
    ds.mean_sal.attrs["units"] = d.salinity.attrs["units"]
    ds.residuals_temp.attrs["standard_name"] = 'Temperature Residuals'
    ds.residuals_sal.attrs["standard_name"] = 'Salinity Residuals'
    ds.residuals_temp.attrs["units"] = d.temperature.attrs["units"]
    ds.residuals_sal.attrs["units"] = d.salinity.attrs["units"]

    #Save chunked dataset
    dsc = ds.chunk()
    dsc.to_netcdf(outfile)
    del plevel, mean_temp, mean_sal, residuals_temp, residuals_sal, time, lat, lon, b, d, NHarm, outfile, ds
    
def save_grid_mean_seasonal_TS( mean_temp, mean_sal, maptimes, latg, long, pressure, NHarm, b, d, outfile ) -> None:
    
    # create xr.dataset and save to outfile here:
    ds = xr.Dataset(
        data_vars = dict(
            mean_temp = ( ['latitude', 'longitude', 'time', 'pressure'], mean_temp ),
            mean_sal = ( ['latitude', 'longitude', 'time', 'pressure'], mean_sal ),
        ),
        coords=dict(
            time = ( ['time'], maptimes ),
            latitude = ( ['latitude'], latg.data ),
            longitude = ( ['longitude'], long.data ),
            pressure = ( ['pressure'], pressure.data )
        ),
        attrs=dict(
            description = 'Gridded estimated mean T&S on time-intervals.',
            pressureindex = b.attrs["pressureindex"],
            number_of_harmonics = NHarm,
            filename = outfile,
            created_date = str( datetime.now() )
        )
    )
    ds.mean_temp.attrs["standard_name"] = 'Estimated mean T'
    ds.mean_sal.attrs["standard_name"] = 'Estimated mean S'
    ds.mean_temp.attrs["units"] = d.temperature.attrs["units"]
    ds.mean_sal.attrs["units"] = d.salinity.attrs["units"]
    ds.time.attrs["standard_name"] = 'time'
    ds.time.attrs["units"] = 'days since 1970-01-01 00:00:00'
    ds.latitude.attrs["standard_name"] = b.latitude.attrs["standard_name"]
    ds.latitude.attrs["units"] = b.latitude.attrs["units"]
    ds.longitude.attrs["standard_name"] = b.longitude.attrs["standard_name"]
    ds.longitude.attrs["units"] = b.longitude.attrs["units"]
    ds.pressure.attrs["standard_name"] = b.pressure.attrs["standard_name"]
    ds.pressure.attrs["units"] = b.pressure.attrs["units"]

    # Save chunked dataset
    ds_chunked = ds.chunk()
    ds_chunked.to_netcdf(outfile)
    del mean_temp, mean_sal, maptimes, latg, long, pressure, NHarm, b, outfile, ds, ds_chunked
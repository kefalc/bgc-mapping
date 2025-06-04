from convenient_bgc import *

##make this generic and save attributes based on input field
def save_BGC_mean_res( datafield, plevel, mean_bgc, residuals_bgc, time, lat, lon, b, d, NHarm, outfile ) -> None:
    
    # Mean and residuals for pressurelevel is saved to output-path:
    ds = xr.Dataset(
        data_vars = dict(
            mean_bgc = ( ["i", "pressure"], mean_bgc.reshape(mean_bgc.size,1) ),
            residuals_bgc = ( ["i", "pressure"], residuals_bgc.reshape(residuals_bgc.size, 1) )
        ),
        coords=dict(
            time = ( ["i"], time.data),
            latitude = ( ["i"], lat.data ),
            longitude = ( ["i"], lon.data ),
            pressure = (["pressure"], b.pressure[:].data )
        ),
        attrs=dict(
            description = 'Estimated mean' + str(datafield)+ ', and residuals by subtracting from observations', #don't know if this is correct
            pressureindex = plevel,
            number_of_harmonics = NHarm,
            window_size = b.attrs["window_size"],
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
    ds.mean_bgc.attrs["standard_name"] = 'Estimated mean ' + str(datafield)
    ds.mean_bgc.attrs["units"] = d[datafield].attrs["units"] #don't know if this is correct
    ds.residuals_bgc.attrs["standard_name"] = str(datafield) + ' residuals'
    ds.residuals_bgc.attrs["units"] = d[datafield].attrs["units"] #don't know if this is correct

    #Save chunked dataset
    dsc = ds.chunk()
    dsc.to_netcdf(outfile)
    del plevel, mean_bgc, residuals_bgc, time, lat, lon, b, d, NHarm, outfile, ds
    

def save_grid_mean_seasonal_bgc(datafield, mean_bgc, maptimes, latg, long, pressure, NHarm, b, d, outfile ) -> None:
    
    # create xr.dataset and save to outfile here:
    ds = xr.Dataset(
        data_vars = dict(
            mean_bgc = ( ['latitude', 'longitude', 'time', 'pressure'], mean_bgc ),
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
    ds.mean_bgc.attrs["standard_name"] = 'Estimated mean ' + str(datafield)
    ds.mean_bgc.attrs["units"] = d[datafield].attrs["units"] #not sure if this is correct
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
    del mean_bgc, maptimes, latg, long, pressure, NHarm, b, outfile, ds, ds_chunked
    
    
def save_betaparams(
    results, outfile, datafile, plevel, pressure, mask, latg, long, window_size, NHarm, Ndata_min, t_start 
        ) -> None:
    
    #display(results.shape)

    # Preallocating array 
    numparams = int(NHarm*2 + 8)
    betaparam_grid = np.full(shape=( latg.size, long.size, numparams, 1 ), fill_value=np.nan)
    # iM, jM = np.where(mask)

    # Allocating parameters to gridpoints using mask
    # for param in range(numparams):
    #     betaparam_grid[iM,jM,param, 0] = results[:,param]
        
    for param in range(numparams):
        tmp1 = np.full(( latg.size, long.size ), np.nan)
        tmp1[mask.mask] = results[:,param]
        betaparam_grid[:,:,param,0] = tmp1
        
    # Create dataset of parameters
    beta_ds = xr.Dataset(
        data_vars = dict(
            betaparam = ( ["latitude", "longitude", "parameter", "pressure"], betaparam_grid )
        ),
        coords=dict(
            latitude = ( ["latitude"], np.squeeze(latg.data) ),
            longitude = ( ["longitude"], np.squeeze(long.data) ),
            parameter = ( ["parameter"], np.arange(numparams) ),
            pressure = (["pressure"], np.array([pressure]) )
        ),
        attrs=dict(
            description = 'Gridded betaparameters for the '+str(pressure)+' decibar pressurelevel.',
            datafile = datafile, 
            pressureindex = plevel,
            number_of_harmonics = NHarm,
            minimum_data_in_window = Ndata_min,
            window_size = window_size,
            total_time = 0, #Time()-t_start,
            filename=outfile,
            created_date = str( datetime.now() )
        )
    )
    beta_ds.latitude.attrs["standard_name"] = 'latitude'
    beta_ds.latitude.attrs["units"] = 'degrees_north'
    beta_ds.longitude.attrs["standard_name"] = 'longitude'
    beta_ds.longitude.attrs["units"] = 'degrees_east'
    beta_ds.betaparam.attrs["standard_name"] = 'betaparameter'
    beta_ds.betaparam.attrs["units"] = 'coefficient_of_fit'
    beta_ds.pressure.attrs["standard_name"] = 'pressure'
    beta_ds.pressure.attrs["units"] = 'decibar'

    # Save chunked dataset
    ds_chunked = beta_ds.chunk()
    ds_chunked.to_netcdf(outfile)
    del beta_ds, results, outfile, plevel, pressure, mask, latg, long, window_size, NHarm

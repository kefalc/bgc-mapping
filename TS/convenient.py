from dask_gateway import Gateway, GatewayCluster
from distributed import Client, LocalCluster
from scipy.interpolate import interpn, NearestNDInterpolator
from datetime import datetime
import json, gcsfs, fsspec
import numpy as np, numpy.ma as ma
import xarray as xr
import zarr
import math

def get_compressed_mask(maskfile) -> (np.ndarray, np.ndarray, ma.core.MaskedArray, np.ndarray, np.ndarray):
    m = xr.open_dataset(maskfile)
    mask0 = m.float_mask.to_numpy()
    long = m.lon.to_numpy()
    latg = m.lat.to_numpy()
    lonM, latM = np.meshgrid(long, latg)
    lon_comp = ma.MaskedArray(lonM, 1-mask0).compressed()  # 1-mask0 since ma.MaskedArray masks wanted (unwanted) values as False (True).
    lat_comp = ma.MaskedArray(latM, 1-mask0).compressed()  # 1-mask0 since ma.MaskedArray masks wanted (unwanted) values to False (True).
    mask = ma.array(mask0, mask=mask0)
    return lat_comp, lon_comp, mask, latg, long


def get_residualdata_2vars(residualfile) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, xr.core.dataset.Dataset):
    a = xr.open_dataset(residualfile, decode_times=False)
    return a.residuals_temp.to_numpy()[:,0], a.residuals_sal.to_numpy()[:,0], a.latitude.to_numpy(), a.longitude.to_numpy(), a.time.to_numpy(), a 

def get_interpolated_parameters( parameters, blat, blon, f_lat, f_lon ) -> np.ndarray:

    """
    The "left-end"-array is concatenated onto the "right-end" of the array.
    """
    
    # Preallocation
    interpolated = np.full( shape=(len(f_lat), parameters.shape[2]), fill_value=np.nan )

    # Wraparound
    param_leftcolumn = np.full( shape=(blat.size, parameters.shape[2]), fill_value=parameters[:,0,:] )
    param_leftcolumn = param_leftcolumn.reshape(blat.size,1,parameters.shape[2])
    param_wrapped = np.concatenate( (parameters, param_leftcolumn), axis=1)    
    wlon = np.append(blon,blon[-1]+1)

    # Interpolation
    points = (blat, wlon)
    xi = (f_lat, f_lon)
    for i in range(parameters.shape[2]):
        values = param_wrapped[:,:,i]
        interpolated[:,i] = interpn(points, values, xi, bounds_error=False, fill_value=np.nan)
        """
        In interpn:
        parameters = beta parameters
        blat/blon = beta param locations
        f_lat, f_lon = data locations
        """
    fail = np.isnan(interpolated[:,0])
    if fail.sum() > 0:
        mask = np.isnan(param_wrapped[:,:,0])
        lon_grid, lat_grid  = np.meshgrid(wlon, blat)
        lonM, latM = ma.MaskedArray(lon_grid, mask).compressed(),ma.MaskedArray(lat_grid, mask).compressed()
        for i in range(parameters.shape[2]):
            interp = NearestNDInterpolator(list(zip(lonM, latM)), param_wrapped[:,:,i][~mask])
            interpolated[fail,i] = interp(f_lon[fail], f_lat[fail])
            
    print('  Interpolated to', f_lat.size,'coordinates. Nan-estimate: '+str(np.isnan(interpolated).sum()/parameters.shape[2])+'.', end='' )
    return interpolated

def get_covparams_2vars(covparamfile) -> (np.ndarray, np.ndarray, np.ndarray, zarr.hierarchy.Group):
    c = xr.open_dataset(covparamfile)
    return c.latitude[:], c.longitude[:], c.covparam[:,:,:,:,0], c


def get_initial_covparams(initcovparamfile=None, datafield=None, mask=None, latg=None, long=None, Opt_nu=False, covparamfile=None) -> tuple:
    if covparamfile:
        ic = xr.open_dataset(covparamfile)
        covparams = ic.covparam[:,:,:,0]
        log_x0 = np.log(covparams)
    elif initcovparamfile and Opt_nu:
        raise ValueError("No code  here yet")
    elif initcovparamfile and not Opt_nu:
        ic = xr.open_dataset(initcovparamfile)
        initial_covparams = ic.initial_covparam[:,:,:,0]
        log_x0 = np.log(initial_covparams)
    elif not initcovparamfile:
        if datafield == 'DH': initparams = (1e-03, 5.0, 4.0, 60.0, 5e-3)
        elif datafield == 'vel': initparams = (1e-01, 5.0, 4.0, 60.0, 5e-3)
        elif datafield == 'DIC': initparams = (120.0, 5.0, 4.0, 60.0, 2.0)
        elif datafield == 'TS': 
            initparams = ((2.0, 5.0, 4.0, 60.0, 1e-1),(0.5, 5.0, 4.0, 60.0, 1e-2))
            log_initparams_T = np.log(np.array([initparams[0][0] , initparams[0][1] , initparams[0][2] , initparams[0][3] , initparams[0][4]]))
            log_initparams_S = np.log(np.array([initparams[1][0] , initparams[1][1] , initparams[1][2] , initparams[1][3] , initparams[1][4]]))
            log_x0 = np.full( shape=( mask.mask.shape[0], mask.mask.shape[1], log_initparams_T.size, 2 ), fill_value=np.nan )
            for parameterindex in range(log_initparams_T.size):
                log_x0[mask.mask,parameterindex,0] = log_initparams_T[parameterindex]
                log_x0[mask.mask,parameterindex,1] = log_initparams_S[parameterindex]
            ic = None
            return log_x0, ic
        else: raise ValueError("No datafield")
        ThetasInit = initparams[0]   # Signal variance
        ThetaxInit = initparams[1]   # Decorrelation length scale 
        ThetayInit = initparams[2]   # Decorrelation length scale
        ThetatInit = initparams[3]   # Decorrelation time scale days 
        SigmaInit = initparams[4]    # Standard deviation of noise
        log_initparams = np.log(np.array([ThetasInit, ThetaxInit, ThetayInit, ThetatInit, SigmaInit]))
        log_x0 = np.full( shape=( mask.mask.shape[0], mask.mask.shape[1], log_initparams.size ), fill_value=np.nan )
        for parameterindex in range(log_initparams.size):
            log_x0[mask.mask,parameterindex] = log_initparams[parameterindex]
        ic = None
    else: raise ValueError("Something wrong.")
    return log_x0, ic


def get_dateN(time_i, time_f, Nblocks) -> np.ndarray:
    """
    Divide data into N chunks in time - 
    here based on time range of full dataset (given by time_i and time_f)
    time_f should be > all data times
    """
    Trange = time_f - time_i
    if Nblocks == 1: 
        dateN = np.array([time_i, time_f])
    else:
        dateN = np.append( 
        np.arange( time_i, time_f, np.int64(np.round(Trange/Nblocks)) ), 
        time_f 
        )
    return dateN


def get_workerlist(client, n_workers) -> list:
    """ Returns list of  n worker-addresses that the client submits tasks to.
    None means all available workers work on the job.
    Ref: https://dask.discourse.group/t/how-can-i-let-only-a-few-workers-start-a-small-task-while-others-wait-on-a-remote-cluster/2202 """
    
    if n_workers == None: return None
    from distributed import get_worker
    a = client.run(lambda: get_worker().address)
    if n_workers < 1 or n_workers > len(a): raise IndexError("Out of range")
    return list(a)[:n_workers]


def get_from_batches( from_compute, batchsize) -> np.ndarray:
    """ 
    Unpacks results after doing computation in batches. When time: try dask.bag-API 
    """
    except_last = np.array(from_compute[:-1])
    shape = except_last.shape
    new_shape = [shape[0]*batchsize]
    for dim in range(2, len(shape)):
        new_shape.append(shape[dim])
    except_last_reshaped = except_last.reshape(tuple(new_shape))
    last = np.array(from_compute[-1])
    return np.concatenate( (except_last_reshaped, last), axis = 0 )

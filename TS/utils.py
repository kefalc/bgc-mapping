import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
# from jax.scipy.optimize import minimize
from scipy.optimize import minimize as sp_minimize
import scipy as sp
import xarray as xr

@jit
def softplus(x):
    return jnp.logaddexp(x, 0.)


@jit
def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


@jit
def det(x):
    return jnp.prod(x)


@jit
def sqrt_dist(x1, x2):
    d = x1 - x2
    is_zero = jnp.allclose(d, 0.)
    # replace d with ones if is_zero
    d = jnp.where(is_zero, jnp.ones_like(d), d)
    d = jnp.linalg.norm(d)
    # replace norm with zero if is_zero
    d = jnp.where(is_zero, 0., d)
    # d = jnp.max(jnp.array([d, 0.001]))
    return d


@jit
def matern32(x1, x2):
    z = jnp.sqrt(3.0) * sqrt_dist(x1, x2)
    return (1 + z) * jnp.exp(-z)


def BFGS_training(params, x, y, marginal_likelihood):
    '''
    perform training on given initial parameters, training data and objective function using LBFGS
    '''

    # convert params to a 1-d arrray while recording its original shape, 
    # since the scipy-version of LBFGS only accepts 1-d array
    def concat_params(params):
        flat_params, params_tree = jax.tree_util.tree_flatten(params)
        params_shape = [x.shape for x in flat_params]
        return jnp.concatenate([x.reshape(-1) for x in flat_params]), (params_tree, params_shape)

    param_vector, (params_tree, params_shape) = concat_params(params)

    ## for tfp lbfgs
    # @jax.value_and_grad
    # def func(param_vector):
    #     split_params = jnp.split(param_vector,
    #             np.cumsum([np.prod(s) for s in params_shape[:-1]]))
    #     flat_params = [x.reshape(s) for x, s in zip(split_params, params_shape)]
    #     params = jax.tree_util.tree_unflatten(params_tree, flat_params)
    #     return marginal_likelihood(params, x, y)

    # calculate marginal likelihood based on given parameters
    @jit
    def func(param_vector):
        split_params = jnp.split(param_vector,
                np.cumsum([np.prod(s) for s in params_shape[:-1]]).astype(int))
        flat_params = [x.reshape(s) for x, s in zip(split_params, params_shape)]
        params = jax.tree_util.tree_unflatten(params_tree, flat_params)
        return marginal_likelihood(params, x, y)

    ## for tfp lbfgs
    # results = tfp.optimizer.lbfgs_minimize(
    #     jax.jit(func), initial_position=param_vector, tolerance=1e-8, max_iterations=1000)

    # perform optimization
    results = sp_minimize(func, x0=param_vector, method='L-BFGS-B', tol=1e-5, options={'maxiter': 1000, 'gtol': 1e-5, })
    
    # convert 1-d optimized parameter array into its original shape
    split_params = jnp.split(results.x, #position,
                np.cumsum([np.prod(s) for s in params_shape[:-1]]).astype(int))
    flat_params = [x.reshape(s) for x, s in zip(split_params, params_shape)]
    params = jax.tree_util.tree_unflatten(params_tree, flat_params)
    
    return results, params


EARTH_RADIUS_METERS = 6366565.0

def get_window(needle, lat, lon, window):
    """Find indices of lat-lon points that are within the window centered at needle
    """
    dlat = jnp.radians(lat) - jnp.radians(needle[0])
    dlon = jnp.radians(lon) - jnp.radians(needle[1])
    a = jnp.square(jnp.sin(dlat / 2.0)) + jnp.cos(jnp.radians(needle[0])) * jnp.cos(jnp.radians(lat)) * jnp.square(jnp.sin(dlon / 2.0))
    great_circle_distance = 2 * jnp.arcsin(jnp.minimum(jnp.sqrt(a), jnp.repeat(1, len(a))))
    d = EARTH_RADIUS_METERS * great_circle_distance
    return d < window

def save_predictGP_dask( from_compute, ocean_mask, latg, long, maptimes, outfile, winsize_TS, winsize_BGC, correlated_nugget=True) -> None:
    
    # create grids for mu, var, and params
    gridshape=(latg.size, long.size, maptimes.size)
    data_anom = np.full(shape=gridshape, fill_value=np.nan)
    data_mse = np.full(shape=gridshape, fill_value=np.nan)

    gridshape=(latg.size, long.size)
    data_beta = np.full(shape=gridshape, fill_value=np.nan)
    data_sigmasq = np.full(shape=gridshape, fill_value=np.nan)
    data_sigmasq2 = np.full(shape=gridshape, fill_value=np.nan)
    data_noise = np.full(shape=gridshape, fill_value=np.nan)
    data_noise2 = np.full(shape=gridshape, fill_value=np.nan)
    if correlated_nugget:
        data_rho = np.full(shape=gridshape, fill_value=np.nan)

    gridshape=(latg.size, long.size, 3)
    data_lengthscale = np.full(shape=gridshape, fill_value=np.nan)
    data_lengthscale2 = np.full(shape=gridshape, fill_value=np.nan)

    iM,jM = np.where(ocean_mask.float_mask)
    
    for n, i, j in zip( range(iM.size), iM, jM ):
         if from_compute[n] is not None:
            data_anom[i,j,:] = from_compute[n]['mu']
            data_mse[i,j,:]  = from_compute[n]['var']
            data_beta[i,j] =  2 * sigmoid(from_compute[n]['params']['beta']) - 1 
            data_sigmasq[i,j] = softplus(from_compute[n]['params']['sigmasq'])
            data_sigmasq2[i,j] = softplus(from_compute[n]['params']['sigmasq2'])
            data_noise[i,j] = softplus(from_compute[n]['params']['noise'])
            data_noise2[i,j] = softplus(from_compute[n]['params']['noise2'])
            data_lengthscale[i,j,:] = softplus(from_compute[n]['params']['lengthscale'])
            data_lengthscale2[i,j,:] = softplus(from_compute[n]['params']['lengthscale2'])
            if correlated_nugget:
                data_rho[i,j] = 2 * sigmoid(from_compute[n]['params']['rho']) - 1 

  

 #    'beta': array([[0.19064483, 0.20129465]]),
 # 'sigmasq': array([[23.18766355, 23.80264997]]),
 # 'sigmasq2': array([[25168.56554935, 25672.1202998 ]]),
 # 'noise': array([[-40.97887075, -36.4061391 ]]),
 # 'noise2': array([[ -82.55610945, -240.03522239]]),
 # 'lengthscale': array([[[ 0.56411947,  2.19177571, 87.53315021],
 #         [ 0.5667759 ,  2.21648898, 89.13597941]]]),
 # 'lengthscale2': array([[[  0.51315608,   2.97241915, 112.85244074],
 #         [  0.51445395,   2.73897454, 127.65879453]]]),
 # 'rho': array([[0.35022219, 0.38985546]]),



    
    # create xr.dataset and save to outfile here:
    if correlated_nugget:
        ds = xr.Dataset(
            data_vars = dict(
                anom = ( ['latitude', 'longitude', 'time'], data_anom ),
                mse = ( ['latitude', 'longitude', 'time'], data_mse ),
                beta = ( ['latitude', 'longitude'], data_beta ),
                sigmasq = ( ['latitude', 'longitude'], data_sigmasq ),
                sigmasq2 = ( ['latitude', 'longitude'], data_sigmasq2 ),
                noise = ( ['latitude', 'longitude'], data_noise ),
                noise2 = ( ['latitude', 'longitude'], data_noise2 ),
                lengthscale = ( ['latitude', 'longitude', 'd'], data_lengthscale ),
                lengthscale2 = ( ['latitude', 'longitude', 'd'], data_lengthscale2 ),
                rho = ( ['latitude', 'longitude'], data_rho ),
            ),
            coords=dict(
                time = ( ['time'], maptimes ),
                latitude = ( ['latitude'], latg ),
                longitude = ( ['longitude'], long ),
                d = (['d'], np.arange(3) )
                #pressure = (['pressure'], c.pressure[:].values )
            ),
            attrs=dict(
                description = 'Gridded estimated anomalies and errors + optimized covariance parameters',
                window_size_TS = winsize_TS,
                window_size_BGC = winsize_BGC,
                filename=outfile,
                #created_date = str( datetime.now() )
            )
        )
    else:
        ds = xr.Dataset(
            data_vars = dict(
                anom = ( ['latitude', 'longitude', 'time'], data_anom ),
                mse = ( ['latitude', 'longitude', 'time'], data_mse ),
                beta = ( ['latitude', 'longitude'], data_beta ),
                sigmasq = ( ['latitude', 'longitude'], data_sigmasq ),
                sigmasq2 = ( ['latitude', 'longitude'], data_sigmasq2 ),
                noise = ( ['latitude', 'longitude'], data_noise ),
                noise2 = ( ['latitude', 'longitude'], data_noise2 ),
                lengthscale = ( ['latitude', 'longitude', 'd'], data_lengthscale ),
                lengthscale2 = ( ['latitude', 'longitude', 'd'], data_lengthscale2 ),
            ),
            coords=dict(
                time = ( ['time'], maptimes ),
                latitude = ( ['latitude'], latg ),
                longitude = ( ['longitude'], long ),
                d = (['d'],  np.arange(3) )
                #pressure = (['pressure'], c.pressure[:].values )
            ),
            attrs=dict(
                description = 'Gridded estimated anomalies and errors + optimized covariance parameters',
                window_size_TS = winsize_TS,
                window_size_BGC = winsize_BGC,
                filename=outfile,
                #created_date = str( datetime.now() )
            )
        )
    
    ds.anom.attrs["standard_name"] = 'Estimated  anomalies'
    ds.time.attrs["standard_name"] = 'time'
    ds.time.attrs["units"] = 'days since 1970-01-01 00:00:00'
    #ds.latitude.attrs = c.latitude.attrs
    #ds.longitude.attrs= c.longitude.attrs
    #ds.pressure.attrs = c.pressure.attrs

    # Save dataset
    ds.to_netcdf(outfile)
    del from_compute, ocean_mask, latg, long, maptimes, outfile, winsize_TS, winsize_BGC, ds

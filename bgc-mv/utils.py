import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
# from jax.scipy.optimize import minimize
from scipy.optimize import minimize as sp_minimize
import scipy as sp

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
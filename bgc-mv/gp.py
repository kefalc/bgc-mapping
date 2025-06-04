import jax.numpy as jnp
from jax import vmap
import jax.scipy as scipy
from jax import config
import scipy as sp
from utils import matern32, det, softplus, sigmoid
from functools import partial
config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)


def KN_univ_cov_map(ls, sigmasq, x, y=None):
    '''obtain covariance matrix given lengthscales and sigmasq'''
    if y is None:
        return sigmasq / det(ls) * vmap(vmap(matern32, (0, None)),
                                        (None, 0))(x / ls, x / ls)
    else:
        return sigmasq / det(ls) * vmap(vmap(matern32, (0, None)),
                                        (None, 0))(x / ls, y / ls).T


def KN_univ_gp(params,
               x,
               y,
               xtest=None,
               compute_marginal_likelihood=False,
               noiseless=False):
    '''Run univariate gaussian process model

    params: dict of parameter pairs ('param_name' : jnp.array)
    x: n-by-3 jnp.array, the first column is latitude, the second is longitude and the third is number of days since 1970.1.1
    y: n-by-1 jnp.array, temp/sal observations
    xtest: n'-by-3 jnp.array or None

    If compute_marginal_likelihood is True, return negative marginal log likelihood for optimization;
    else return predicted mean and var

    If noiseless is True, include noise parameter in the model
    '''
    # transform raw parameters to positive using sofrplus
    if not noiseless:
        noise = softplus(params['noise'])
    ls = softplus(params['lengthscale'])
    sigmasq = softplus(params['sigmasq'])
    d = x.shape[0]
    xmean = jnp.mean(x, axis=0)
    x = x - xmean
    eye = jnp.eye(d)
    # add noise to diagonal entries if noiseless=False
    if noiseless:
        train_cov = KN_univ_cov_map(ls, sigmasq, x)
    else:
        train_cov = KN_univ_cov_map(ls, sigmasq, x) + eye * (noise + 1e-6)

    chol = scipy.linalg.cholesky(train_cov, lower=True)
    kinvy = scipy.linalg.solve_triangular(
        chol.T, scipy.linalg.solve_triangular(chol, y, lower=True))
    if compute_marginal_likelihood:
        log2pi = jnp.log(2. * 3.1415)
        ml = jnp.sum(-0.5 * jnp.dot(y.T, kinvy) -
                     jnp.sum(jnp.log(jnp.diag(chol))) - (d / 2.) * log2pi)
        return -ml

    if xtest is not None:
        xtest = xtest - xmean
    cross_cov = KN_univ_cov_map(ls, sigmasq, x, xtest)
    mu = jnp.dot(cross_cov.T, kinvy)
    v = scipy.linalg.solve_triangular(chol, cross_cov, lower=True)
    var = (KN_univ_cov_map(ls, sigmasq, xtest) - jnp.dot(v.T, v))
    return mu, var


# http://gpss.cc/gpss17/slides/multipleOutputGPs.pdf
# covariance fitting and prediction
# https://www.sciencedirect.com/science/article/pii/S0047259X12001376
# Nonstationary modeling for multivariate spatial processes, Kleiber and Nychka 2012
def KN_bivar_cov_map(ls1, sigmasq1, ls2, sigmasq2, beta, x, y=None):
    '''obtain covariance matrix given lengthscales and sigmasq's
    based on Kleiber and Nychka 2012'''
    ls12 = jnp.sqrt((ls1**2 + ls2**2) / 2.0)  # cross covariance lengthscale
    if y is None:
        mar_cov1 = sigmasq1 / det(ls1) * vmap(vmap(matern32, (0, None)),
                                              (None, 0))(x / ls1, x / ls1)
        mar_cov2 = sigmasq2 / det(ls2) * vmap(vmap(matern32, (0, None)),
                                              (None, 0))(x / ls2, x / ls2)
        cross_cov = beta * jnp.sqrt(sigmasq1 * sigmasq2) / det(ls12) * \
            vmap(vmap(matern32, (0, None)), (None, 0))(x / ls12, x / ls12)
        kxx = jnp.block([[mar_cov1, cross_cov], [cross_cov.T, mar_cov2]])
        return kxx
    else:
        mar_cov1_xy = sigmasq1 / det(ls1) * vmap(vmap(matern32, (0, None)),
                                                 (None, 0))(x / ls1, y / ls1).T
        mar_cov2_xy = sigmasq2 / det(ls2) * vmap(vmap(matern32, (0, None)),
                                                 (None, 0))(x / ls2, y / ls2).T
        cross_cov_xy = beta * jnp.sqrt(sigmasq1 * sigmasq2) / det(ls12) * \
            vmap(vmap(matern32, (0, None)), (None, 0))(x / ls12, y / ls12).T
        kxy = jnp.block([[mar_cov1_xy, cross_cov_xy],
                         [cross_cov_xy, mar_cov2_xy]])
        return kxy


def KN_bivariate_gp(params,
                    x,
                    y,
                    fixed_params=None,
                    xtest=None,
                    compute_marginal_likelihood=False,
                    noiseless=False,
                    beta=None,
                    correlated_nugget=False,
                    nugget_prime=False):
    '''Run bivariate gaussian process model based on Kleiber and Nychka 2012

    params: dict of parameters ('param_name' : jnp.array)
    x: n-by-3 jnp.array, the first column is latitude, the second is longitude and the third is number of days since 1970.1.1
    y: n-by-2 jnp.array, the first column corresponds to temp/sal and the second corresponds to a BGC variable
    fixed_params: dict of fixed parameters obtained from univariate gp
    xtest: n'-by-3 jnp.array or None

    If compute_marginal_likelihood is True, return negative marginal log likelihood for optimization;
    else return predicted mean and var

    If noiseless is True, include noise2 in the model

    If beta is None, include beta as a free parameter; otherwise, treat it as a constant
    
    If correlated_nugget is True, include a parameter rho for correlation between the two nuggets

    If correlated_nugget and nugget_prime are both True, include arameters
    noise_prime and noise2_prime to capture the uncorrelated nuggets, while noise and noise2 are
    for the correlated components; If correlated_nugget is True and nugget_prime is False, noise and
    noise2 are for uncorrelated components
    '''
    # use softplus transform raw parameters that are supposed to be nonnegative 
    # use 2 * sigmoid(...) - 1 to transform raw parameters that are supposed to be within [-1, 1]
    if not noiseless:
        noise2 = softplus(params['noise2'])
        if correlated_nugget:
            rho = 2 * sigmoid(params['rho']) - 1
            if nugget_prime:
                noise1 = softplus(params['noise'])
                noise1_prime = softplus(params['noise_prime'])
                noise2_prime = softplus(params['noise2_prime'])
            else:
                noise1 = softplus(fixed_params['noise'])

        else:
            noise1 = softplus(fixed_params['noise'])
    ls1 = softplus(fixed_params['lengthscale'])
    ls2 = softplus(params['lengthscale2'])
    sigmasq1 = softplus(fixed_params['sigmasq'])
    sigmasq2 = softplus(params['sigmasq2'])
    if beta is None:
        beta = 2 * sigmoid(params['beta']) - 1
    p = y.shape[1]
    y = jnp.reshape(y, -1, order='F')
    xmean = jnp.mean(x, axis=0)
    x = x - xmean
    if not noiseless:
        eye = jnp.eye(x.shape[0])
        if correlated_nugget:
            corr = rho * jnp.sqrt(noise1 * noise2) * eye
            if nugget_prime:
                nugget = jnp.block(
                    [[(noise1+noise1_prime) * eye, corr], [corr, (noise2 + noise2_prime) * eye]])
            else:
                nugget = jnp.block(
                    [[noise1 * eye, corr], [corr, noise2 * eye]])
        else:
            nugget = jnp.kron(jnp.diag(jnp.hstack([noise1, noise2])), eye)
        train_cov = KN_bivar_cov_map(
            ls1, sigmasq1, ls2, sigmasq2, beta, x) + nugget
    else:
        train_cov = KN_bivar_cov_map(ls1, sigmasq1, ls2, sigmasq2, beta, x)

    chol = scipy.linalg.cholesky(train_cov, lower=True)
    kinvy = scipy.linalg.solve_triangular(
        chol.T, scipy.linalg.solve_triangular(chol, y, lower=True))
    if compute_marginal_likelihood:
        log2pi = jnp.log(2. * 3.1415)
        ml = jnp.sum(-0.5 * jnp.dot(y.T, kinvy) -
                     jnp.sum(jnp.log(jnp.diag(chol))) -
                     (x.shape[0] * p / 2.) * log2pi)
        return -ml

    if xtest is not None:
        xtest = xtest - xmean
    cross_cov = KN_bivar_cov_map(ls1, sigmasq1, ls2, sigmasq2, beta, x, xtest)
    mu = jnp.dot(cross_cov.T, kinvy)
    v = scipy.linalg.solve_triangular(chol, cross_cov, lower=True)
    var = (KN_bivar_cov_map(ls1, sigmasq1, ls2, sigmasq2, beta, xtest) -
           jnp.dot(v.T, v))
    mu = mu.reshape((-1, p), order='F')
    return mu, var


def KN_bivar_cov_map_predict(ls1,
                             sigmasq1,
                             ls2,
                             sigmasq2,
                             beta,
                             x1,
                             x2,
                             y=None,
                             component=11,
                             both=False):
    '''obtain covariance matrix for prediction given lengthscales and sigmasq's
    based on Kleiber and Nychka 2012
    
    If component is 11 or 22, return covariance matrix for variable 1 or 2
    If component is 12 or 21, return cross-covariance matrix

    If both is True, return covariance matrix for both variables
    Else, return covariance matrix for variable 2
    '''
    ls12 = jnp.sqrt((ls1**2 + ls2**2) / 2.0)
    if y is None:
        if component == 11:
            k11 = sigmasq1 / det(ls1) * vmap(vmap(matern32, (0, None)),
                                             (None, 0))(x1 / ls1, x1 / ls1)
            k22 = sigmasq2 / det(ls2) * vmap(vmap(matern32, (0, None)),
                                             (None, 0))(x2 / ls2, x2 / ls2)
            k12 = beta * jnp.sqrt(sigmasq1 * sigmasq2) / det(ls12) * \
                vmap(vmap(matern32, (0, None)), (None, 0))(
                    x1 / ls12, x2 / ls12).T
            k = jnp.block([[k11, k12], [k12.T, k22]])
        else:
            raise RuntimeError('wrong cov type')
        return k
    else:
        if both:  # predict variable 1 and variable 2
            if component == 22:
                k11 = sigmasq1 / det(ls1) * vmap(vmap(matern32, (0, None)),
                                                 (None, 0))(y / ls1, y / ls1)
                k22 = sigmasq2 / det(ls2) * vmap(vmap(matern32, (0, None)),
                                                 (None, 0))(y / ls2, y / ls2)
                k12 = beta * jnp.sqrt(sigmasq1 * sigmasq2) / det(ls12) * \
                    vmap(vmap(matern32, (0, None)), (None, 0))(
                        y / ls12, y / ls12).T
                k = jnp.block([[k11, k12], [k12.T, k22]])
            elif component == 12:
                k11 = sigmasq1 / det(ls1) * vmap(vmap(
                    matern32, (0, None)), (None, 0))(x1 / ls1, y / ls1).T
                k22 = sigmasq2 / det(ls2) * vmap(vmap(
                    matern32, (0, None)), (None, 0))(x2 / ls2, y / ls2).T
                k12 = beta * jnp.sqrt(sigmasq1 * sigmasq2) / det(ls12) * \
                    vmap(vmap(matern32, (0, None)), (None, 0))(
                        x1 / ls12, y / ls12).T
                k21 = beta * jnp.sqrt(sigmasq1 * sigmasq2) / det(ls12) * \
                    vmap(vmap(matern32, (0, None)), (None, 0))(
                        x2 / ls12, y / ls12).T
                k = jnp.block([[k11, k12], [k21, k22]])
            elif component == 21:
                k11 = sigmasq1 / det(ls1) * vmap(vmap(
                    matern32, (0, None)), (None, 0))(y / ls1, x1 / ls1).T
                k22 = sigmasq2 / det(ls2) * vmap(vmap(
                    matern32, (0, None)), (None, 0))(y / ls2, x2 / ls2).T
                k21 = beta * jnp.sqrt(sigmasq1 * sigmasq2) / det(ls12) * \
                    vmap(vmap(matern32, (0, None)), (None, 0))(
                        y / ls12, x1 / ls12).T
                k12 = beta * jnp.sqrt(sigmasq1 * sigmasq2) / det(ls12) * \
                    vmap(vmap(matern32, (0, None)), (None, 0))(
                        y / ls12, x2 / ls12).T
                k = jnp.block([[k11, k12], [k21, k22]])
            else:
                raise RuntimeError('wrong cov type')
        else:  # predict variable 2 only
            if component == 22:
                k = sigmasq2 / det(ls2) * vmap(vmap(matern32, (0, None)),
                                               (None, 0))(y / ls2, y / ls2)
            elif component == 12:
                k22 = sigmasq2 / det(ls2) * vmap(vmap(
                    matern32, (0, None)), (None, 0))(x2 / ls2, y / ls2).T
                k12 = beta * jnp.sqrt(sigmasq1 * sigmasq2) / det(ls12) * \
                    vmap(vmap(matern32, (0, None)), (None, 0))(
                        x1 / ls12, y / ls12).T
                k = jnp.block([[k12], [k22]])
            elif component == 21:
                k22 = sigmasq2 / det(ls2) * vmap(vmap(
                    matern32, (0, None)), (None, 0))(y / ls2, x2 / ls2).T
                k21 = beta * jnp.sqrt(sigmasq1 * sigmasq2) / det(ls12) * \
                    vmap(vmap(matern32, (0, None)), (None, 0))(
                        y / ls12, x1 / ls12).T
                k = jnp.block([k21, k22])
            else:
                raise RuntimeError('wrong cov type')
        return k


def predict(params,
            xbi,
            ybi,
            xuni,
            yuni,
            xtest,
            noiseless=False,
            beta=None,
            var=False,
            both=False,
            correlated_nugget=False,
            nugget_prime=False):
    """
    parameters
    xbi: m-by-d, data points where both observations are present
    ybi: m-by-2, outcome observations with both fields are present
    xuni: n-by-d, data points where only temp are present
    yuni: n-by-1, outcome observations with only temp are present
    xtest: m*-by-d, new locations
    """
    m = xbi.shape[0]
    n = xuni.shape[0]
    y = jnp.hstack([ybi[:, 0], yuni.reshape(-1), ybi[:, 1]])
    x1 = jnp.vstack([xbi, xuni])
    x2 = xbi
    if not noiseless:
        noise2 = softplus(params['noise2'])
        if correlated_nugget:
            rho = 2 * sigmoid(params['rho']) - 1
            if nugget_prime:
                noise1_prime = softplus(params['noise_prime'])
                noise2_prime = softplus(params['noise2_prime'])
        noise1 = softplus(params['noise'])
    ls1 = softplus(params['lengthscale'])
    ls2 = softplus(params['lengthscale2'])
    sigmasq1 = softplus(params['sigmasq'])
    sigmasq2 = softplus(params['sigmasq2'])
    if beta is None:
        beta = 2 * sigmoid(params['beta']) - 1
    p = 2
    if not noiseless:
        if correlated_nugget:
            corr = jnp.eye(n + m, m, -n) * rho * jnp.sqrt(noise1 * noise2)
            if nugget_prime:
                nugget = jnp.block([[(noise1+noise1_prime) * jnp.eye(n + m), corr],
                                   [corr.T, (noise2 + noise2_prime) * jnp.eye(m)]])
            else:
                nugget = jnp.block(
                    [[noise1 * jnp.eye(n + m), corr], [corr.T, noise2 * jnp.eye(m)]])
        else:
            nugget = jnp.diag(jnp.hstack(
                [jnp.ones(n + m) * noise1, jnp.ones(m) * noise2]))
        sigmaXX = KN_bivar_cov_map_predict(
            ls1, sigmasq1, ls2, sigmasq2, beta, x1, x2) + nugget
    else:
        sigmaXX = KN_bivar_cov_map_predict(
            ls1, sigmasq1, ls2, sigmasq2, beta, x1, x2)

    chol = sp.linalg.cholesky(sigmaXX, lower=True) # sp or scipy
    kinvy = scipy.linalg.solve_triangular(
        chol.T, scipy.linalg.solve_triangular(chol, y, lower=True))
    sigmaXY = KN_bivar_cov_map_predict(ls1,
                                       sigmasq1,
                                       ls2,
                                       sigmasq2,
                                       beta,
                                       x1,
                                       x2,
                                       xtest,
                                       component=12,
                                       both=both)
    sigmaYX = KN_bivar_cov_map_predict(ls1,
                                       sigmasq1,
                                       ls2,
                                       sigmasq2,
                                       beta,
                                       x1,
                                       x2,
                                       xtest,
                                       component=21,
                                       both=both)
    mu = jnp.dot(sigmaYX, kinvy)
    if both:
        mu = mu.reshape((-1, p), order='F')
    if not var:
        return mu
    else:
        v = scipy.linalg.solve_triangular(chol, sigmaXY, lower=True)
        var = (KN_bivar_cov_map_predict(ls1,
                                        sigmasq1,
                                        ls2,
                                        sigmasq2,
                                        beta,
                                        None,
                                        None,
                                        xtest,
                                        component=22,
                                        both=both) - jnp.dot(v.T, v))
        var = jnp.diag(var)
        # mu = mu.reshape((-1, p), order='F')
        return mu, var


# jit before use
# params: noiseless (default False)
KN_univ_marginal_likelihood = partial(KN_univ_gp,
                                      compute_marginal_likelihood=True)
KN_univ_predict = partial(KN_univ_gp, 
                          compute_marginal_likelihood=False)
# jit before use
# params: noiseless (default False)
#         beta (default None)
#         fixed_params
KN_bivariate_marginal_likelihood = partial(KN_bivariate_gp,
                                           compute_marginal_likelihood=True)
KN_bivariate_predict_complete_obs = partial(KN_bivariate_gp,
                                            compute_marginal_likelihood=False)

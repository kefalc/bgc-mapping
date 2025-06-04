import xarray as xr
import numpy as np
from datetime import datetime
import pandas as pd
import jax
import jax.extend
import jax.numpy as jnp
from functools import partial
import time
from tqdm import tqdm
from utils import softplus, sigmoid, det, BFGS_training, get_window
from gp import KN_univ_gp, KN_bivariate_gp, predict, KN_univ_marginal_likelihood, KN_bivariate_marginal_likelihood


KN_univ_marginal_likelihood = partial(KN_univ_marginal_likelihood, noiseless=False)
KN_bivariate_marginal_likelihood = partial(KN_bivariate_marginal_likelihood, noiseless=False)


def filter_data(filename_TS,
                filename_BGC):
    '''filter out observations that are too close in time'''

    anomalies_BGC = xr.open_dataset(filename_BGC)
    anomalies_TS = xr.open_dataset(filename_TS)

    # extract float_id from prof_descr and decode them to ASCII
    anomalies_BGC['float_id'] = anomalies_BGC.prof_descr.str.extract(
        b'(?<=//)([A-Za-z0-9]+)(?=-)', dim=None)
    anomalies_TS['float_id'] = anomalies_TS.prof_descr.str.extract(
        b'(?<=//)([A-Za-z0-9]+)(?=-)', dim=None)
    # https://stackoverflow.com/questions/6269765/
    anomalies_TS['float_id'] = anomalies_TS.float_id.str.decode('ASCII')
    anomalies_BGC['float_id'] = anomalies_BGC.float_id.str.decode('ASCII')

    # turn date from float to str
    anomalies_BGC['prof_YYYYMMDD'] = anomalies_BGC.prof_YYYYMMDD.astype(
        int).astype(str)
    anomalies_TS['prof_YYYYMMDD'] = anomalies_TS.prof_YYYYMMDD.astype(
        int).astype(str)

    # turn date into days since 1970.1.1 and sort
    days_BGC = (pd.to_datetime(
        anomalies_BGC['prof_YYYYMMDD'].values) - datetime(1970, 1, 1)).days.values
    anomalies_BGC = anomalies_BGC.assign({
        'prof_days': ('iPROF', days_BGC)
    })
    anomalies_BGC = anomalies_BGC.sortby(anomalies_BGC.prof_days)

    days_TS = (pd.to_datetime(
        anomalies_TS['prof_YYYYMMDD'].values) - datetime(1970, 1, 1)).days.values
    anomalies_TS = anomalies_TS.assign({
        'prof_days': ('iPROF', days_TS)
    })
    anomalies_TS = anomalies_TS.sortby(anomalies_TS.prof_days)

    # pick out observations that are at lease 4 days apart from the previous ones
    subsampled = []
    count = 0
    for id in np.unique(anomalies_BGC.float_id.values).flatten():
        indices = np.argwhere(anomalies_BGC.float_id.values == id).flatten()
        days = anomalies_BGC.prof_days[indices].values
        count += len(indices)
        left = 0
        subsampled.append(indices[0])
        for j in range(1, days.shape[0]):
            if abs(days[j] - days[left]) >= 3.5:
                left = j
                subsampled.append(indices[j])
    anomalies_BGC = anomalies_BGC.sel(iPROF=anomalies_BGC.iPROF[subsampled])

    subsampled = []
    count = 0
    for id in np.unique(anomalies_TS.float_id.values).flatten():
        indices = np.argwhere(anomalies_TS.float_id.values == id).flatten()
        days = anomalies_TS.prof_days[indices].values
        count += len(indices)
        left = 0
        subsampled.append(indices[0])
        for j in range(1, days.shape[0]):
            if abs(days[j] - days[left]) >= 3.5:
                left = j
                subsampled.append(indices[j])
    anomalies_TS = anomalies_TS.sel(iPROF=anomalies_TS.iPROF[subsampled])

    return anomalies_TS, anomalies_BGC


# funcitons to perform modelling tasks

def prepare_data(anomalies_TS, anomalies_BGC, TS, BGC, depth, outputfile):
    '''Based on given anomalies data in .nc file, extract necessary data at given depth and save them to npz file
    
    parameters:
    anomalies_TS and anomalies_BGC: .nc files containing temp/sal and BGC anomalies respectively
    TS: str, 'T' for temperature anomalies and 'S' for salinity anomalies
    BGC: str, choices include 'O2anomaly', 'NO3', 'CHL', 'DIC', 'POC' and 'PH' or any other BGC variables that can be found in anomalies_BGC
    depth: int, depth at which we want to perform modelling
    outputfile: str, filename; must end with '.npz'
    '''
    y_TS = anomalies_TS['anom_' + TS]
    lat = anomalies_TS['prof_lat']
    lon = anomalies_TS['prof_lon']
    date = anomalies_TS['prof_YYYYMMDD']
    y_TS = y_TS.sel(iDEPTH=depth)
    mask = y_TS.notnull()
    y_TS = y_TS.where(mask, drop=True)
    lat_TS = jnp.array(lat.where(mask, drop=True).values)

    lon_TS = jnp.array(lon.where(mask, drop=True).values)
    t_ = date.where(mask, drop=True)
    t_TS = jnp.array([(datetime.strptime(str(int(x)), '%Y%m%d') - datetime(1970, 1, 1)).days
                for x in t_])  
    y_TS = jnp.array(y_TS).reshape((-1, 1))

    BGC = anomalies_BGC['anom_' + BGC]
    TS = anomalies_BGC['anom_' + TS]
    lat = anomalies_BGC['prof_lat']
    lon = anomalies_BGC['prof_lon']
    date = anomalies_BGC['prof_YYYYMMDD']
    BGC = BGC.sel(iDEPTH=depth)
    TS = TS.sel(iDEPTH=depth)
    mask = BGC.notnull() * TS.notnull()
    BGC = BGC.where(mask, drop=True)
    TS = TS.where(mask, drop=True)
    lat_BGC = jnp.array(lat.where(mask, drop=True).values)
    lon_BGC = jnp.array(lon.where(mask, drop=True).values)
    t_ = date.where(mask, drop=True)
    t_BGC = jnp.array([(datetime.strptime(str(int(x)), '%Y%m%d') - datetime(1970, 1, 1)).days
                for x in t_])  
    y_BGC = jnp.vstack([TS, BGC]).T

    # note: consider jax.numpy.savez to save these arrays
    data = {'lat_TS': lat_TS, 'lon_TS': lon_TS, 
            't_TS': t_TS, 'y_TS': y_TS, 'lat_BGC': lat_BGC, 
            'lon_BGC': lon_BGC, 't_BGC': t_BGC, 'y_BGC': y_BGC}
    jnp.savez(outputfile, **data)
    print('done')


def pipeline(x_TS, y_TS, x_BGC, y_BGC, 
             init_params, bivar_init_params=None,
             return_params=False,
             correlated_nugget=False,
             nugget_prime=False,
             return_estim=True,
             estimtimes=None,
             ):
    '''pipeline to perform univ and bivar gps on a single grid point with preprocessed data

    Return:
    params (dict of jnp.arrays) or None (if error occurs)

    If return_estim is True, estimated mean, estimation variance and params are return in one dict
    '''
    # calculate empirical correlation and variances
    empirical_corr = np.corrcoef(y_BGC[:, 0], y_BGC[:, 1])[0, 1]
    empirical_var = np.var(y_BGC, axis=0)
    print(f'empirical_corr: {empirical_corr}', f'\tempirical_var: {empirical_var[0]}  {empirical_var[1]}')

    # add 'try except' to avoid erroring out because of linear algebra error (mainly due to the data itself)
    # return None if such error is raised
    try:
        results_TS, params_TS = BFGS_training(init_params, x_TS, y_TS, KN_univ_marginal_likelihood)
    except jax.extend.linear_util.StoreException as se:
        print(se)
        print()
        print()
        return None
    except FloatingPointError as fe:
        print(fe)
        print()
        print()
        return None
    
    # extracting parameters from univariate gp gor temp/sal
    univ_noise = params_TS['noise']
    univ_sigmasq = params_TS['sigmasq']
    univ_lengthscale = params_TS['lengthscale']

    # alpha_ is the ratio of estimated variance to empirical variance
    # theta_ is the ratio of noise to empirical variance
    # these are useful for determining initial values for sigmasq2 and noise2
    alpha = (softplus(univ_sigmasq) / det(softplus(univ_lengthscale)) + \
             softplus(univ_noise)) / empirical_var[0]
    theta = softplus(univ_noise) / softplus(univ_sigmasq) * det(softplus(univ_lengthscale))
    theta_ = alpha * theta / (1 + theta) 
    alpha_ = alpha * det(softplus(univ_lengthscale)) / (1 + theta)
    if correlated_nugget:
        params_BGC = {'noise2': jnp.array(empirical_var[1]) * theta_, 
                    'sigmasq2': alpha_ * jnp.array(empirical_var[1]),
                    'lengthscale2': params_TS['lengthscale'],
                    'beta': jnp.array(empirical_corr),
                    'rho': jnp.array(empirical_corr),
        } if bivar_init_params is None else bivar_init_params
        if nugget_prime:
            params_BGC['noise2_prime'] =params_TS['noise'] * alpha
            params_BGC['noise'] = params_TS['noise'] * alpha
            params_BGC['noise_prime'] = params_TS['noise'] * alpha
    else:
        params_BGC = {'noise2': params_TS['noise'] * alpha, #params_TS['noise'], 
                    'sigmasq2': alpha * jnp.array(empirical_var[1]),  # empirical var = \alpha (univ _sigmasq / det(...) + univ_noise^2)
                    'lengthscale2': params_TS['lengthscale'],
                    'beta': jnp.array(empirical_corr),
        } if bivar_init_params is None else bivar_init_params
    
    
    bivariate_marginal_likelihood = partial(KN_bivariate_marginal_likelihood, 
                                            fixed_params=params_TS, 
                                            correlated_nugget=correlated_nugget, 
                                            nugget_prime=nugget_prime)
    
    # add 'try except' to avoid erroring out because of linear algebra error (mainly due to the data itself)
    # return None if such error is raised
    try:
        results_BGC, params_BGC = BFGS_training(params_BGC, x_BGC, y_BGC, bivariate_marginal_likelihood)
    except jax.extend.linear_util.StoreException as se:
        print(se)
        print()
        print()
        return None
    except FloatingPointError as fe:
        print(fe)
        print()
        print()
        return None
    
    print(f'univ converged: {results_TS.success}', 
          f'\tbivar converged: {results_BGC.success}')
    
    # print estimated correlation and variances
    factor = jnp.pi**0.5 / 2**(2 - 3/2)
    est_beta = 2 * sigmoid(params_BGC['beta']) - 1
    est_sigmasq_BGC, est_sigmasq_TS = softplus(params_BGC['sigmasq2']), softplus(params_TS['sigmasq'])
    est_noise_BGC, est_noise_TS = softplus(params_BGC['noise2']), softplus(params_TS['noise'])
    est_ls_BGC, est_ls_TS = softplus(params_BGC['lengthscale2']), softplus(params_TS['lengthscale'])
    est_var_TS = factor * est_sigmasq_TS / det(est_ls_TS) + est_noise_TS
    est_var_BGC = factor * est_sigmasq_BGC / det(est_ls_BGC) + est_noise_BGC
    if correlated_nugget:
        est_rho = 2 * sigmoid(params_BGC['rho']) - 1
        print(f"rho: {est_rho}")
        normal = jnp.sqrt(est_sigmasq_BGC + est_noise_BGC) * jnp.sqrt(est_sigmasq_TS + est_noise_TS)
        est_corr = (est_sigmasq_BGC**0.5 * est_sigmasq_TS**0.5 * est_beta + est_noise_BGC**0.5 * est_noise_TS**0.5 * est_rho) / normal
    else:
        normal = jnp.sqrt(est_sigmasq_BGC + est_noise_BGC) * jnp.sqrt(est_sigmasq_TS + est_noise_TS)
        est_corr = (est_sigmasq_BGC**0.5 * est_sigmasq_TS**0.5 * est_beta) / normal
    
    print('estimated beta: ', est_beta)
    print(f'estimated_corr: {est_corr}', f'\testimated_sigmasq/det: {est_var_TS}  {est_var_BGC}', )
    print(f'')
    print()
    print()
    params = {**params_TS, **params_BGC}

    if not return_estim:
        return {'params': params}

    try:
        #days = (datetime.strptime('20160101', '%Y%m%d') - datetime(1970, 1, 1)).days
        days = (np.datetime64('2017-01-01') - np.datetime64('1970-01-01')) / np.timedelta64(1, 'ns') / 1e9 / 86400.0
        mu = np.zeros(len(estimtimes))
        var = np.zeros(len(estimtimes))
        # mu, var = predict(params, x_BGC, y_BGC, x_TS, y_TS, jnp.array([[0., 0., days]]), 
        #                   noiseless=False, var=True, both=False,
        #                   correlated_nugget=correlated_nugget, nugget_prime=nugget_prime)
        # display(mu)
        # display(var)
        for iday, maptime in enumerate(estimtimes):
            imu, ivar = predict(params, x_BGC, y_BGC, x_TS, y_TS, jnp.array([[0., 0., maptime]]), 
                          noiseless=False, var=True, both=False,
                          correlated_nugget=correlated_nugget, nugget_prime=nugget_prime)
            mu[iday] = imu[0]
            var[iday] = ivar[0, 0]
    except np.linalg.LinAlgError as err:
        print(err)
        print()
        print()
        return None
    #print(f'estimation_var: {var[0, 0]}')
    print()
    print()

    return {'mu': mu, 'var': var, 'params': params}


def task(center, return_params, lat_TS, lon_TS, t_TS, y_TS,
         lat_BGC, lon_BGC, t_BGC, y_BGC, return_estim=True,
         correlated_nugget=False, nugget_prime=False, winsize_TS=10., winsize_BGC=10., estimtimes=None):
    '''extract data within a the window plot defined by center, winsize_TS and winsize_BGC
    from given lat, lon and time data; run modeling pipeline and return results
    '''
    center_lat, center_lon = center
    idata_TS = get_window((center_lat, center_lon), lat_TS,
                    lon_TS, winsize_TS * 1e5)
    idata_BGC = get_window((center_lat, center_lon), lat_BGC,
                    lon_BGC, winsize_BGC * 1e5)

    print(center, f'\t#TS: {sum(idata_TS)}\t', f'#BGC: {sum(idata_BGC)}')
    if sum(idata_BGC) <= 10:
        return None

    y_TS = y_TS[idata_TS]
    lat_TS, lon_TS = lat_TS[idata_TS], lon_TS[idata_TS]
    t_TS = t_TS[idata_TS]

    y_BGC = y_BGC[idata_BGC]
    lat_BGC, lon_BGC = lat_BGC[idata_BGC], lon_BGC[idata_BGC]
    t_BGC = t_BGC[idata_BGC]

    lat_TS = lat_TS - center_lat
    lon_TS = lon_TS - center_lon
    x_TS = jnp.vstack((lat_TS, lon_TS, t_TS)).T

    lat_BGC = lat_BGC - center_lat
    lon_BGC = lon_BGC - center_lon
    x_BGC = jnp.vstack((lat_BGC, lon_BGC, t_BGC)).T
    
    # change these if needed
    init_params = {'noise': jnp.array(0.1), 
                   'sigmasq': jnp.array(1.),
                   'lengthscale': jnp.array([2., 3., 90.])}
    bivar_init_params = {'noise2': jnp.array(0.5), 
                         'sigmasq2': jnp.array(0.5),
                         'lengthscale2': jnp.array([1., 1., 10.]),
                         'beta': jnp.array(-1.)}

    res = pipeline(x_TS, y_TS, x_BGC, y_BGC, init_params, 
                   bivar_init_params=None,  # put bivar_init_params if we want fixed initial params
                   return_params=return_params,
                   return_estim=return_estim,
                   correlated_nugget=correlated_nugget, 
                   nugget_prime=nugget_prime,
                   estimtimes=estimtimes)

    return res


def gridded_tasks(lat_grid, lon_grid, depth, filename, ocean_mask, return_params=False, 
                  correlated_nugget=False, nugget_prime=False, return_estim=True, 
                  winsize_TS=10., winsize_BGC=10., estimtimes=None):
    '''extract raw data from .npz file (filename) generated from generate_data and perform modelling tasks on given 
    grid points defined by lat_grid and lon_grid sequentially
    
    params:
    lat_grid: 1-d array of latitude
    lon_grid: 1-d array of longitude
    ocean_mask: .nc file of ocean mask

    Note: depth is not needed in current implementation

    return:
    dict of jnp.arrays: fitted parameters at requested grid points
    '''
    # set up data containers to store fitted parameters
    data = jnp.load(filename)
    if return_estim:
        estim = np.zeros((len(lat_grid), len(lon_grid), len(estimtimes)))
        var = np.zeros((len(lat_grid), len(lon_grid), len(estimtimes)))
    beta = np.zeros((len(lat_grid), len(lon_grid)))
    sigmasq = np.zeros((len(lat_grid), len(lon_grid)))
    sigmasq2 = np.zeros((len(lat_grid), len(lon_grid)))
    noise = np.zeros((len(lat_grid), len(lon_grid)))
    noise2 = np.zeros((len(lat_grid), len(lon_grid)))
    
    if correlated_nugget:
        rho = np.zeros((len(lat_grid), len(lon_grid)))
        if nugget_prime:
            noise_prime = np.zeros((len(lat_grid), len(lon_grid)))
            noise2_prime = np.zeros((len(lat_grid), len(lon_grid)))
    
    lengthscale = np.zeros((len(lat_grid), len(lon_grid), 3))
    lengthscale2 = np.zeros((len(lat_grid), len(lon_grid), 3))
    # np.zeros((len(depth), len(lat_grid), len(lon_grid)))
    for ilat, lat in tqdm(enumerate(lat_grid)):
        for ilon, lon in tqdm(enumerate(lon_grid)):
            res = task((lat, lon), return_params, **data, return_estim=return_estim, correlated_nugget=correlated_nugget,
                       nugget_prime=nugget_prime, winsize_TS=winsize_TS,  winsize_BGC=winsize_BGC,  estimtimes=estimtimes)
            # res is None means error occured during training task
            if res is None or ocean_mask.float_mask.sel(lon=lon, lat=lat) == 0.:
                if return_estim:
                    estim[ilat, ilon, :] = np.nan
                    var[ilat, ilon, :] = np.nan
                beta[ilat, ilon] = np.nan
                sigmasq[ilat, ilon] = np.nan
                sigmasq2[ilat, ilon] = np.nan
                noise[ilat, ilon] = np.nan
                noise2[ilat, ilon] = np.nan
                if correlated_nugget:
                    rho[ilat, ilon] = np.nan
                    if nugget_prime:
                        noise_prime[ilat, ilon] = np.nan
                        noise2_prime[ilat, ilon] = np.nan
                lengthscale[ilat, ilon, :] = np.nan
                lengthscale2[ilat, ilon, :] = np.nan
                continue

            if return_estim:
                estim[ilat, ilon, :] = res['mu']
                var[ilat, ilon, :] = res['var']
            beta[ilat, ilon] = res['params']['beta']
            sigmasq[ilat, ilon] = res['params']['sigmasq']
            sigmasq2[ilat, ilon] = res['params']['sigmasq2']
            noise[ilat, ilon] = res['params']['noise']
            noise2[ilat, ilon] = res['params']['noise2']
            if correlated_nugget:
                rho[ilat, ilon] = res['params']['rho']
                if nugget_prime:
                    noise_prime[ilat, ilon] = res['params']['noise_prime']
                    noise2_prime[ilat, ilon] = res['params']['noise2_prime']
            lengthscale[ilat, ilon, :] = res['params']['lengthscale']
            lengthscale2[ilat, ilon, :] = res['params']['lengthscale2']

    
    res = {'beta': beta, 'sigmasq': sigmasq, 'sigmasq2': sigmasq2,
            'noise': noise, 'noise2': noise2, 'lengthscale': lengthscale, 'lengthscale2': lengthscale2}
    if correlated_nugget:
        res['rho'] = rho
        if nugget_prime:
            res['noise_prime'] = noise_prime
            res['noise2_prime'] = noise2_prime
    
    if return_estim:
        res['estim'] = estim
        res['var'] = var

    return res
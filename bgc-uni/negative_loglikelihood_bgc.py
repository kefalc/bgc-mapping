import numpy as np
import math
import covariancefunctions_bgc as cov
import distancefunctions_bgc as dist
import solvers_bgc as solve
from njit_options_bgc import opts, not_cache
from numba import njit, prange, uint32

##This is for the univariate case
@njit(**not_cache(), locals=dict(Nblocks=uint32))
def negLogLik_MaternSpaceTime_Nblocks_nu32( params, lat_iw, lon_iw, time_iw, data_iw, dateN, Nblocks ) -> float:
    """ nu = 3/2
    # math.log(2*math.pi) = 1.8378770664093453
    """
    logThetas=params[0]
    logThetax=params[1]
    logThetay=params[2]
    logThetat=params[3]
    logSigma=params[4]
    if not_valid(logThetas,logThetax,logThetay,logThetat,logSigma):
        return 1e10
    
    thetas = math.exp(logThetas)
    thetax = math.exp(logThetax)
    thetay = math.exp(logThetay)
    thetat = math.exp(logThetat)
    sigma_squared = math.exp(logSigma)**2

    val = 0.0
    for n in prange(Nblocks):
        in_time = (time_iw < dateN[n+1]) & (time_iw >= dateN[n])
        data_it = data_iw[in_time]
        inum = data_it.size
        if inum > 0:

            Ktheta = cov.Matern_SpaceTimeCovariance_Nblocks_latlon_nu32(inum, lat_iw[in_time], lon_iw[in_time], time_iw[in_time], thetas, thetax, thetay, thetat, sigma_squared)
            L, INFO = solve.dpotrf(Ktheta, inum)
            if INFO !=0: # check for spd
                return 1e10
            sumlogdiagL = 0.0
            for i in range(inum): # solve for logdet
                sumlogdiagL = sumlogdiagL + math.log(L[i,i])
            
            val = val + 2*sumlogdiagL + data_it @ solve.dpotrs(L, data_it.copy(), inum) + 1.8378770664093453 * inum 
        
    return val*0.5 # Definition of log-likelihood, does not affect optimization.
    

##USE THIS FOR MULTIVARIATE CASE
def negLogLik_MaternSpaceTime_Nblocks_nu32_multi( params, ts_covparam_values, lat_iw, lon_iw, time_iw, data_iw, ts_iw, dateN, Nblocks ) -> float:
    """ nu = 3/2
    # math.log(2*math.pi) = 1.8378770664093453
    """
    ##BGC Params
    logThetasB = params[0]   # Signal variance
    logThetaxB = params[1]   # Decorrelation length scale degrees lon
    logThetayB = params[2]   # Decorrelation length scale degrees lat
    logThetatB = params[3]   # Decorrelation time scale days 
    logSigmaB = params[4]    # Standard deviation of noise
    arctanhBeta = params[5]      # Correlation coefficient of T and B
    if not_valid(logThetasB,logThetaxB,logThetayB,logThetatB,logSigmaB):
        return 1e10
    
    ThetasB = math.exp(logThetasB)
    ThetaxB = math.exp(logThetaxB)
    ThetayB = math.exp(logThetayB)
    ThetatB = math.exp(logThetatB)
    SigmaB = math.exp(logSigmaB)**2
    Beta = math.tanh(arctanhBeta) 

    ##TS Params- already calculated
    ThetasTS = ts_covparam_values[0]
    ThetaxTS = ts_covparam_values[1]
    ThetayTS = ts_covparam_values[2]
    ThetatTS = ts_covparam_values[3]
    SigmaTS = ts_covparam_values[4]**2

    val = 0.0
    for n in prange(Nblocks):
        in_time = (time_iw < dateN[n+1]) & (time_iw >= dateN[n])
        data_it = data_iw[in_time]
        ts_it = ts_iw[in_time]
        inum = data_it.size
        if inum > 0:
            
            Ktheta, y = cov.Matern_nu32_multivar(inum, lat_iw[in_time], lon_iw[in_time], time_iw[in_time], data_it, ts_it, ThetasB, ThetasTS, ThetaxB, ThetaxTS, ThetayB, ThetayTS, ThetatB, ThetatTS, Beta, SigmaTS, SigmaB)
            twonum = 2*inum
            L, INFO = solve.dpotrf(Ktheta, twonum)
            if INFO != 0: # check for spd
                return 1e10
            x_chol = solve.dpotrs(L, y.copy(), twonum)
            sumlogdiagL = 0.0
            yT_invK_y = 0.0
            for i in range(twonum): # Solve for logdet(K)/2, and y^T * K^-1 * y
                yT_invK_y = yT_invK_y + y[i] * x_chol[i]
                sumlogdiagL = sumlogdiagL + math.log(L[i,i])
            val = val + 2*sumlogdiagL + yT_invK_y + 1.8378770664093453 * twonum
        
    return val # Definition of log-likelihood, does not affect optimization.

@njit(**opts())
def not_valid(logThetas, logThetax, logThetay, logThetat, logSigma) -> bool:
    # Machine-eps on workers = 2.220446049250313e-16; 
    logSmallval = -18.0218 # math.log(math.sqrt(2.220446049250313e-16))
    logLargeval = 354.8913 # math.log(math.sqrt(sys.float_info.max))
        
    if logThetas<logSmallval:
        return True
    elif logThetax<logSmallval:
        return True
    elif logThetay<logSmallval:
        return True
    elif logThetat<logSmallval:
        return True
    elif logSigma<logSmallval:
        return True
    
    elif logThetas>logLargeval:
        return True
    elif logThetax>logLargeval:
        return True
    elif logThetay>logLargeval:
        return True
    elif logThetat>logLargeval:
        return True
    elif logSigma>logLargeval:
        return True
    else:
        return False

@njit(**opts())
def not_valid_GenExp(logThetas, logThetax, logThetay, logThetat, logSigma, lognu) -> bool:
    # Machine-eps on workers = 2.220446049250313e-16; 
    logSmallval = -18.0218 # math.log(math.sqrt(2.220446049250313e-16))
    logLargeval = 354.8913 # math.log(math.sqrt(sys.float_info.max))
        
    if logThetas<logSmallval:
        return True
    elif logThetax<logSmallval:
        return True
    elif logThetay<logSmallval:
        return True
    elif logThetat<logSmallval:
        return True
    elif logSigma<logSmallval:
        return True
    elif lognu<logSmallval:
        return True
    
    elif logThetas>logLargeval:
        return True
    elif logThetax>logLargeval:
        return True
    elif logThetay>logLargeval:
        return True
    elif logThetat>logLargeval:
        return True
    elif logSigma>logLargeval:
        return True
    elif lognu>logLargeval:
        return True
    
    else:
        return False
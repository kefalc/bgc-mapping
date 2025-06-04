import numpy as np
import math
import covariancefunctions as cov
import distancefunctions as dist
import solvers as solve
from njit_options import opts, not_cache
from numba import njit, prange, uint32

@njit(**not_cache(), locals=dict(Nblocks=uint32))
def negLogLik_MaternSpaceTime_Nblocks_nu32( params, lat_iw, lon_iw, time_iw, data_iw, dateN, Nblocks ) -> float:
    """ nu = 3/2
    # math.log(2*math.pi) = 1.8378770664093453
    """
    #need to check what these represent. Do I add additional parameters here for oxygen or change one of these to oxygen?
    logThetas=params[0] #signal variance
    logThetax=params[1] #length scale in 2d dimension
    logThetay=params[2] #length scale in 2d dimension
    logThetat=params[3] #time scale
    logSigma=params[4]  #noise
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

            Ktheta = cov.Matern_SpaceTimeCovariance_Nblocks_latlon_nu32(
                inum, lat_iw[in_time], lon_iw[in_time], time_iw[in_time], thetas, thetax, thetay, thetat, sigma_squared 
            )
            L, INFO = solve.dpotrf(Ktheta, inum)
            if INFO > 0: # check for spd
                return 1e10
            sumlogdiagL = 0.0
            for i in range(inum): # solve for logdet
                sumlogdiagL = sumlogdiagL + math.log(L[i,i])
            
            val = val + 2*sumlogdiagL + data_it @ solve.dpotrs(L, data_it.copy(), inum) + 1.8378770664093453 * inum 
        
    return val*0.5 # Definition of log-likelihood, does not affect optimization.


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

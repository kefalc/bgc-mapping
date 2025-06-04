import math
import numpy as np
from numba import njit, prange, uint32
from njit_options import opts, not_cache
from numba.extending import get_cython_function_address
import ctypes
import scipy

@njit( **opts(), locals=dict(n=uint32) )
def Matern_SpaceTimeCovariance_Nblocks_latlon_nu32( n, lat, lon, time, thetas, thetax, thetay, thetat, sigma_squared ) -> np.ndarray:
    """ nu = 3/2 """
    r = np.empty((n,n))
    for i in prange(n):
        lat1 = lat[i]
        lon1 = lon[i]
        time1 = time[i]
        for j in range(n):
            if i == j:
                r[i,j] = thetas + sigma_squared
            else:
                dlon = lon1 - lon[j]
                # https://docs.dask.org/en/stable/delayed-best-practices.html#don-t-mutate-inputs
                if dlon > 180.0:
                    dlon = dlon - 360.0
                if dlon <= -180.0:
                    dlon = dlon + 360.0
                x = (dlon/thetax)**2  +  ((lat1-lat[j])/thetay)**2  + ((time1-time[j])/thetat)**2
                distSpaceTime = math.sqrt(3.0*x)
                r[i,j] = thetas * (1 + distSpaceTime) * math.exp(-distSpaceTime)
    return r


@njit( **opts(), locals=dict(n=uint32) )
def Matern_SpaceTimeCovariance_Nblocks_Ktheta_covPt_scalardata_nu32( 
    n, lat0, lon0, time0, lat, lon, time, thetas, thetax, thetay, thetat, sigma_squared, data
) -> np.ndarray:
    """ 
    nu = 3/2 
    # 180.0/math.pi = 57.29577951308232
    """
    Ktheta = np.empty((n,n))
    data_covPt = np.empty((2,n))
    coslat0 = math.cos(math.radians(lat0))
    for i in prange(n):
        lat1 = lat[i]
        lon1 = lon[i]
        time1 = time[i]
        
        #### INSERT COVPT-CALCULATION HERE
        dlon0 = lon0 - lon1
        # https://docs.dask.org/en/stable/delayed-best-practices.html#don-t-mutate-inputs
        if dlon0 > 180.0:
            dlon0 = dlon0 - 360.0
        if dlon0 <= -180.0:
            dlon0 = dlon0 + 360.0
        x = (dlon0/thetax)**2  +  ((lat0 - lat1)/thetay)**2  + ((time0-time1)/thetat)**2
        distSpaceTime = math.sqrt(3*x)
        
        ## covPt and data in same array for convenience
        data_covPt[0,i] = data[i] 
        data_covPt[1,i] = thetas * (1 + distSpaceTime) * math.exp(-distSpaceTime)
        
        #### END OF COVPT-CALCULATION
        
        for j in range(n):
            if i == j:
                Ktheta[i,j] = thetas + sigma_squared
            else:
                dlon = lon1 - lon[j]
                # https://docs.dask.org/en/stable/delayed-best-practices.html#don-t-mutate-inputs
                if dlon > 180.0:
                    dlon = dlon - 360.0
                if dlon <= -180.0:
                    dlon = dlon + 360.0
                x = (dlon/thetax)**2  +  ((lat1-lat[j])/thetay)**2  + ((time1-time[j])/thetat)**2
                distSpaceTime = math.sqrt(3.0*x)
                Ktheta[i,j] = thetas * (1 + distSpaceTime) * math.exp(-distSpaceTime)
                
    return Ktheta, data_covPt



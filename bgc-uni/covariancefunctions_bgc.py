import math
import numpy as np
from numba import njit, prange, uint32
from njit_options_bgc import opts, not_cache
from numba.extending import get_cython_function_address
import ctypes
import scipy

@njit(**opts(), locals=dict(n=uint32))
def Matern_nu32_multivar(
    n, lat, lon, time, b_data, ts_data, ThetasB, ThetasTS, ThetaxB, ThetaxTS, ThetayB, ThetayTS, ThetatB, ThetatTS, Beta, SigmaTS, SigmaB) -> (np.ndarray, np.ndarray): ##check on inputs
    
    Ktheta = np.zeros((2*n, 2*n))
    y = np.empty((2*n))

     ##for diagonal
    ThetasTS = ThetasTS * ThetaxTS * ThetayTS * ThetatTS * math.sqrt(2/math.pi)
    
    cross_thetas = math.sqrt((ThetasTS * ThetasB) / (1/8*(ThetaxTS**2+ThetaxB**2)*(ThetayTS**2+ThetayB**2)*(ThetatTS**2+ThetatB**2))) 

    ThetasBvar = ThetasB/(ThetaxB*ThetayB*ThetatB)
    ThetasTSvar = ThetasTS/(ThetaxTS*ThetayTS*ThetatTS)

    for i in range(n): 
        lat1 = lat[i]
        lon1 = lon[i]
        time1 = time[i]

        Ktheta[i,i] = ThetasTSvar + SigmaTS
       # Ktheta[i, i+n] = Beta * cross_thetasth
        Ktheta[i+n, i] = Beta * cross_thetas
        Ktheta[i+n, i+n] = ThetasBvar + SigmaB

        ##### Concatenate data here to simplify solving Kx=y
        y[i  ] = ts_data[i] #t residuals
        y[i+n] = b_data[i] #o residuals
        #####



        for j in range(i): # Build lower triangular TT/TB/BT/BB-matrix
               
            dlat = lat1 - lat[j]
            dtime = time1 - time[j]
            dlon = lon1 - lon[j]
                    # https://docs.dask.org/en/stable/delayed-best-practices.html#don-t-mutate-inputs
            if dlon > 180.0:
                dlon = dlon - 360.0
            if dlon <= -180.0:
                dlon = dlon + 360.0
    
            ##for upper TT
            xTS = (dlon/ThetaxTS)**2  +  (dlat/ThetayTS)**2  + (dtime/ThetatTS)**2 
            distSpaceTimeTS = math.sqrt(3.0*xTS)
            ##for lower BB
            xB = (dlon/ThetaxB)**2  +  (dlat/ThetayB)**2  + (dtime/ThetatB)**2
            distSpaceTimeB = math.sqrt(3.0*xB)
           
            xTB = (2*((dlon**2/(ThetaxTS**2 + ThetaxB**2))+(dlat**2/(ThetayB**2 + ThetayTS**2))+(dtime**2/(ThetatB**2+ThetatTS**2))))
            distSpaceTimeTB = math.sqrt(3.0*xTB)
                
    
    
            Ktheta[i  ,j  ] = ThetasTSvar * (1 + distSpaceTimeTS) * math.exp(-distSpaceTimeTS) # TT
            Ktheta[i+n,j  ] = Beta * cross_thetas * (1 + distSpaceTimeTB) * math.exp(-distSpaceTimeTB) ##lower TO
            Ktheta[j+n,i  ] = Beta * cross_thetas * (1 + distSpaceTimeTB) * math.exp(-distSpaceTimeTB) # upper TO
            Ktheta[i+n,j+n] = ThetasBvar * (1 + distSpaceTimeB) * math.exp(-distSpaceTimeB) ##lower right OO

    return Ktheta, y



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
                x = (dlon/thetax)**2  +  ((lat1-lat[j])/thetay)**2  + ((time1-time[j])/thetat)**2 ##This is ACTUALLY x^2 (or z^2 from my notes)
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
    Ktheta = np.empty((n,n)) ##this will be 2n+p, blocks for temp and bgc
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

@njit( **opts(), locals=dict(n=uint32) )

def Matern_SpaceTimeCovariance_Nblocks_Ktheta_covPt_scalardata_nu32_multi(nbgc, nts, lat0, lon0, time0, bgclat, bgclon, bgctime, bgcanom, tsanom_bgc, tslat, tslon, tstime, tsanom, thetasTS, thetaxTS, thetayTS, thetatTS, sigmaTS_squared, thetasB, thetaxB, thetayB, thetatB, sigmaB_squared, beta) -> np.ndarray:
    """ 

    nu = 3/2 

    # 180.0/math.pi = 57.29577951308232

    """
    Ktheta = np.zeros((2*nbgc+nts,2*nbgc+nts))
    data_covPt = np.zeros((2,2*nbgc+nts))

    ##for diagonal
    thetasTS = thetasTS * thetaxTS * thetayTS * thetatTS * math.sqrt(2/math.pi)
    cross_thetas = math.sqrt((thetasTS * thetasB) / (1/8*(thetaxTS**2+thetaxB**2)*(thetayTS**2+thetayB**2)*(thetatTS**2+thetatB**2))) 
    thetasBvar = thetasB/(thetaxB*thetayB*thetatB)
    thetasTSvar = thetasTS/(thetaxTS*thetayTS*thetatTS)

    # first - bgc locations -- TT, TB, BB
    for i in prange(nbgc):
        lat1 = bgclat[i]
        lon1 = bgclon[i]
        time1 = bgctime[i]

        #### INSERT COVPT-CALCULATION HERE
        dlon0 = lon0 - lon1
        dlat0 = lat0 - lat1
        dtime0 = time0 - time1

        if dlon0 > 180.0:
            dlon0 = dlon0 - 360.0
        if dlon0 <= -180.0:
            dlon0 = dlon0 + 360.0

        xB = (dlon0/thetaxB)**2  +  (dlat0/thetayB)**2  + (dtime0/thetatB)**2
        distSpaceTimeB = math.sqrt(3.0*xB)
        xTB = (2*((dlon0**2/(thetaxTS**2 + thetaxB**2))+(dlat0**2/(thetayB**2 + thetayTS**2))+(dtime0**2/(thetatB**2+thetatTS**2))))
        distSpaceTimeTB = math.sqrt(3.0*xTB)   

        ## covPt and data in same array for convenience
        data_covPt[0,i] = tsanom_bgc[i] 
        data_covPt[1,i] = beta * cross_thetas * (1 + distSpaceTimeTB) * math.exp(-distSpaceTimeTB)
        data_covPt[0,i+nbgc] = bgcanom[i]
        data_covPt[1,i+nbgc] = thetasBvar * (1 + distSpaceTimeB) * math.exp(-distSpaceTimeB)

        #### END OF COVPT-CALCULATION
        # Diagonal
        Ktheta[i,      i] = thetasTSvar + sigmaTS_squared
        Ktheta[i+nbgc, i] = beta * cross_thetas
        Ktheta[i+nbgc, i+nbgc] = thetasBvar + sigmaB_squared

        for j in range(i):
            dlon = lon1 - bgclon[j]
            dlat = lat1 - bgclat[j]
            dtime = time1 - bgctime[j]

            if dlon > 180.0:
                dlon = dlon - 360.0
            if dlon <= -180.0:
                dlon = dlon + 360.0
                
            ##for upper TT
            xTS = (dlon/thetaxTS)**2  +  (dlat/thetayTS)**2  + (dtime/thetatTS)**2 
            distSpaceTimeTS = math.sqrt(3.0*xTS)

            ##for lower BB
            xB = (dlon/thetaxB)**2  +  (dlat/thetayB)**2  + (dtime/thetatB)**2
            distSpaceTimeB = math.sqrt(3.0*xB)

            ##for cross-covariance TB
            xTB = (2*((dlon**2/(thetaxTS**2 + thetaxB**2))+(dlat**2/(thetayB**2 + thetayTS**2))+(dtime**2/(thetatB**2+thetatTS**2))))
            distSpaceTimeTB = math.sqrt(3.0*xTB)
            
            #Ktheta[i,j] = thetas * (1 + distSpaceTime) * math.exp(-distSpaceTime)
            Ktheta[i  ,   j  ] = thetasTSvar * (1 + distSpaceTimeTS) * math.exp(-distSpaceTimeTS) # TT
            Ktheta[i+nbgc,j  ] = beta * cross_thetas * (1 + distSpaceTimeTB) * math.exp(-distSpaceTimeTB) ##lower TO
            Ktheta[j+nbgc,i  ] = beta * cross_thetas * (1 + distSpaceTimeTB) * math.exp(-distSpaceTimeTB) # upper TO
            Ktheta[i+nbgc,j+nbgc] = thetasBvar * (1 + distSpaceTimeB) * math.exp(-distSpaceTimeB) ##lower right OO

    # second ts-only location
    for i in prange(nts):
        lat1 = tslat[i]
        lon1 = tslon[i]
        time1 = tstime[i]

        #### INSERT COVPT-CALCULATION HERE
        dlon0 = lon0 - lon1
        dlat0 = lat0 - lat1
        dtime0 = time0 - time1

        if dlon0 > 180.0:
            dlon0 = dlon0 - 360.0
        if dlon0 <= -180.0:
            dlon0 = dlon0 + 360.0

        xTB = (2*((dlon0**2/(thetaxTS**2 + thetaxB**2))+(dlat0**2/(thetayB**2 + thetayTS**2))+(dtime0**2/(thetatB**2+thetatTS**2))))
        distSpaceTimeTB = math.sqrt(3.0*xTB)    

        ## covPt and data in same array for convenience
        data_covPt[0,2*nbgc + i] = tsanom[i] 
        data_covPt[1,2*nbgc + i] = beta * cross_thetas * (1 + distSpaceTimeTB) * math.exp(-distSpaceTimeTB) 

        #### END OF COVPT-CALCULATION
        # Diagonal
        Ktheta[2*nbgc+i, 2*nbgc+i] = thetasTSvar + sigmaTS_squared

        for j in range(nbgc):
            dlon = lon1 - bgclon[j]
            dlat = lat1 - bgclat[j]
            dtime = time1 - bgctime[j]
            if dlon > 180.0:
                dlon = dlon - 360.0
            if dlon <= -180.0:
                dlon = dlon + 360.0

            ##for upper TT
            xTS = (dlon/thetaxTS)**2  +  (dlat/thetayTS)**2  + (dtime/thetatTS)**2 
            distSpaceTimeTS = math.sqrt(3.0*xTS)

            ##for cross-covariance TB
            xTB = (2*((dlon**2/(thetaxTS**2 + thetaxB**2))+(dlat**2/(thetayB**2 + thetayTS**2))+(dtime**2/(thetatB**2+thetatTS**2))))
            distSpaceTimeTB = math.sqrt(3.0*xTB)

            Ktheta[2*nbgc+i, j  ] = thetasTSvar * (1 + distSpaceTimeTS) * math.exp(-distSpaceTimeTS) # TT
            Ktheta[2*nbgc+i, j+nbgc] = beta * cross_thetas * (1 + distSpaceTimeTB) * math.exp(-distSpaceTimeTB) ##lower TO
        
        for j in range(i):
            dlon = lon1 - tslon[j]
            dlat = lat1 - tslat[j]
            dtime = time1 - tstime[j]
            if dlon > 180.0:
                dlon = dlon - 360.0
            if dlon <= -180.0:
                dlon = dlon + 360.0
                
            ##for upper TT
            xTS = (dlon/thetaxTS)**2  +  (dlat/thetayTS)**2  + (dtime/thetatTS)**2 
            distSpaceTimeTS = math.sqrt(3.0*xTS)

            Ktheta[2*nbgc+i, 2*nbgc+j ] = thetasTSvar * (1 + distSpaceTimeTS) * math.exp(-distSpaceTimeTS) # TT   

    return Ktheta, data_covPt

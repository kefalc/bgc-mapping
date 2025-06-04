from numba import njit, uint32
from njit_options import opts

"""
Slightly faster than a @ b, on remote worker with 1 core.
"""


@njit(**opts(), locals=dict(n=uint32))
def covPt_fitcoef_njit_1d(covPt, fitcoef,n) -> float:
    """ covPt @ fitcoef """
    C = 0.0
    for i in range(n):
        C = C + covPt[i]*fitcoef[i]
    return C

@njit(**opts(), locals=dict(n=uint32))
def covPt_scalardata(n, covPt, sol) -> (float, float):
    """ covPt.shape, sol.shape = (n,), (2,n,) """
    Cd = 0.0
    Ced = 0.0
    for i in range(n):
        covPt_i = covPt[i]
        Cd = Cd + covPt_i*sol[0,i]
        Ced = Ced + covPt_i*sol[1,i]
    return Cd, Ced
""" Set njit-compilation options """

def opts():
    options = dict(
        parallel = False,
        fastmath = True,
        cache = True,
        nogil = True,
        error_model = "numpy"
    )
    return options


def opts_not_fm():
    """ 
    For when compiling np.nanmean and similar functions, since "fastmath" assumes no NaNs.
    Ref: https://github.com/numba/numba/issues/7701
    """
    options = dict(
        parallel = False,
        fastmath = False,
        cache = True,
        nogil = True,
        error_model = "numpy"
    )
    return options
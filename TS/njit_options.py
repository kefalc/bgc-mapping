""" Set njit-compilation options """

def opts():
    return dict(
        parallel = False,
        fastmath = True,
        cache = True,
        nogil = True,
        error_model = "numpy"
    )


def parallel():
    return dict(
        parallel=True,
        fastmath=True,
        cache=True,
        nogil=True,
        error_model = "numpy"
    )


def not_cache():
    """ numba scipy """
    return dict(
        parallel = False,
        fastmath = True,
        cache = False,
        nogil = True,
        error_model = "numpy"
    )
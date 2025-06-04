import math, numpy as np, ctypes
from numba import njit
from numba.extending import get_cython_function_address
from njit_options_bgc import opts, not_cache


@njit(**opts())
def least_squares(Mn, data) -> np.ndarray:
    return np.linalg.lstsq(Mn, data)[0]


### MUTABLE GLOBAL VARIABLES:

_ptr_dbl = ctypes.POINTER(ctypes.c_double)
_ptr_real = ctypes.POINTER(ctypes.c_float)
_ptr_int = ctypes.POINTER(ctypes.c_int)
_ptr_char = ctypes.POINTER(ctypes.c_int)

# Get addresses of functions
dpotrf_addr = get_cython_function_address('scipy.linalg.cython_lapack', 'dpotrf')
dpotrs_addr = get_cython_function_address('scipy.linalg.cython_lapack', 'dpotrs')
spotrf_addr = get_cython_function_address('scipy.linalg.cython_lapack', 'spotrf')
spotrs_addr = get_cython_function_address('scipy.linalg.cython_lapack', 'spotrs')


###DPOTR
## Define function types
# void dpotrf(char *uplo, int *n, d *a, int *lda, int *info)
dpotrf_functype = ctypes.CFUNCTYPE(
    None,       # void
    _ptr_char,  # UPLO
    _ptr_int,   # N
    _ptr_dbl,   # A
    _ptr_int,   # LDA
    _ptr_int,   # INFO
)
# void dpotrs(char *uplo, int *n, int *nrhs, d *a, int *lda, d *b, int *ldb, int *info)
dpotrs_functype = ctypes.CFUNCTYPE(
    None,       # void
    _ptr_char,  # UPLO
    _ptr_int,   # N
    _ptr_int,   # NRHS
    _ptr_dbl,   # A
    _ptr_int,   # LDA
    _ptr_dbl,   # B
    _ptr_int,   # LDB
    _ptr_int,   # INFO
)

####SPOTR
## Define function types
# void spotrf(char *uplo, int *n, real *a, int *lda, int *info)
spotrf_functype = ctypes.CFUNCTYPE(
    None,       # void
    _ptr_char,  # UPLO
    _ptr_int,   # N
    _ptr_real,   # A
    _ptr_int,   # LDA
    _ptr_int,   # INFO
)
# void spotrs(char *uplo, int *n, int *nrhs, real *a, int *lda, real *b, int *ldb, int *info)
spotrs_functype = ctypes.CFUNCTYPE(
    None,       # void
    _ptr_char,  # UPLO
    _ptr_int,   # N
    _ptr_int,   # NRHS
    _ptr_real,   # A
    _ptr_int,   # LDA
    _ptr_real,   # B
    _ptr_int,   # LDB
    _ptr_int,   # INFO
)

# Create function objects
dpotrf_fn = dpotrf_functype(dpotrf_addr)
dpotrs_fn = dpotrs_functype(dpotrs_addr)
spotrf_fn = spotrf_functype(spotrf_addr)
spotrs_fn = spotrs_functype(spotrs_addr)


@njit(**not_cache())
def spotrf(A, size, lower=True):
    """
    URL:https://netlib.org/lapack/explore-html/d2/d09/group__potrf_gafc806a229db445c723f8e1abc2ef4370.html
    SPOTRF computes the Cholesky factorization (U or L) of a real symmetric positive definite matrix A.
    """
    # Data type must be integer representing Unicode
    UPLO = np.array(ord('L') if lower else ord('U'), np.int32)
    N = np.array(size, dtype=np.int32)
    LDA = np.array(size, dtype=np.int32)
    INFO = np.array(0, dtype=np.int32)
    spotrf_fn(
        UPLO.ctypes,
        N.ctypes,
        A.ctypes,       # out
        LDA.ctypes,
        INFO.ctypes,    # out
    )

    if INFO < 0:
        raise Exception('INFO < 0. The i-th argument had an illegal value')
        
    return A, INFO


@njit(**not_cache())
def spotrs(L, b, size, lower=True):
    """
    URL:https://netlib.org/lapack/explore-html/d3/dc8/group__potrs_gaa14667c816c78f9d1dcae9017e473746.html
    SPOTRS solves a system of linear equations A*X = B with a symmetric
    positive definite matrix A, using the Cholesky factorization (U or L) computed by SPOTRF.
    
    void spotrs(char *uplo, int *n, int *nrhs, real *a, int *lda, real *b, int *ldb, int *info)
    lapack_int LAPACKE_spotrs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, const real* a, lapack_int lda,
                           real* b, lapack_int ldb )
    """
    UPLO = np.array(ord('L') if lower else ord('U'), np.int32)
    INFO = np.array(0, dtype=np.int32)
    N = np.array(size, dtype=np.int32)
    LDA = np.array(size, dtype=np.int32)
    NRHS = np.array(1, dtype=np.int32)
    LDB = np.array(size, dtype=np.int32)

    spotrs_fn(
        UPLO.ctypes,
        N.ctypes,
        NRHS.ctypes,
        L.ctypes,
        LDA.ctypes,
        b.ctypes,    # out
        LDB.ctypes,
        INFO.ctypes,    # out
    )
    
    if INFO:
        raise Exception('INFO < 0. The i-th argument had an illegal value')
        
    return b


@njit(**not_cache())
def dpotrf(A, size, lower=True):
    """
    URL:https://netlib.org/lapack/explore-html/d2/d09/group__potrf_ga84e90859b02139934b166e579dd211d4.html
    DPOTRF computes the Cholesky factorization (U or L) of a real symmetric positive definite matrix A.
    
    void dpotrf(char *uplo, int *n, d *a, int *lda, int *info)
    lapack_int LAPACKE_dpotrf( int matrix_layout, char uplo, lapack_int n, double* a,
                               lapack_int lda )
    """
    # Data type must be integer representing Unicode
    UPLO = np.array(ord('L') if lower else ord('U'), np.int32)
    N = np.array(size, dtype=np.int32)
    LDA = np.array(size, dtype=np.int32)
    INFO = np.array(0, dtype=np.int32)
    dpotrf_fn(
        UPLO.ctypes,
        N.ctypes,
        A.ctypes,       # out
        LDA.ctypes,
        INFO.ctypes,    # out
    )
    
    if INFO < 0:
        raise Exception('INFO < 0. The i-th argument had an illegal value')
        
    return A, INFO


@njit(**not_cache())
def dpotrs(L, b, size, lower=True):
    """
    URL:https://netlib.org/lapack/explore-html/d3/dc8/group__potrs_ga70a04d13ff2123745a26b1e236212cf7.html
    DPOTRS solves a system of linear equations A*X = B with a symmetric
    positive definite matrix A, using the Cholesky factorization (U or L) computed by DPOTRF.
    
    b.shape = (size,)
    
    void dpotrs(char *uplo, int *n, int *nrhs, d *a, int *lda, d *b, int *ldb, int *info)
    lapack_int LAPACKE_dpotrs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, const double* a, lapack_int lda,
                           double* b, lapack_int ldb )
    """
    UPLO = np.array(ord('L') if lower else ord('U'), np.int32)
    INFO = np.array(0, dtype=np.int32)
    N = np.array(size, dtype=np.int32)
    LDA = np.array(size, dtype=np.int32)
    NRHS = np.array(1, dtype=np.int32)
    LDB = np.array(size, dtype=np.int32)

    dpotrs_fn(
        UPLO.ctypes,
        N.ctypes,
        NRHS.ctypes,
        L.ctypes,
        LDA.ctypes,
        b.ctypes,    # out
        LDB.ctypes,
        INFO.ctypes,    # out
    )
    
    if INFO:
        raise Exception('INFO < 0. The i-th argument had an illegal value')
        
    return b

@njit(**not_cache())
def dpotrs_multi(L, b, nb, n, lower=True):
    """
    URL:https://netlib.org/lapack/explore-html/d3/dc8/group__potrs_ga70a04d13ff2123745a26b1e236212cf7.html
    DPOTRS solves a system of linear equations A*x=b with a symmetric
    positive definite matrix A, using the Cholesky factorization (U or L) computed by DPOTRF.
    
    b.shape = (nb, n,)
    
    void dpotrs(char *uplo, int *n, int *nrhs, d *a, int *lda, d *b, int *ldb, int *info)
    lapack_int LAPACKE_dpotrs( int matrix_layout, char uplo, lapack_int n,
                           lapack_int nrhs, const double* a, lapack_int lda,
                           double* b, lapack_int ldb )
    """
    UPLO = np.array(ord('U') if lower else ord('L'), np.int32)
    INFO = np.array(0, dtype=np.int32)
    N = np.array(n, dtype=np.int32)
    LDA = np.array(n, dtype=np.int32)
    NRHS = np.array(nb, dtype=np.int32)
    LDB = np.array(n, dtype=np.int32)
    dpotrs_fn(
        UPLO.ctypes,
        N.ctypes,
        NRHS.ctypes,
        L.ctypes,
        LDA.ctypes,
        b.ctypes,    # out
        LDB.ctypes,
        INFO.ctypes,    # out
    )
        
    return b, INFO


@njit(**not_cache())
def dpotrs_b2(L, b2, size, lower=True):
    """
    b2.shape = (2, size,)
    """
    UPLO = np.array(ord('L') if lower else ord('U'), np.int32)
    INFO = np.array(0, dtype=np.int32)
    N = np.array(size, dtype=np.int32)
    LDA = np.array(size, dtype=np.int32)
    NRHS = np.array(2, dtype=np.int32)
    LDB = np.array(size, dtype=np.int32)

    dpotrs_fn(
        UPLO.ctypes,
        N.ctypes,
        NRHS.ctypes,
        L.ctypes,
        LDA.ctypes,
        b2.ctypes,    # out
        LDB.ctypes,
        INFO.ctypes,    # out
    )
    
    if INFO:
        raise Exception('INFO < 0. The i-th argument had an illegal value')
        
    return b2


@njit(**not_cache())
def dpotrs_b3(L, b3, size, lower=True):
    """
    b3.shape = (3, size,)
    """
    UPLO = np.array(ord('L') if lower else ord('U'), np.int32)
    INFO = np.array(0, dtype=np.int32)
    N = np.array(size, dtype=np.int32)
    LDA = np.array(size, dtype=np.int32)
    NRHS = np.array(3, dtype=np.int32)
    LDB = np.array(size, dtype=np.int32)

    dpotrs_fn(
        UPLO.ctypes,
        N.ctypes,
        NRHS.ctypes,
        L.ctypes,
        LDA.ctypes,
        b3.ctypes,    # out
        LDB.ctypes,
        INFO.ctypes,    # out
    )
    
    if INFO:
        raise Exception('INFO < 0. The i-th argument had an illegal value')
        
    return b3
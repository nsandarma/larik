import ctypes,ctypes.util,os,numpy as np

# Konstanta CBLAS
CblasRowMajor     = 101
CblasColMajor     = 102
CblasNoTrans      = 111
CblasTrans        = 112
CblasConjTrans    = 113
CblasConjNoTrans  = 114
CblasUpper        = 121
CblasLower        = 122
CblasNonUnit      = 131
CblasUnit         = 132
CblasLeft         = 141
CblasRight        = 142

# lib = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Accelerate.framework/Versions/Current/Accelerate")
# class TEST(ctypes.Structure):
#   _fields_ = [
#     ("x",ctypes.c_int),
#     ("y",ctypes.c_bool)
#   ]
#
# class Number(ctypes.Union):
#     _fields_ = [
#         ("i", ctypes.c_int),
#         ("f", ctypes.c_float),
#     ]

path = "./blas/OpenBLAS-0.3.29/libopenblas.dylib"
# path_np = "/opt/homebrew/lib/python3.11/site-packages/numpy/.dylibs/libopenblas64_.0.dylib"
lib = {}
lib["blas"] = ctypes.cdll.LoadLibrary(path)
# lib["blas"] = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Accelerate.framework/Versions/Current/Accelerate")


try:
  cblas_sdot = lib["blas"].cblas_sdot
  cblas_sdot.restype = ctypes.c_float
  cblas_sdot.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_int
  ]
except AttributeError:pass

try:
  cblas_ddot = lib["blas"].cblas_ddot
  cblas_ddot.restype = ctypes.c_double
  cblas_ddot.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int
  ]
except AttributeError:pass

try:
  cblas_sasum = lib["blas"].cblas_sasum
  cblas_sasum.restype = ctypes.c_float
  cblas_sasum.argtypes = [
    ctypes.c_int,ctypes.POINTER(ctypes.c_float),
    ctypes.c_int
  ]
except AttributeError:pass

try:
  cblas_dasum = lib["blas"].cblas_dasum
  cblas_dasum.restype = ctypes.c_double
  cblas_dasum.argtypes = [
    ctypes.c_int,ctypes.POINTER(ctypes.c_double),
    ctypes.c_int
  ]
except AttributeError:pass

try:
  cblas_ssum = lib["blas"].cblas_ssum
  cblas_ssum.restype = ctypes.c_float
  cblas_ssum.argtypes = [
    ctypes.c_int,ctypes.POINTER(ctypes.c_float),
    ctypes.c_int
  ]
except AttributeError:pass

try:
  cblas_dsum = lib["blas"].cblas_dsum
  cblas_dsum.restype = ctypes.c_double
  cblas_dsum.argtypes = [
    ctypes.c_int,ctypes.POINTER(ctypes.c_double),
    ctypes.c_int
  ]
except AttributeError:pass


try:
  cblas_isamax = lib["blas"].cblas_isamax
  cblas_isamax.restype = ctypes.c_int
  cblas_isamax.argtypes = [
    ctypes.c_int,ctypes.POINTER(ctypes.c_float),ctypes.c_int
  ]
except AttributeError:pass

try:
  cblas_idamax = lib["blas"].cblas_idamax
  cblas_idamax.restype = ctypes.c_int
  cblas_idamax.argtypes = [
    ctypes.c_int,ctypes.POINTER(ctypes.c_double),ctypes.c_int
  ]
except AttributeError:pass

try:
  cblas_isamin = lib["blas"].cblas_isamin
  cblas_isamin.restype = ctypes.c_int
  cblas_isamin.argtypes = [
    ctypes.c_int,ctypes.POINTER(ctypes.c_float),ctypes.c_int
  ]
except AttributeError:pass

try:
  cblas_idamin = lib["blas"].cblas_idamin
  cblas_idamin.restype = ctypes.c_int
  cblas_idamin.argtypes = [
    ctypes.c_int,ctypes.POINTER(ctypes.c_double),ctypes.c_int
  ]
except AttributeError:pass

try:
  cblas_samax = lib["blas"].cblas_samax
  cblas_samax.restype = ctypes.c_float
  cblas_samax.argtypes = [
    ctypes.c_int,ctypes.POINTER(ctypes.c_float),ctypes.c_int
  ]
except AttributeError:pass

try:
  cblas_damax = lib["blas"].cblas_damax
  cblas_damax.restype = ctypes.c_double
  cblas_damax.argtypes = [
    ctypes.c_int,ctypes.POINTER(ctypes.c_double),ctypes.c_int
  ]
except AttributeError: pass

try:
  cblas_samin = lib["blas"].cblas_samin
  cblas_samin.restype = ctypes.c_float
  cblas_samin.argstypes = [ctypes.c_int,ctypes.POINTER(ctypes.c_double),ctypes.c_int]
except AttributeError:pass

try:
  cblas_damin = lib["blas"].cblas_damin
  cblas_damin.restype = ctypes.c_float
  cblas_damin.argstypes = [ctypes.c_int,ctypes.POINTER(ctypes.c_double),ctypes.c_int]
except AttributeError:pass

try:
  cblas_ismax = lib["blas"].cblas_ismax
  cblas_ismax.restype = ctypes.c_size_t
  cblas_ismax.argtypes = [ctypes.c_int,ctypes.POINTER(ctypes.c_float),ctypes.c_int]
except AttributeError:pass

try:
  cblas_idmax = lib["blas"].cblas_idmax
  cblas_idmax.restype = ctypes.c_size_t
  cblas_idmax.argtypes = [ctypes.c_int,ctypes.POINTER(ctypes.c_double),ctypes.c_int]
except AttributeError:pass

try:
  cblas_ismin = lib["blas"].cblas_ismin
  cblas_ismin.restype = ctypes.c_size_t
  cblas_ismin.argtypes = [ctypes.c_int,ctypes.POINTER(ctypes.c_float),ctypes.c_int]
except AttributeError:pass

try:
  cblas_idmin = lib["blas"].cblas_idmin
  cblas_idmin.restype = ctypes.c_size_t
  cblas_idmin.argtypes = [ctypes.c_int,ctypes.POINTER(ctypes.c_double),ctypes.c_int]
except AttributeError:pass

try:
  cblas_saxpy = lib["blas"].cblas_saxpy
  cblas_saxpy.restype = None
  cblas_saxpy.argtypes = [
    ctypes.c_int, # n
    ctypes.c_float, # alpha
    ctypes.POINTER(ctypes.c_float), #x
    ctypes.c_int, #incx
    ctypes.POINTER(ctypes.c_float), #y
    ctypes.c_int #incy
  ]
except AttributeError:pass

try:
  cblas_daxpy = lib["blas"].cblas_daxpy
  cblas_daxpy.restype = None
  cblas_daxpy.argtypes = [
    ctypes.c_int, # n
    ctypes.c_double, # alpha
    ctypes.POINTER(ctypes.c_double), #x
    ctypes.c_int, #incx
    ctypes.POINTER(ctypes.c_double), #y
    ctypes.c_int #incy
  ]
except AttributeError:pass

try:
  cblas_scopy = lib["blas"].cblas_scopy
  cblas_scopy.restype = None
  cblas_scopy.argtypes = [
    ctypes.c_int, # n
    ctypes.POINTER(ctypes.c_float), #x
    ctypes.c_int, #incx
    ctypes.POINTER(ctypes.c_float), #y
    ctypes.c_int # incy
  ]
except AttributeError:pass

try:
  cblas_dcopy = lib["blas"].cblas_dcopy
  cblas_dcopy.restype = None
  cblas_dcopy.argtypes = [
    ctypes.c_int, # n
    ctypes.POINTER(ctypes.c_double), #x
    ctypes.c_int, #incx
    ctypes.POINTER(ctypes.c_double), #y
    ctypes.c_int # incy
  ]
except AttributeError:pass

try:
  cblas_sswap = lib["blas"].cblas_sswap
  cblas_sswap.restype = None
  cblas_sswap.argtypes = [
    ctypes.c_int, # n
    ctypes.POINTER(ctypes.c_float), #x
    ctypes.c_int, #incx
    ctypes.POINTER(ctypes.c_float), #y
    ctypes.c_int # incy
  ]
except AttributeError:pass

try:
  cblas_dswap = lib["blas"].cblas_dswap
  cblas_dswap.restype = None
  cblas_dswap.argtypes = [
    ctypes.c_int, # n
    ctypes.POINTER(ctypes.c_double), #x
    ctypes.c_int, #incx
    ctypes.POINTER(ctypes.c_double), #y
    ctypes.c_int # incy
  ]
except AttributeError:pass

try:
  cblas_dswap = lib["blas"].cblas_sswap
  cblas_dswap.restype = None
  cblas_dswap.argtypes = [
    ctypes.c_int, # n
    ctypes.POINTER(ctypes.c_double), #x
    ctypes.c_int, #incx
    ctypes.POINTER(ctypes.c_double), #y
    ctypes.c_int # incy
  ]
except AttributeError:pass

try:
  cblas_srot = lib["blas"].cblas_srot
  cblas_srot.restype = None
  cblas_srot.argtypes = [
    ctypes.c_int, # n
    ctypes.POINTER(ctypes.c_float), #x
    ctypes.c_int, #incx
    ctypes.POINTER(ctypes.c_float), #y
    ctypes.c_int, # incy
    ctypes.c_float, #c
    ctypes.c_float  #s
  ]
except AttributeError:pass

try:
  cblas_drot = lib["blas"].cblas_drot
  cblas_drot.restype = None
  cblas_drot.argtypes = [
    ctypes.c_int, # n
    ctypes.POINTER(ctypes.c_double), #x
    ctypes.c_int, #incx
    ctypes.POINTER(ctypes.c_double), #y
    ctypes.c_int, # incy
    ctypes.c_double, #c
    ctypes.c_double  #s
  ]
except AttributeError:pass

try:
  cblas_srotm = lib["blas"].cblas_srotm
  cblas_srotm.restype = None
  cblas_srotm.argtypes = [
    ctypes.c_int, # n
    ctypes.POINTER(ctypes.c_float), #x
    ctypes.c_int, #incx
    ctypes.POINTER(ctypes.c_float), #y
    ctypes.c_int, # incy
    ctypes.POINTER(ctypes.c_float), #p
  ]
except AttributeError:pass

try:
  cblas_drotm = lib["blas"].cblas_drotm
  cblas_drotm.restype = None
  cblas_drotm.argtypes = [
    ctypes.c_int, # n
    ctypes.POINTER(ctypes.c_double), #x
    ctypes.c_int, #incx
    ctypes.POINTER(ctypes.c_double), #y
    ctypes.c_int, # incy
    ctypes.POINTER(ctypes.c_double), #p
  ]
except AttributeError:pass

try:
  cblas_sscal = lib["blas"].cblas_sscal
  cblas_sscal.restype = None
  cblas_sscal.argtypes = [
    ctypes.c_int, # n
    ctypes.c_float, # alpha
    ctypes.POINTER(ctypes.c_float), #X
    ctypes.c_int, #incx
  ]
except AttributeError:pass

try:
  cblas_dscal = lib["blas"].cblas_dscal
  cblas_dscal.restype = None
  cblas_dscal.argtypes = [
    ctypes.c_int, # n
    ctypes.c_float, # alpha
    ctypes.POINTER(ctypes.c_double), #X
    ctypes.c_int, #incx
  ]
except AttributeError:pass

try:
  cblas_sgemv = lib["blas"].cblas_sgemv
  cblas_sgemv.restype = None
  cblas_sgemv.argtypes = [
    ctypes.c_int, # Order Enum
    ctypes.c_int, # Trans Enum
    ctypes.c_int, # m
    ctypes.c_int,  # n
    ctypes.c_float, # alpha
    ctypes.POINTER(ctypes.c_float), # a
    ctypes.c_int, #lda
    ctypes.c_int, #incx
    ctypes.c_float, #beta
    ctypes.POINTER(ctypes.c_float), # y
    ctypes.c_int #incy
  ]
except AttributeError:pass

try:
  cblas_dgemv = lib["blas"].cblas_dgemv
  cblas_dgemv.restype = None
  cblas_dgemv.argtypes = [
    ctypes.c_int, # Order Enum
    ctypes.c_int, # Trans Enum
    ctypes.c_int, # m
    ctypes.c_int,  # n
    ctypes.c_double, # alpha
    ctypes.POINTER(ctypes.c_float), # a
    ctypes.c_int, #lda
    ctypes.c_int, #incx
    ctypes.c_double, #beta
    ctypes.POINTER(ctypes.c_double), # y
    ctypes.c_int #incy
  ]
except AttributeError:pass

# void cblas_sgemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
# 		 OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float beta, float *C, OPENBLAS_CONST blasint ldc);
try:
  cblas_sgemm = lib["blas"].cblas_sgemm
  cblas_sgemm.restype = None
  cblas_sgemm.argtypes = [
    ctypes.c_int, # Layout Order
    ctypes.c_int, # TransA
    ctypes.c_int, # Trans B
    ctypes.c_int, ctypes.c_int, ctypes.c_int,   # M, N, K
    ctypes.c_float,                             # alpha
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,  # A, lda
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,  # B, ldb
    ctypes.c_float,                               # beta
    ctypes.POINTER(ctypes.c_float), ctypes.c_int   # C, ldc
  ]
except AttributeError:pass

try:
  cblas_dgemm = lib["blas"].cblas_dgemm
  cblas_dgemm.restype = None
  cblas_dgemm.argtypes = [
    ctypes.c_int, # Layout Order
    ctypes.c_int, # TransA
    ctypes.c_int, # Trans B
    ctypes.c_int, ctypes.c_int, ctypes.c_int,   # M, N, K
    ctypes.c_double,                             # alpha
    ctypes.POINTER(ctypes.c_double), ctypes.c_int,  # A, lda
    ctypes.POINTER(ctypes.c_double), ctypes.c_int,  # B, ldb
    ctypes.c_double,                               # beta
    ctypes.POINTER(ctypes.c_double), ctypes.c_int   # C, ldc
  ]
except AttributeError:pass




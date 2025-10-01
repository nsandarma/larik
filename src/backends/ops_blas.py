import ctypes,math
from src._buffer import Buffer
import src.backends.autogen.blas as libblas
from src.dtype import Dtype

def argmax(a) -> int:
  buff = a.buffer()
  n = a.size
  if (a.dtype == Dtype.float32):
    return libblas.cblas_ismax(n,buff.as_ctypes_(),1)
  elif (a.dtype == Dtype.float64):
    return libblas.cblas_idmax(n,buff.as_ctypes_(),1)
  else: raise NotImplementedError

def argmin(a) -> int:
  buff = a.buffer()
  n = a.size
  if (a.dtype == Dtype.float32):
    return libblas.cblas_ismin(n,buff.as_ctypes_(),1)
  elif (a.dtype == Dtype.float64):
    return libblas.cblas_idmin(n,buff.as_ctypes_(),1)
  else: raise NotImplementedError

def add(a,b):
  na = a.size
  a = a.buffer()
  b = b.buffer()
  libblas.cblas_daxpy(na,
                      1.0,
                      a.as_ctypes_(),1,
                      b.as_ctypes_(),1)
  

if __name__ == "__main__":
  import numpy as np
  from src.larik import Larik
  n = 10
  a = np.random.rand(n,n)
  b = np.random.rand(n,n)
  ma = Larik(a)
  mb = Larik(b)
  add(ma,mb)
  np.testing.assert_allclose(mb.numpy(),a+b)






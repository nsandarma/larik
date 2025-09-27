import math,numpy as np,time
from typing import List, Sequence

from src.dtype import Dtype
from device import Devices,Device
from src._buffer import Buffer
from src.helpers import (
    get_shape,fully_flatten,
    compute_strides,is_jagged)
from src.dtype import _Dtype

class MathTrait:pass

class Larik(MathTrait):
  def __init__(self,data,dtype=Dtype.float32,device=Devices.CPU,**kwargs):
    self._dtype = _Dtype(dtype)
    if isinstance(data,(list,tuple,dict)):
      if kwargs.get("shape",False):
        shape = kwargs["shape"]
        assert isinstance(shape,(list,tuple)), f"shape is {type(shape)}"
        shape = kwargs["shape"]
      else: shape = get_shape(data)
      if kwargs.get("strides",False):
        strides = kwargs["strides"]
        assert isinstance(strides,(list,tuple)), f"shape is {type(strides)}"
      else: strides = compute_strides(shape)
      size = math.prod(shape)
      assert not(is_jagged(data)), "inhomogeneous shape" 
      _flatten_data = data if len(shape) == 1 else fully_flatten(data)
      _buffer = Buffer(self.dtype,_flatten_data,device=device,shape=shape)
    elif isinstance(data,np.ndarray):
      shape = data.shape
      size  = data.size
      _buffer = Buffer.from_numpy(data.flatten(),shape=shape)
      strides = compute_strides(shape)
    elif isinstance(data,Buffer):
      _buffer = data
      shape = kwargs["shape"]
      strides = compute_strides(shape)
      size = math.prod(shape)

    else: raise NotImplementedError("alamak najisnyee datatype apa ni")

    self.device = Device(device)
    self._buffer = _buffer
    self._shape = shape
    self._ndim = len(shape)
    self._size = size
    self._strides = strides
    self._is_contiguous = True
    self._offset = 0
  
  @property
  def shape(self): return self._shape
  
  @property
  def strides(self): return self._strides

  @property
  def offset(self): return self._offset

  @property
  def ndim(self): return self._ndim

  @property
  def T(self): return self.transpose()

  @property
  def size(self): return self._size

  @property
  def is_contiguous(self) -> bool : return self._is_contiguous

  @property
  def dtype(self): return self._dtype

  def data(self) -> memoryview: return self._buffer.as_buffer()

  def buffer(self) -> Buffer: return self._buffer

  def flatten(self) -> List:
    assert self._is_contiguous, "data is not contiguous !"
    return self.buffer().as_flatten()
  
  def numpy(self) -> np.ndarray: 
    data = self.contiguous()
    return np.frombuffer(data.data(),dtype=data.dtype.name).reshape(self.shape)

  def tolist(self) -> List: return self.data().tolist()

  def reshape(self,*shape):
    assert math.prod(shape) == math.prod(self.shape), f"cannot reshape array of size {self.size} into shape {shape}"
    self._shape = shape
    buffer = self._buffer
    buffer.shape = shape
    lr = Larik(buffer,dtype=self.dtype.base(),device="",shape=shape)
    return lr

  def transpose(self, axes=None):
    if axes is None: axes = tuple(reversed(range(self._ndim)))
    else: assert len(axes) == self._ndim, "axes mismatch ndim"

    new_shape = tuple(self._shape[ax] for ax in axes)
    new_strides = tuple(self._strides[ax] for ax in axes)

    view = Larik.__new__(Larik)
    view._buffer = self._buffer
    view._dtype = self._dtype
    view._shape = new_shape
    view._ndim = len(new_shape)
    view._size = self._size
    view._strides = new_strides
    view._offset = 0
    view.device = self.device
    view._is_contiguous = False
    return view
  
  def __str__(self) -> str:
    return f"<shape:{self.shape} | size:{self.size} | strides:{self.strides} | dtypes:{self.dtype.name} | is_contiguous:{self._is_contiguous}>"

  def __repr__(self) -> str: return str(self)

  def __getitem__(self,idx): return self._getitem(idx)

  def _getitem(self,indices):
    if isinstance(indices,(list,tuple)):
      row,col = indices
      return self.data().tolist()[row][col]
    else: return self.data().tolist()[indices]

  def contiguous(self):
    if self.is_contiguous: return self

    new_data = [None] * self.size
    idx = [0] * self.ndim   # posisi multidimensi sekarang

    for k in range(self.size):
      # hitung flat index lama
      flat_idx = self._offset + sum(i * s for i, s in zip(idx, self.strides))
      new_data[k] = self.buffer().as_flatten()[flat_idx]

      # update idx seperti odometer
      for d in reversed(range(self.ndim)):
        idx[d] += 1
        if idx[d] < self.shape[d]: break
        idx[d] = 0

    new_buf = Buffer(self.dtype, new_data, device=self.device, shape=self.shape)
    lra = Larik.__new__(Larik)
    lra._buffer = new_buf
    lra._dtype = self.dtype
    lra._shape = self.shape
    lra._ndim = self.ndim
    lra._size = self.size
    lra._strides = self.strides
    lra._offset = 0
    lra.device = self.device
    lra._is_contiguous = True
    return lra

  @staticmethod
  def zeros(*shape,dtype=Dtype.float32):
    size = math.prod(shape)
    data = [0] * size
    buff = Buffer(_Dtype(dtype),data,shape=shape)
    lr = Larik(buff,dtype,shape=shape)
    return lr
  
  @staticmethod
  def ones(*shape,dtype=Dtype.float32):
    size = math.prod(shape)
    data = [1] * size
    buff = Buffer(_Dtype(dtype),data,shape=shape)
    lr = Larik(buff,dtype,shape=shape)
    return lr

  # TODO: in future will make it
  @staticmethod
  def rand(*shape): 
    rn = np.random.rand(*shape).astype("float32")
    return Larik(rn,dtype="float32")

  def __iadd__(self,val): 
    self.device.assign(self.size,val,"+",self.buffer(),self.dtype.name_c)
    return self

  def __isub__(self,val):
    self.device.assign(self.size,val,"-",self.buffer(),self.dtype.name_c)
    return self

  def __imul__(self,val):
    self.device.assign(self.size,val,"*",self.buffer(),self.dtype.name_c)
    return self

  def __idiv__(self,val):
    self.device.assign(self.size,val,"/",self.buffer(),self.dtype.name_c)
    return self

  def _arithmatic(self,lr,out,op):
    assert isinstance(lr,Larik), f"{lr} is not Larik"
    assert self.size == lr.size, f"{self.size} != {lr.size}"
    self.device.arithmatic(self.size,op,self.dtype,out.buffer(),self.buffer(),lr.buffer())

  def __add__(self,lr):
    out = self.zeros(*self.shape,dtype=self.dtype._dtype)
    self._arithmatic(lr,out,"+")
    return out

  def __sub__(self,lr):
    out = self.zeros(*self.shape,dtype=self.dtype._dtype)
    self._arithmatic(lr,out,"-")
    return out

  def __mul__(self,lr):
    out = self.zeros(*self.shape,dtype=self.dtype._dtype)
    self._arithmatic(lr,out,"*")
    return out

  def __div__(self,lr):
    out = self.zeros(*self.shape,dtype=self.dtype._dtype)
    self._arithmatic(lr,out,"/")
    return out

  def __matmul__(self):pass


# TODO: Memperbaiki contiguous dan viewer
if __name__ == "__main__":
  import numpy as np
  np.random.seed(42)
  np_arr = np.random.randint(2,100,(100,50)).astype(np.int32)
  my_arr = Larik.ones(1024,1024,dtype=Dtype.float32)
  my_arr1 = Larik.ones(1024,1024,dtype=Dtype.float32)
  my_arr_o = my_arr * my_arr1
  print(my_arr_o.numpy())

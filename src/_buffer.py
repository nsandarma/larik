import ctypes,struct,numpy as np,math
from src.dtype import _Dtype


class Buffer:
  def __init__(self,dtype:_Dtype,data,shape,device=None):
    self._dtype = dtype
    self.device = device
    self.shape = shape
    if device : raise NotImplementedError("NotImplementedError Device")
    else:
      if isinstance(data,bytes): self._packed = bytearray(data)
      else: self._packed = bytearray(struct.pack(f"{len(data)}{self.dtype.fmtstr}",*data))

  def as_buffer(self): return memoryview(self._packed).cast(self.dtype.fmtstr,shape=self.shape)
  
  def as_ptr(self): return ctypes.addressof(ctypes.c_uint8.from_buffer(self._packed))

  def as_ctypes(self): return (ctypes.c_uint8 * self.as_buffer().nbytes).from_buffer(self._packed)

  # follow data type , not raw buffer
  def as_ctypes_(self): return (self._dtype.to_cty * math.prod(self.shape)).from_buffer(self._packed)

  def data_as(self,obj): return ctypes.cast(self.as_ctypes(),obj)

  def as_flatten(self) -> list : return memoryview(self._packed).cast(self.dtype.fmtstr)
  
  @property
  def nbytes(self): return self.as_buffer().nbytes

  @property
  def itemsize(self): return self.as_buffer().itemsize

  @property
  def dtype(self): return self._dtype

  def __str__(self): return f"<Buffer dtype:{self.dtype} itemsize:{self.as_buffer().itemsize} nbytes:{self.nbytes}>"
  
  @staticmethod
  def from_numpy(data:np.ndarray,shape): return Buffer(_Dtype(data.dtype.name),data.tobytes(),shape=shape)


if __name__ == "__main__":
  rnd = np.random.rand(3,2).astype(np.float32).flatten()
  # b = Buffer.from_numpy(rnd)
  buff = Buffer.from_numpy(rnd)
  print(buff.as_buffer().tolist())




from enum import Enum,auto
from typing import List
from enum import Enum,auto


class Dtype(Enum):
  bool = 0; int8 = auto(); uint8 = auto(); int16 = auto(); uint16 = auto(); int32 = auto()
  uint32 = auto(); uint64 = auto(); int64 = auto(); float32 = auto(); float64 = auto()
  

def str_to_dtype(dtype_str:str):
  mapping = {
    "bool" : Dtype.bool,
    "int8": Dtype.int8, "uint8": Dtype.uint8, "int16": Dtype.int16,
    "uint16": Dtype.uint16, "int32": Dtype.int32, "uint32": Dtype.uint32,
    "int64": Dtype.int64, "uint64": Dtype.uint64, "float32": Dtype.float32,
    "float64": Dtype.float64,
  }
  return mapping[dtype_str]

FMTSTR:List[str] = ['?', 'b', 'B', 'h', 'H', 'i', 'I', 'q', 'Q', 'f', 'd']
DTYPE_NAMES_C:List[str] = ["bool","char","uchar","short","ushort","int","uint","long","long long","float","double"]
DTYPE_NAMES:List[str] = ["bool","int8","uint8","int16","uint16","int32","uint32","int64","uint64","float32","float64"]

assert len(FMTSTR) == len(DTYPE_NAMES) == len(DTYPE_NAMES_C)

class _Dtype:
  def __init__(self,dtype):
    # assert isinstance(dtype,(Dtype,str)),f"{type(dtype)} unknown!"
    if isinstance(dtype,str): 
      assert dtype in DTYPE_NAMES, f"{dtype} is unknown !"
      dtype = Dtype(DTYPE_NAMES.index(dtype))

    self._dtype = dtype

  def __str__(self): return str(self._dtype)

  @property
  def itemsize(self): return [1,1,1,2,2,4,4,8,8,2,4,8][self._dtype.value]

  @property
  def bits(self): 
    if self._dtype == Dtype.bool: return 1
    return int("".join([i for i in self.name if i.isdigit()]))

  @property
  def fmtstr(self): return FMTSTR[self._dtype.value]
  
  @property
  def name(self): return DTYPE_NAMES[self._dtype.value]

  @property
  def name_c(self): return DTYPE_NAMES_C[self._dtype.value]

  @property
  def base_ptr_ir(self):
    from llvmlite import ir
    return ir.IntType(8)

  @property
  def to_ir(self) :
    from llvmlite import ir
    if self.is_float(): return ir.FloatType() if self._dtype == Dtype.float32 else ir.DoubleType()
    else: return ir.IntType(self.bits)

  @property
  def to_ir_ptr(self): return self.to_ir.as_pointer()

  @property
  def to_cty(self):
    import ctypes
    CTYPES = [ctypes.c_bool,ctypes.c_int8,ctypes.c_uint8,ctypes.c_int16,ctypes.c_uint16,ctypes.c_int32,ctypes.c_uint32,
              ctypes.c_int64,ctypes.c_uint64,ctypes.c_float,ctypes.c_double]
    return CTYPES[self._dtype.value]

  @property
  def to_cty_ptr(self):
    import ctypes
    return ctypes.POINTER(self.to_cty)


  def base(self): return self._dtype

  def __eq__(self, value: object, /) -> bool:
    if isinstance(value,str):
      assert value in DTYPE_NAMES, f"{value} is unknown"
      return self._dtype == Dtype(DTYPE_NAMES.index(value))
    elif isinstance(value,Dtype): return self._dtype == value
    elif isinstance(value,_Dtype): return self._dtype == value._dtype
    else: raise NotImplementedError(f"{value} is unknown type")

  def is_float(self): return "float" in self.name

  def is_int(self): return "int" in self.name

  def is_unsigned(self): return "u" in self.name

  def is_bool(self): return self._dtype is Dtype.bool

def promote_dtype(dtype1, dtype2):
  """Promote dtype seperti NumPy"""
  priority = {
    Dtype.int8: 0, Dtype.uint8: 1, Dtype.int16: 2, Dtype.uint16: 3,
    Dtype.int32: 4, Dtype.uint32: 5, Dtype.int64: 6, Dtype.uint64: 7,
    Dtype.float32: 8, Dtype.float64: 9,
  }
  return dtype1 if priority[dtype1] >= priority[dtype2] else dtype2


if __name__ == "__main__" :
  import ctypes
  d2 = _Dtype(Dtype.int32)
  print(d2.to_cty)


import ctypes
from llvmlite import binding, ir
from dataclasses import dataclass
from src.dtype import _Dtype,Dtype
from src.codegen import RendererModule

from src.larik import Larik
import numpy as np,time

class LLVMCompiler:
  def __init__(self,block:RendererModule):
    binding.initialize()
    binding.initialize_native_target()
    binding.initialize_native_asmprinter()

    self.module = block.module
    self.args_type = block.args_type
    self.return_type = block.return_type
    self.func_name = block.func.name

    self.engine = self._create_exec_machine()
    self.mod = self._compile_ir(str(block.module))

  def _create_exec_machine(self):
    target = binding.Target.from_default_triple()
    target_machine = target.create_target_machine()
    backing_mod = binding.parse_assembly("")
    engine = binding.create_mcjit_compiler(backing_mod, target_machine)
    return engine

  def _compile_ir(self, llvm_ir):
    mod = binding.parse_assembly(llvm_ir)
    mod.verify()
    self.engine.add_module(mod)
    self.engine.finalize_object()
    self.engine.run_static_constructors()
    return mod
  
  def get_func(self,return_dtype,*args_dtype):
    addr = self.engine.get_function_address(self.func_name)
    CFUN = ctypes.CFUNCTYPE(return_dtype,
                            *args_dtype)
    return CFUN(addr)

def add(a,b,n): return Larik([a[i] + b[i] for i in range(n)],dtype="int32")


def test1():
  dty = _Dtype("int32")
  n = 4085
  a = Larik.ones(n,dtype="int32")
  b = Larik.ones(n,dtype="int32")
  c = Larik.zeros(n,dtype="int32")

  builder = RendererModule("test")
  args_dtype = [dty,dty,dty]
  builder.function("addvect",None,*args_dtype)
  builder.add()
  comp = LLVMCompiler(builder)
  fn = comp.get_func(None,dty.to_cty_ptr,dty.to_cty_ptr,dty.to_cty_ptr,ctypes.c_longlong)
  tic = time.monotonic()
  fn(
    c.buffer().as_ctypes_(),
    a.buffer().as_ctypes_(),
    b.buffer().as_ctypes_(),
    ctypes.c_longlong(n)
  )
  toc = time.monotonic()
  print(f"times : {toc-tic}")
  return c

def test2():
  dty = _Dtype("int32")
  n = 4085
  a = Larik.ones(n,dtype="int32")
  b = Larik.ones(n,dtype="int32")
  c = Larik.zeros(n,dtype="int32")

  builder = RendererModule("test")
  args_dtype = [dty,dty,dty]
  builder.function("addvect",None,*args_dtype)
  builder.add()

  print(builder.func)
  comp = LLVMCompiler(builder)
  fn = comp.get_func(None,dty.to_cty_ptr,dty.to_cty_ptr,dty.to_cty_ptr,ctypes.c_longlong)
  tic = time.monotonic()
  fn(
    c.buffer().data_as(c.dtype.to_cty_ptr),
    a.buffer().data_as(c.dtype.to_cty_ptr),
    b.buffer().data_as(c.dtype.to_cty_ptr),
    ctypes.c_longlong(n)
  )
  toc = time.monotonic()
  print(f"times : {toc-tic}")
  return c


if __name__ == "__main__":
  c1 = test1()
  c2 = test2()
  np.testing.assert_allclose(c1.numpy(),c2.numpy())

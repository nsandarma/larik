import ctypes
from llvmlite import binding, ir
from dataclasses import dataclass
from src.dtype import _Dtype,Dtype
from src.codegen import RendererModule
import src.backends.autogen.libllvm as llvm
from src.helpers import OSX
from src.backends.support.elf import jit_loader

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


def cerr(): return ctypes.pointer(ctypes.pointer(ctypes.c_char()))

def expect(x, err, ret=None):
  if x: raise RuntimeError(llvm.string_cast(err.contents) if not isinstance(err, str) else err)
  return ret

class LLVMCompiler():
  def __init__(self, host_arch:str):
    for component in ['Target', 'TargetInfo', 'TargetMC', 'AsmParser', 'AsmPrinter']: getattr(llvm, f'LLVMInitialize{host_arch}{component}')()

    triple = {'AArch64': b'aarch64', 'X86': b'x86_64'}[host_arch] + b'-none-unknown-elf'
    target = expect(llvm.LLVMGetTargetFromTriple(triple, ctypes.pointer(tgt:=llvm.LLVMTargetRef()), err:=cerr()), err, tgt)
    # +reserve-x18 here does the same thing as -ffixed-x18 in ops_cpu.py, see comments there for why it's needed on arm osx
    cpu, feats = ctypes.string_at(llvm.LLVMGetHostCPUName()), (b'+reserve-x18,' if OSX else b'') + ctypes.string_at(llvm.LLVMGetHostCPUFeatures())
    self.target_machine = llvm.LLVMCreateTargetMachine(target, triple, cpu, feats,
                                                       llvm.LLVMCodeGenLevelDefault, llvm.LLVMRelocPIC, llvm.LLVMCodeModelDefault)

    self.pbo = llvm.LLVMCreatePassBuilderOptions()
    if 1:
      self.passes = b'default<O2>'
      llvm.LLVMPassBuilderOptionsSetLoopUnrolling(self.pbo, True)
      llvm.LLVMPassBuilderOptionsSetLoopVectorization(self.pbo, True)
      llvm.LLVMPassBuilderOptionsSetSLPVectorization(self.pbo, True)
      llvm.LLVMPassBuilderOptionsSetVerifyEach(self.pbo, True)
    else:
      self.passes = b'default<O0>'


  def __del__(self): llvm.LLVMDisposePassBuilderOptions(self.pbo)

  def compile(self, src:str) -> bytes:
    src_buf = llvm.LLVMCreateMemoryBufferWithMemoryRangeCopy(ctypes.create_string_buffer(src_bytes:=src.encode()), len(src_bytes), b'src')
    mod = expect(llvm.LLVMParseIRInContext(llvm.LLVMGetGlobalContext(), src_buf, ctypes.pointer(m:=llvm.LLVMModuleRef()), err:=cerr()), err, m)
    expect(llvm.LLVMVerifyModule(mod, llvm.LLVMReturnStatusAction, err:=cerr()), err)
    expect(llvm.LLVMRunPasses(mod, self.passes, self.target_machine, self.pbo), 'failed to run passes')
    obj_buf = expect(llvm.LLVMTargetMachineEmitToMemoryBuffer(self.target_machine, mod, llvm.LLVMObjectFile, err:=cerr(),
                                                              ctypes.pointer(buf:=llvm.LLVMMemoryBufferRef())), err, buf)
    llvm.LLVMDisposeModule(mod)
    obj = ctypes.string_at(llvm.LLVMGetBufferStart(obj_buf), llvm.LLVMGetBufferSize(obj_buf))
    llvm.LLVMDisposeMemoryBuffer(obj_buf)
    return jit_loader(obj)

if __name__ == "__main__":
  csrc = r"""
  define i32 @add(i32 %a, i32 %b) {
  entry:
    %sum = add i32 %a, %b
    ret i32 %sum
  }
  """
  comp = LLVMCompiler("AArch64")
  print(comp.compile(csrc))



from os import stat
import ctypes.util,sys,platform
from src.codegen import CRenderer
from src.backends.cpu import ClangJITCompiler
from src.helpers import mv_address,OSX,cpu_time_execution
from enum import Enum,auto

class LLVMProgram:
  def __init__(self):pass

class CPUProgram:
  rt_lib = ctypes.CDLL(ctypes.util.find_library('System' if OSX else 'kernel32') if OSX or sys.platform == "win32" else 'libgcc_s.so.1')
  atomic_lib = ctypes.CDLL(ctypes.util.find_library('atomic')) if sys.platform == "linux" else None

  def __init__(self, name:str, lib:bytes):
    if sys.platform == "win32":
      PAGE_EXECUTE_READWRITE = 0x40
      MEM_COMMIT =  0x1000
      MEM_RESERVE = 0x2000
      ctypes.windll.kernel32.VirtualAlloc.restype = ctypes.c_void_p
      self.mem = ctypes.windll.kernel32.VirtualAlloc(ctypes.c_void_p(0), ctypes.c_size_t(len(lib)), MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE)
      ctypes.memmove(self.mem, lib, len(lib))
      ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
      proc = ctypes.windll.kernel32.GetCurrentProcess()
      ctypes.windll.kernel32.FlushInstructionCache(ctypes.c_void_p(proc), ctypes.c_void_p(self.mem), ctypes.c_size_t(len(lib)))
      self.fxn = ctypes.CFUNCTYPE(None)(self.mem)
    else:
      from mmap import mmap, PROT_READ, PROT_WRITE, PROT_EXEC, MAP_ANON, MAP_PRIVATE
      MAP_JIT = 0x0800
      self.mem = mmap(-1, len(lib), MAP_ANON | MAP_PRIVATE | (MAP_JIT if OSX else 0), PROT_READ | PROT_WRITE | PROT_EXEC)

      if OSX: CPUProgram.rt_lib.pthread_jit_write_protect_np(False)
      self.mem.write(lib)
      if OSX: CPUProgram.rt_lib.pthread_jit_write_protect_np(True)

      CPUProgram.rt_lib["__clear_cache"](ctypes.c_void_p(mv_address(self.mem)), ctypes.c_void_p(mv_address(self.mem) + len(lib)))

      self.fxn = ctypes.CFUNCTYPE(None)(mv_address(self.mem))

  def __call__(self, *bufs, vals=(), wait=False):
    args = list(bufs) + list(vals)
    if platform.machine() == "arm64" and OSX: args = args[:8] + [ctypes.c_int64(a) if isinstance(a, int) else a for a in args[8:]]
    return cpu_time_execution(lambda: self.fxn(*args), enable=wait)

  def __del__(self):
    if sys.platform == 'win32': ctypes.windll.kernel32.VirtualFree(ctypes.c_void_p(self.mem), ctypes.c_size_t(0), 0x8000) #0x8000 - MEM_RELEASE

class Devices(Enum): CPU = auto(); LLVM = auto();

class Device:
  def __init__(self,devices:Devices):
    if devices == Devices.CPU:
      self.renderer = CRenderer
      self.compiler = ClangJITCompiler()
    else:raise ValueError("devices compiler not implemented")

  def assign(self,size,val,op,buff,dtype):
    csimd = self.renderer.assignOps(size,val,op,dtype)
    prog = CPUProgram("assign", self.compiler.compile(csimd))
    prog(buff.as_ctypes(),val)

  def arithmatic(self,size,op,dtype,buff0,buff1,buff2):
    csimd = self.renderer.arithmatic(dtype,size,op)
    prog = CPUProgram("arithmatic",self.compiler.compile(csimd))
    prog(buff0.as_ctypes(),buff1.as_ctypes(),buff2.as_ctypes())

if __name__ == "__main__":
  from src.backends.cpu import ClangJITCompiler
  from src.larik import Larik
  import time
  csrc = r'''
  void add(float *a,float b,int n) {
    for (int i = 0; i < n ; i++){
      a[i] += b;
    }
  }
  '''

  # n = 5
  # arr = (ctypes.c_float * n)(1.0, 2.0, 3.0, 4.0, 5.0)
  # comp = ClangJITCompiler()
  # out = ctypes.c_float()
  # prog = CPUProgram("add", comp.compile(csrc))
  # prog(arr, ctypes.c_float(20), n)
  # print([arr[i] for i in range(n)])

  N = 1024 
  arr = Larik.ones(N,N)
  tic = time.monotonic()
  comp = ClangJITCompiler()
  out = ctypes.c_float()
  prog = CPUProgram("add", comp.compile(csrc))
  prog(arr.buffer().as_ctypes_(), ctypes.c_float(20), arr.size)
  print(arr.numpy())





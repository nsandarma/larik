import ctypes
import ctypes.util
import numpy as np
from enum import IntEnum,Enum,auto
from llvmlite import ir, binding
from src._dtype import Dtype  # Assuming this is correctly implemented


class LAYOUT(IntEnum): RowMajor = 101; ColMajor = 102

class TRANSPOSE(IntEnum): NoTrans = 111; Trans = 112; ConjTrans = 113

class OPERATION(Enum):
  GEMM = auto()   # General Matrix-Matrix multiplication
  GEMV = auto()   # General Matrix-Vector multiplication
  GER  = auto()   # General Rank-1 update
  DOT  = auto()   # Dot product
  AXPY = auto()   # y := a*x + y
  SCAL = auto()   # x := a*x
  COPY = auto()   # y := x
  TRSM = auto()   # Solve triangular system
  SYRK = auto()   # Symmetric rank-k update
  TRMM = auto()   # Triangular matrix-matrix multiply

class BLASLEVEL(Enum): VECwVEC = auto(); MATwVEC = auto(); MATwMAT = auto();

# Preload BLAS library
blas_path = ctypes.util.find_library("blas") or ctypes.util.find_library( "openblas") or ctypes.util.find_library("mkl")
if not blas_path: raise RuntimeError("BLAS library not found!")
ctypes.CDLL(blas_path)

class CBlas:
  def __init__(self, dtype: Dtype):
    if dtype not in [Dtype.float32, Dtype.float64]: raise ValueError(f"{dtype} is not supported by BLAS!")
    self.module = ir.Module("test_123")
    self.engine = None
    self.function = None
    self.dtype = dtype
    self.ptr = Dtype.to_llvm_ir(dtype).as_pointer()
    self.float_type = ir.DoubleType() if self.dtype == Dtype.float64 else ir.FloatType()

  def declare_gemm(self):
    int_type = ir.IntType(32)
    name = "cblas_dgemm" if self.dtype == Dtype.float64 else "cblas_sgemm"
    func_blas_type = ir.FunctionType(ir.VoidType(), [
        int_type, int_type, int_type,  # layout, TransA, TransB
        int_type, int_type, int_type,  # M, N, K
        self.float_type,               # alpha
        self.ptr, int_type,            # A, lda
        self.ptr, int_type,            # B, ldb
        self.float_type,               # beta
        self.ptr, int_type             # C, ldc
    ])
    func_blas_gemm = ir.Function(self.module, func_blas_type, name=name)
    return func_blas_gemm

  def gemm(self, layout=LAYOUT.RowMajor, transA=TRANSPOSE.NoTrans, transB=TRANSPOSE.NoTrans):
    int_type = ir.IntType(32)
    func_type = ir.FunctionType(ir.VoidType(), [
        self.ptr, self.ptr, self.ptr,  # A, B, C
        int_type, int_type, int_type   # M, N, K
    ])
    func = ir.Function(self.module, func_type, name="matmul")
    self.function = func

    A, B, C, M, N, K = func.args
    block = func.append_basic_block("entry")
    builder = ir.IRBuilder(block)
    gemm = self.declare_gemm()

    layout_const = ir.Constant(int_type, int(layout))
    transA_const = ir.Constant(int_type, int(transA))
    transB_const = ir.Constant(int_type, int(transB))
    alpha = ir.Constant(self.float_type, 1.0)
    beta = ir.Constant(self.float_type, 0.0)

    # Adjust leading dimensions based on layout and transpose
    lda = N if layout == LAYOUT.RowMajor and transA == TRANSPOSE.NoTrans else K
    ldb = N if layout == LAYOUT.RowMajor and transB == TRANSPOSE.NoTrans else K
    ldc = N if layout == LAYOUT.RowMajor else M

    builder.call(gemm, [
      layout_const, transA_const, transB_const,
      M, N, K,
      alpha,
      A, lda,
      B, ldb,
      beta,
      C, ldc
    ])
    builder.ret_void()
  

  def compile(self):
    try:
      llvm_ir = str(self.module)
      llvm_mod = binding.parse_assembly(llvm_ir)
      llvm_mod.verify()

      target = binding.Target.from_default_triple()
      target_machine = target.create_target_machine()
      engine = binding.create_mcjit_compiler(llvm_mod, target_machine)
      engine.finalize_object()
      self.engine = engine

      addr = engine.get_function_address(self.function.name)
      if not addr: raise RuntimeError("Failed to get function address for matmul")
      ctype = ctypes.c_double if self.dtype == Dtype.float64 else ctypes.c_float
      return ctypes.CFUNCTYPE( None,
        ctypes.POINTER(ctype),
        ctypes.POINTER(ctype),
        ctypes.POINTER(ctype),
        ctypes.c_int, ctypes.c_int, ctypes.c_int
      )(addr)
    except Exception as e: raise RuntimeError(f"Compilation failed: {str(e)}")

  def cleanup(self):
    if self.engine:
      self.engine.finalize_object()
      self.engine = None

  def renderer(self): return str(self.module)

if __name__ == "__main__":
  import time
  N = 4096
  dtype = Dtype.float64
  blas = CBlas("matmul_blas", dtype)
  func = blas.gemm()
  matmul_fn = blas.compile()

  A = np.random.rand(N, N).astype(np.float64)
  B = np.random.rand(N, N).astype(np.float64)
  C = np.zeros((N, N), dtype=np.float64)

  if A.shape != (N, N) or B.shape != (N, N) or C.shape != (N, N): raise ValueError("Matrix dimensions are incompatible!")

  tic = time.monotonic()

  matmul_fn(
    A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    B.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    C.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    N, N, N
  )
  toc = time.monotonic()
  print(f"times : {toc-tic}")
  # Verify result
  tic = time.monotonic()
  C_expected = np.dot(A, B)
  toc = time.monotonic()
  print(f"times : {toc-tic}")
  np.testing.assert_allclose(C, C_expected)
  print("Matrix multiplication verified successfully!")

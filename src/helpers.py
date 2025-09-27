from typing import Optional, Sequence,Union,TypeGuard,Any
import ctypes,struct,math,platform,time

OSX = platform.system() == "Darwin"

def colored(st, color:Optional[str], background=False): return f"\u001b[{10*background+60*(color.upper() == color)+30+['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'].index(color.lower())}m{st}\u001b[0m" if color is not None else st  # replace the termcolor library with one line  # noqa: E501
class Colors:
  HEADER = '\033[95m'
  BLUE = '\033[94m'
  GREEN = '\033[92m'
  RED = '\033[91m'
  YELLOW = '\033[93m'
  END = '\033[0m'

def get_ratio_time(time_larik, time_numpy):
  """Calculate and display the relative speed of larik vs numpy."""
  if time_numpy > time_larik and time_larik > 0:
    ratio = time_numpy / time_larik
    print(f"{Colors.GREEN}larik is {ratio:.2f} times faster than numpy!{Colors.END}")
  elif time_larik > time_numpy and time_numpy > 0:
    ratio = time_larik / time_numpy
    print(f"{Colors.RED}numpy is {ratio:.2f} times faster than larik!{Colors.END}")
  else: print(f"{Colors.YELLOW}Both implementations have identical execution times or one of the times is zero.{Colors.END}")

def get_shape(nested_list):
  shape = []
  current = nested_list
  while isinstance(current, list):
    shape.append(len(current))
    if not current: break
    current = current[0]
  return tuple(shape)

def fully_flatten(nested_list:Union[list,tuple,dict]):
  assert isinstance(nested_list,(list,tuple,dict)), f"nested_list if {type(nested_list)}" 
  flat = []
  stack = [iter(nested_list)]
  while stack:
    curr_iter = stack[-1]
    try: elem = next(curr_iter)
    except StopIteration:
      stack.pop()
      continue
    if isinstance(elem, list): stack.append(iter(elem))
    else: flat.append(elem)
  return flat

def compute_strides(shape):
  strides = []
  for i in range(len(shape)-1, -1, -1):
    if i == len(shape)-1: strides.insert(0,1)
    else: strides.insert(0, strides[0]*shape[i+1])
  return tuple(strides)

def contains_float(data_flatten:Sequence): return any(isinstance(x,float) for x in data_flatten)

def from_mv(mv:memoryview, to_type=ctypes.c_char):
  return ctypes.cast(ctypes.addressof(to_type.from_buffer(mv)), ctypes.POINTER(to_type * len(mv))).contents

def all_int(t: Sequence[Any]) -> TypeGuard[tuple[int, ...]]: return all(isinstance(s, int) for s in t)

def to_mv(ptr:int, sz:int) -> memoryview: return memoryview(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8 * sz)).contents).cast("B")
def mv_address(mv): return ctypes.addressof(ctypes.c_char.from_buffer(mv))
def flat_mv(mv:memoryview): return mv if len(mv) == 0 else mv.cast("B", shape=(mv.nbytes,))
def get_address(buf):
  if isinstance(buf, ctypes.Array): return ctypes.addressof(buf[0])
  else: return ctypes.cast(buf, ctypes.c_void_p).value

def is_jagged(seq):
  if not isinstance(seq, (list, tuple)): return False
  try: first_len = len(seq[0])
  except Exception: return False
  for x in seq:
    if not isinstance(x, (list, tuple)) or len(x) != first_len: return True
    if is_jagged(x):  return True
  return False

def truncate_fp16(x):
  try: return struct.unpack("@e", struct.pack("@e", float(x)))[0]
  except OverflowError: return math.copysign(math.inf, x)

def truncate_bf16(x):
  max_bf16 = struct.unpack('f', struct.pack('I', 0x7f7f0000))[0]
  if x > max_bf16 or x < -max_bf16: return math.copysign(math.inf, x)
  f32_int = struct.unpack('I', struct.pack('f', x))[0]
  bf = struct.unpack('f', struct.pack('I', f32_int & 0xFFFF0000))[0]
  return bf

def cpu_time_execution(cb, enable):
  if enable: st = time.perf_counter()
  cb()
  if enable: return time.perf_counter()-st


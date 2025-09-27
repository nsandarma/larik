import sys,platform,subprocess,os
from src.backends.support.elf import jit_loader

class ClangJITCompiler:
  def __init__(self,cache_key="compile_clang_jit"): 
    self.cache_key = cache_key

  def compile(self,src:str):
    # -fno-math-errno is required for __builtin_sqrt to become an instruction instead of a function call
    # x18 is a reserved platform register. It is clobbered on context switch in macos and is used to store TEB pointer in windows on arm, don't use it
    target = 'x86_64' if sys.platform == 'win32' else platform.machine()
    # args = ['-march=native', f'--target={target}-none-unknown-elf', '-O2', '-fPIC', '-ffreestanding', '-fno-math-errno', '-nostdlib']
    args = ['-march=native', f'--target={target}-none-unknown-elf', '-O2', '-fPIC', '-ffreestanding', '-fno-math-errno']
    arch_args = ['-ffixed-x18'] if target == 'arm64' else []
    obj = subprocess.check_output(["CC", '-c', '-x', 'c', *args, *arch_args, '-', '-o', '-'], input=src.encode('utf-8'))
    return jit_loader(obj)

# clang++ -std=c++17 test.cc -o test -Iinclude -framework Accelerate -DACCELERATE_NEW_LAPACK



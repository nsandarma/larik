# Render Kernel 
from typing import List
from llvmlite import ir

from enum import Enum,auto
OP = ["+","/","-","*","%"]

class RendererModule:
  def __init__(self,module_name:str):
    self.module = ir.Module(module_name)

  def function(self,name:str,return_type,*args_type):
    self.return_type = return_type
    self.args_type = args_type
    if not return_type: return_type = ir.VoidType()
    args_type = [i.to_ir_ptr for i in args_type] 
    args_type.append(ir.IntType(64))
    func_type = ir.FunctionType(return_type,args_type)
    self.func = ir.Function(self.module,func_type,name=name)

  def add(self): 
    fn = self.func
    out_ptr, a_ptr, b_ptr, n = fn.args

    entry = fn.append_basic_block("entry")
    builder = ir.IRBuilder(entry)

    # i = 0
    i32 = ir.IntType(64)
    i_var = builder.alloca(i32, name="i")
    builder.store(ir.Constant(i32, 0), i_var)

    loop_cond = fn.append_basic_block("loop_cond")
    loop_body = fn.append_basic_block("loop_body")
    loop_end = fn.append_basic_block("loop_end")

    builder.branch(loop_cond)

    # cond
    builder.position_at_end(loop_cond)
    i_val = builder.load(i_var, name="i_val")
    cond = builder.icmp_signed("<", i_val, n)
    builder.cbranch(cond, loop_body, loop_end)

    # body
    builder.position_at_end(loop_body)
    # gep pointers
    out_gep = builder.gep(out_ptr, [i_val], inbounds=True)
    a_gep = builder.gep(a_ptr, [i_val], inbounds=True)
    b_gep = builder.gep(b_ptr, [i_val], inbounds=True)

    a_val = builder.load(a_gep, name="a_val")
    b_val = builder.load(b_gep, name="b_val")
    res = builder.add(a_val, b_val, name="res")
    builder.store(res, out_gep)

    # i++
    next_i = builder.add(i_val, ir.Constant(i32, 1))
    builder.store(next_i, i_var)
    builder.branch(loop_cond)

    # end
    builder.position_at_end(loop_end)
    builder.ret_void()


class CRenderer:
  @staticmethod
  def assignOps(size:int,value,op:str,dtype:str):
    return f"""
    void assign({dtype}* restrict data0) {{
      for (int ridx0 = 0; ridx0 < {size}; ridx0++) {{
          *(data0+ridx0) = *(data0+ridx0) {op} {value};
      }}
    }}
    """
  
  @staticmethod
  def arithmatic(dtype,size:int,op:str):
    d = dtype.name_c
    if (size % 2 == 0):
      n_size = size / dtype.itemsize
      if d == "float":
        csrc = f"""
        typedef float float4 __attribute__((aligned(16),vector_size(16)));
        void arithmatic(float* restrict data0, float* restrict data1, float* restrict data2) {{
          for (int ridx0 = 0; ridx0 < {n_size}; ridx0++) {{
            int alu0 = (ridx0<<2);
            float4 val0 = *((float4*)((data1+alu0)));
            float4 val1 = *((float4*)((data2+alu0)));
            *((float4*)((data0+alu0))) = (float4){{(val0[0]{op}val1[0]),(val0[1]{op}val1[1]),(val0[2]{op}val1[2]),(val0[3]{op}val1[3])}};
          }}
        }}
        """
      else:
        csrc = f"""
        void arithmatic({d}* restrict data0, {d}* restrict data1, {d}* restrict data2) {{
          for (int ridx0 = 0; ridx0 < {n_size}; ridx0++) {{
            {d} alu0 = (ridx0<<2);
            {d} val0 = *(data1+alu0);
            {d} val1 = *(data2+alu0);
            {d} alu1 = (alu0+1);
            {d} val2 = *(data1+alu1);
            {d} val3 = *(data2+alu1);
            {d} alu2 = (alu0+2);
            {d} val4 = *(data1+alu2);
            {d} val5 = *(data2+alu2);
            {d} alu3 = (alu0+3);
            {d} val6 = *(data1+alu3);
            {d} val7 = *(data2+alu3);
            *(data0+alu1) = (val2{op}val3);
            *(data0+alu2) = (val4{op}val5);
            *(data0+alu3) = (val6{op}val7);
            *(data0+alu0) = (val0{op}val1);
          }}
        }}
        """
    else:
      csrc =  f"""
      void arithmatic({d}* restrict data0, {d}* restrict data1, {d}* restrict data2) {{
        for (int ridx0 = 0; ridx0 < {size}; ridx0++) {{
          {d} val0 = *(data1+ridx0);
          {d} val1 = *(data2+ridx0);
          *(data0+ridx0) = (val0{op}val1);
        }}
      }}
      """
    return csrc




from llvmlite import ir
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


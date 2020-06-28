"""
Customized optimziation passes for BERT
This is based on @t-vi 's notebooks
"""
import threading

import tvm
from tvm import relay

class ShapeConstDedupMutator(tvm.relay.ExprMutator):
    def __init__(self):
        super().__init__()
        self.shape_consts = {}

    def visit_call(self, call):
        if (isinstance(call.op, tvm.ir.Op) and call.op.name == "reshape"
            and isinstance(call.args[1], tvm.relay.Constant)):
            assert list(call.attrs.newshape) == list(call.args[1].data.asnumpy())
            new_fn = self.visit(call.op)
            new_args = [self.visit(arg) for arg in call.args]
            const = new_args[1]
            assert const.data.dtype.startswith('int') and len(const.data.shape)==1
            key = tuple(const.data.asnumpy())
            if key in self.shape_consts:
                new_args[1] = self.shape_consts[key]
            else:
                self.shape_consts[key] = new_args[1]
            return tvm.relay.Call(new_fn, new_args, call.attrs)
        return super().visit_call(call)

@tvm.relay.transform.function_pass(opt_level=1)
def ShapeConstDedup(fn, mod, ctx):
    return ShapeConstDedupMutator().visit(fn)

def optimize_bert_worker(mod, params, ret_list):
    new_mod = ShapeConstDedup(mod)
    new_mod = tvm.relay.transform.EliminateCommonSubexpr()(new_mod)
    BindPass = tvm.relay.transform.function_pass(lambda fn, new_mod, ctx:
            tvm.relay.build_module.bind_params_by_name(fn, params), opt_level=1)
    new_mod = BindPass(new_mod)
    new_mod = tvm.relay.transform.FoldConstant()(new_mod)
    new_mod = tvm.relay.transform.CombineParallelBatchMatmul()(new_mod)
    new_mod = tvm.relay.transform._ffi_api.BatchMatmulWeightTranspose()(new_mod)
    new_mod = tvm.relay.transform.FoldConstant()(new_mod)
    ret_list.append(new_mod)

def optimize_bert(mod, params):
    # wrap the FoldConstant pass in a new thread to avoid the conflict
    # between python's multiprocessing and tvm's thread pool
    ret_list = []
    t = threading.Thread(target=optimize_bert_worker, args=(mod, params, ret_list))
    t.start()
    t.join()
    return ret_list[0]


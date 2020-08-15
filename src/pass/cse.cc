#include <tvm/api_registry.h>
#include <tvm/ir_pass.h>


namespace tvm {
namespace ir {


Stmt CSE(Stmt stmt, Stmt src)
{
        LOG(INFO) << "stmt: " << stmt;
        LOG(INFO) << "src: "  << src;
        return stmt;
}


TVM_REGISTER_API("ir_pass.CSE").set_body_typed(CSE);

}  // namespace ir
}  // namespace tvm

#include <tvm/ir_pass.h>


namespace tvm {
namespace ir {


Stmt CSE(Stmt stmt, Stmt src)
{
        LOG(INFO) << "stmt: " << stmt;
        LOG(INFO) << "src: "  << src;
        return stmt;
}


}  // namespace ir
}  // namespace tvm

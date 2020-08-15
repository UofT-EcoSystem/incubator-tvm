#include <tvm/ir_pass.h>


namespace tvm {
namespace ir {


Stmt CSE(Stmt stmt)
{
        LOG(INFO) << "Optimizing for statement " << stmt;

        return stmt;
}


}  // namespace ir
}  // namespace tvm

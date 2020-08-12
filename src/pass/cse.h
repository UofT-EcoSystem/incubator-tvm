#pragma once

#include <queue>

#include <tvm/ir_visitor.h>

namespace tvm {
namespace ir {

/// @brief Eliminiate common subexpressions.
void CSE(const Tensor & src,
         Tensor * const ptgt)
{
        Tensor & tgt = *ptgt;

        LOG(INFO) << "Decomposing Tensor " << tgt->op->name;

        std::queue < Tensor > worklist;
        worklist.push(tgt);
        for (; !worklist.empty(); worklist.pop())
        {
                
        }  // for (workitem ∈ worklist)
}

}  // namespace ir
}  // namespace tvm

#pragma once

#include <tvm/ir_visitor.h>

namespace tvm {
namespace ir {

/// @brief Eliminiate common subexpressions.
void CSE(const Tensor & src,
         Tensor * const ptgt)
{
        Tensor & tgt = *ptgt;

        if (const ComputeOpNode * compute = tgt->op.as < ComputeOpNode > ()) 
        {
                PostOrderVisit(comp->body[tgt->value_index],
                        [](const NodeRef & node_ref)
                        {
                                LOG(INFO) << node_ref;
                        });
        }
}

}  // namespace ir
}  // namespace tvm

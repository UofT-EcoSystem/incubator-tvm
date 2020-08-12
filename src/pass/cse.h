#pragma once

#include <queue>
#include <unordered_set>

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
        std::unordered_set < Tensor > visited_workitems;
        worklist.push(tgt);
        for (; !worklist.empty(); worklist.pop())
        {
                const Tensor & workitem = worklist.front();

                if (visited_workitems.count(workitem) != 0)
                {
                        continue;
                }
                visited_workitems.insert(workitem);

                LOG(INFO) << "Visiting Tensor " << workitem->op->name;

                if (const ComputeOpNode * compute_op =
                    workitem->op.as < ComputeOpNode > ()) 
                {
                        for (const Tensor & input : compute_op->InputTensors())
                        {
                                worklist.push(input);
                        }
                }
                if (const PlaceholderOpNode * placeholder_op = 
                    workitem->op.as < PlaceholderOpNode > ())
                {
                        LOG(INFO) << "Visiting Placeholder " << placeholder_op;
                }
        }  // for (workitem ∈ worklist)
}

}  // namespace ir
}  // namespace tvm

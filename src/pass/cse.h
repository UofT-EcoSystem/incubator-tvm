#pragma once

#include <functional>
#include <queue>
#include <unordered_set>

#include <tvm/ir_visitor.h>

namespace tvm {
namespace ir {


class IRPreOrderVisitor : public IRVisitor
{
private:
        std::function < void(const NodeRef &) >  _visit_func;
        std::unordered_set < const Node * >      _visited_nodes;
public:
        explicit IRPreOrderVisitor(
                std::function < void(const NodeRef &) > visit_func)
                : _visit_func(visit_func) 
        {}

        void Visit(const NodeRef & node) final
        {
                if (_visited_nodes.count(node.get()) != 0)
                {
                        return;
                }
                _visited_nodes.insert(node.get());
                _visit_func(node);
                IRVisitor::Visit(node);
        }
};  // class IRPreOrderVisitor;


/// @brief  Common Subexpression Elimination (Top-Level Function Call)
void CSE(const Tensor & src,
         Tensor * const ptgt)
{
        Tensor & tgt = *ptgt;

        // TODO: We limit the scope of analysis to compute.gamma.grad, but will
        //       remove this limitation in later stages.
        if (tgt->op->name != "compute.gamma.grad")
        {
                return;
        }

        std::queue < Tensor > worklist;
        std::unordered_set < Tensor > visited_workitems;
        worklist.push(tgt);
        for (; !worklist.empty(); worklist.pop())
        {
                const Tensor & workitem = worklist.front();

                if (visited_workitems.count(workitem) != 0)
                {
                        // continue;
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
                        LOG(INFO) << "Visiting Placeholder " << workitem->op;
                }
        }  // for (workitem ∈ worklist)
}

}  // namespace ir
}  // namespace tvm

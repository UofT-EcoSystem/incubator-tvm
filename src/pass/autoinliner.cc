#include "autoinliner.h"


namespace tvm {
namespace ir {


Expr BodyStmtAutoInliner::Mutate_(
        const Call * op,
        const Expr & e)
{
        Expr expr = IRMutator::Mutate_(op, e);

        if (op->func == src_op)
        {
                CHECK_EQ(op->value_index, 0);
                CHECK_EQ(op->call_type, Call::CallType::Halide);

                const ComputeOpNode * compute_op
                        = src_op.as < ComputeOpNode > ();
                CHECK(compute_op != nullptr);

                if (compute_op->reduce_axis.empty())
                {
                        CHECK_EQ(op->args.size(), src_axis_vars.size());

                        Map < Var, Expr > vmap;
                        for (size_t i = 0; i < src_axis_vars.size(); ++i)
                        {
                                vmap.Set(src_axis_vars[i], op->args[i]);
                        }
                        expr = Substitute(Evaluate::make(expr), vmap)
                               .as < Evaluate > ()->value;
                        return expr;
                }
                else
                {
                        // In the case in which the source Op matches the
                        // function name, but NOT a reduction. We still have to
                        // make an udpate to the CallNode to point it to the
                        // latest version of the body statement.
                        return Call::make(op->type,
                                          op->name,
                                          op->args,
                                          op->call_type,
                                          src_op);
                }
        }  // if (op->func == src_op)
        else
        {
                return expr;
        }
}  // BodyStmtAutoInliner::Mutate_


void TensorAutoInliner::VisitPostOrder(const Tensor & tensor)
{
        if (std::find(_tensor_post_order.begin(),
                      _tensor_post_order.end(), tensor) !=
            _tensor_post_order.end())
        {
                return;
        }
        if (const ComputeOpNode * compute_op =
            tensor->op.as < ComputeOpNode > ()) 
        {
                for (const Tensor & input_tensor :
                     compute_op->InputTensors())
                {
                        VisitPostOrder(input_tensor);
                        _tensor_reverse_map[input_tensor].insert(tensor);
                }
                _tensor_post_order.push_back(tensor);
        }
}  // TensorAutoInliner::VisitPostOrder


Array < Tensor >
TensorAutoInliner::Mutate(const Array < Tensor > & tensors)
{
        for (const Tensor & tensor : tensors) { VisitPostOrder(tensor); }

        for (const Tensor & itensor : _tensor_post_order)
        {
                for (const Tensor & otensor :
                     _tensor_reverse_map[itensor])
                {
                        const ComputeOpNode
                                * const icompute_op = itensor->op.as < ComputeOpNode > (),
                                * const ocompute_op = otensor->op.as < ComputeOpNode > ();
                        CHECK(icompute_op != nullptr &&
                              ocompute_op != nullptr);

                        Array < Var > iaxis_vars;
                        for (const IterVar & iv : icompute_op->axis)
                        {
                                iaxis_vars.push_back(iv->var);
                        }
                        auto GetBodyStmt = [this](const Tensor & tensor) -> Expr
                                {
                                        CHECK(tensor->value_index == 0);
                                        auto iter = this->_tensor_compute_op_map.find(tensor);
                                        const Operation & op =
                                                iter != this->_tensor_compute_op_map.end() ? 
                                                _tensor_compute_op_map[tensor] : tensor->op;
                                        const ComputeOpNode * const compute_op = op.as < ComputeOpNode > ();
                                        CHECK(compute_op != nullptr);
                                        return compute_op->body[0];
                                };
                        // BodyStmtAutoInliner inliner = {
                        //         .src_op = itensor->op,
                        //         .src_axis_vars = iaxis_vars,
                        //         .src_body_stmt = GetBodyStmt(itensor)};
                        BodyStmtAutoInliner inliner;
                        inliner.src_op = itensor->op;
                        inliner.src_axis_vars = iaxis_vars;
                        inliner.src_body_stmt = GetBodyStmt(itensor);
                        
                }  // for (otensor ∈ _tensor_reverse_map[itensor])
        }  // for (itensor ∈ _tensor_post_order)
}  // TensorAutoInliner::Mutate


}  // namespace ir
}  // namespace tvm

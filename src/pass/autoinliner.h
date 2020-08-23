#pragma once

#include <unordered_map>
#include <vector>

#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/operation.h>


namespace tvm {
namespace ir {


/// @brief  @c BodyStmtAutoInliner automatically inlines the body statement
///         of source into the target.
struct BodyStmtAutoInliner : public IRMutator
{
        Operation src_op;
        Array < Var > src_axis; 
        Expr src_body_stmt;

        Expr Mutate_(const Call * op, const Expr & e) override final;

};  // class BodyStmtAutoInliner


/// @brief  @c TensorAutoInliner automatically inlines injective tensor
///         expressions into their respective consumers.     
class TensorAutoInliner
{
private:
        std::vector < Tensor > _tensor_post_order;
        std::unordered_map < Tensor,
                             std::unordered_set < Tensor > >
                _tensor_reverse_map;
        std::unordered_map < Tensor, Operation >
                _tensor_body_stmt_map;

        /// @brief Make a post-order walk through of the tensors to initialize 
        ///        @c tensor_post_order and @c tensor_reverse_map .
        void InitPostOrder();
public:
        void Mutate(Tensor * const ptensors);
};  // class TensorAutoInliner


}  // namespace ir
}  // namespace tvm

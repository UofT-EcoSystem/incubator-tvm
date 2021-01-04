/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file cse.cc
 * \brief Common subexpression elimination
 */
#include <tvm/arith/analyzer.h>
#include <tvm/node/functor.h>
#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>

#include <tuple>
#include <functional>

#include "ad_utils.h"

namespace tvm {
namespace te {
namespace {


class CSEOptimizer;  // Forward Declaration

using ComputeOpAxis = std::pair<const ComputeOpNode*, size_t>;

class TensorExprNode
    : public ExprFunctor<bool(const PrimExpr&, const TensorExprNode&)> {
 public:
  /*!
   * \brief  Compare two tensor expressions. 
   * \return true if the tensor expressions are deemed equal, false otherwise
   */
  bool Equal(const TensorExprNode& other);

  TensorExprNode() = default;
  explicit TensorExprNode(const PrimExpr& expr) : expr_(expr) {}
  friend class CSEOptimizer;
 private:
  virtual bool VisitExpr_(const VarNode* op,
                          const TensorExprNode& other) override final;
  // The ProducerLoadNode's will be handled by the high-level Compare method.
  virtual bool VisitExpr_(const CallNode* op,
                          const TensorExprNode& other) override final;
  virtual bool VisitExpr_(const AddNode* op,
                          const TensorExprNode& other) override final;
  virtual bool VisitExpr_(const SubNode* op,
                          const TensorExprNode& other) override final;
  virtual bool VisitExpr_(const MulNode* op,
                          const TensorExprNode& other) override final;
  virtual bool VisitExpr_(const DivNode* op,
                          const TensorExprNode& other) override final;
  virtual bool VisitExpr_(const ReduceNode* op,
                          const TensorExprNode& other) override final;
  virtual bool VisitExpr_(const IntImmNode* op,
                          const TensorExprNode& other) override final;
  virtual bool VisitExpr_(const FloatImmNode* op,
                          const TensorExprNode& other) override final;
  /*! \brief Auxiliary methods for comparing between reducer and placeholder nodes.
   */
  static bool Compare_(const CommReducer& lhs, const CommReducer& rhs);

  PrimExpr expr_;
};


/*!
 * \brief The \p CSEOptimizer eliminates the common subexpressions between the
 *        source and target tensor.
 * \note  The \p CSEOptimizer inherits from the \p ExprFunctor interface twice.
 *        The former is responsible for optimizing the target expression while
 *        the latter searching the target within the source.
 */
class CSEOptimizer
    : public ExprFunctor<PrimExpr(const PrimExpr&)>,
      public ExprFunctor<void(const PrimExpr&, PrimExpr* const)> {
 public:
  /*!
   * \brief  Perform *inplace* CSE optimization on the \p tgt tensor.
   * \note   By *inplace*, we mean that the optimized \p tgt tensor will
   *         directly leverage the common subexpression in the form of a
   *         \p ProducerLoadNode , whereas in the non-inplace version, in the
   *         form of a \p PlaceholderOpNode .
   * \return optimized \p src and \p tgt tensor pair
   */
  std::pair<Tensor, Tensor> OptimizeInplace(const Tensor& tgt);
  /*!
   * \brief  Perform CSE optimization on the \p tgt tensor.
   * \return optimized \p src and \p tgt tensor pair. The former outputs feature
   *         maps which will serve as the \p PlaceholderOpNode 's of the latter.
   *         Note that \p src might also be part of the feature maps.
   */
  std::tuple<Tensor, std::vector<Tensor>, Tensor>
  Optimize(const Tensor& tgt);

  explicit CSEOptimizer(const Tensor& src) : src_(src) {}
 private:
#define DECLARE_OP_SEARCH_AND_OPTMIZE(Op)            \
  PrimExpr VisitExpr_(const Op* op) override final;  \
  void VisitExpr_(const Op* op, PrimExpr* const) override final;

  DECLARE_OP_SEARCH_AND_OPTMIZE(ProducerLoadNode)
  DECLARE_OP_SEARCH_AND_OPTMIZE(CallNode)
  DECLARE_OP_SEARCH_AND_OPTMIZE(AddNode)
  DECLARE_OP_SEARCH_AND_OPTMIZE(SubNode)
  DECLARE_OP_SEARCH_AND_OPTMIZE(MulNode)
  DECLARE_OP_SEARCH_AND_OPTMIZE(DivNode)
  DECLARE_OP_SEARCH_AND_OPTMIZE(ReduceNode)
  /*!
   * \brief  Optimize the target expression.
   * \return two expressions correspond respectively to the optimized source and
   *         target expression
   */
  std::pair<PrimExpr, PrimExpr> Optimize(const PrimExpr& tgt);
  /*!
   * \brief  Locate the target tensor expression within \p src .
   * \return true if the \p tgt tensor expression has been found, along with the
   *         located expression, false otherwise
   */
  PrimExpr Search(const PrimExpr& tgt);
  /*! \brief Infer the extent (i.e., shape) of an experssion.
   *  \sa    WrapProducerLoad
   */
  std::vector<PrimExpr> InferExtents(const PrimExpr& expr);
  /*! \brief Wrap the expression into a \p ProducerLoadNode .
   */
  ProducerLoad WrapProducerLoad(const PrimExpr& expr);

  Tensor src_;
  // internal variables
  TensorExprNode tgt_expr_;
  bool optimize_inplace_;
  unsigned feature_maps_cnt_ = 0;
};  // class CSEOptimizer


}  // namespace


GradientResult CSE(const Tensor& output, const std::vector<Tensor>& input_grads) {
  Tensor opted_output = output;
  std::vector<Tensor> stashed_feature_maps, opted_input_grads;
  // 1. Remove the common subexpressions between the input gradients.
  // 2. Remove the common subexpressions between the input gradients and output.
  //    This is in essence infering the backward dependency.
  for (const Tensor& input_grad : input_grads) {
    // optimized output with common subexpressions
    std::vector<Tensor> requested_feature_maps;
    Tensor opted_input_grad;
    // leverage the output to eliminate the common subexpression of the input gradient
    std::tie(opted_output,
             requested_feature_maps,
             opted_input_grad)
        = CSEOptimizer(opted_output).Optimize(input_grad);
    for (const Tensor& requested_feature_map : requested_feature_maps) {
      // stash the feature map if this has not been done before
      bool feature_map_already_stashed = false;
      for (const Tensor& stashed_fm : stashed_feature_maps) {
        if (stashed_fm.same_as(requested_feature_map)) {
          feature_map_already_stashed = true;
        }
      }
      if (!feature_map_already_stashed) {
        stashed_feature_maps.push_back(requested_feature_map);
      }
    }  // for (cs ∈ opted_output_w_css.second)
  }  // for (input_grad ∈ input_grads)
  return GradientResult(opted_output, stashed_feature_maps, opted_input_grads);
}


/*******************************************************************************
 * TensorExprTree/Node
 *******************************************************************************/
bool
TensorExprNode::VisitExpr_(
    const VarNode* op,
    const TensorExprNode& other) {
  const VarNode* const other_op = other.expr_.as<VarNode>();
  CHECK(other_op != nullptr);
  return true;  // always assume that IterVar's are equal
}

bool
TensorExprNode::VisitExpr_(
    const CallNode* op,
    const TensorExprNode& other) {
  const CallNode* const other_op = other.expr_.as<CallNode>();
  CHECK(other_op != nullptr);
  if (!op->op.same_as(other_op->op)) {
    return false;
  }
  if (op->args.size() != other_op->args.size()) {
    return false;
  }
  for (size_t i = 0; i < op->args.size(); ++i) {
    if (!TensorExprNode(op->args[i]).Equal(
         TensorExprNode(other_op->args[i]))) {
      return false;
    }
  }
  return true;
}

#define DEFINE_BINARY_OP_COMMUTATIVE_COMPARE(OpNodeType)            \
bool                                                                \
TensorExprNode::VisitExpr_(                                         \
    const OpNodeType* op,                                           \
    const TensorExprNode& other) {                                  \
  const OpNodeType* const other_op = other.expr_.as<OpNodeType>();  \
  CHECK(other_op != nullptr);                                       \
  bool cmp_result_aa = TensorExprNode(op->a).Equal(                 \
                       TensorExprNode(other_op->a)),                \
       cmp_result_bb = TensorExprNode(op->b).Equal(                 \
                       TensorExprNode(other_op->b));                \
  if (cmp_result_aa && cmp_result_bb) {                             \
    return true;                                                    \
  } else {                                                          \
    bool cmp_result_ab = TensorExprNode(op->a).Equal(               \
                         TensorExprNode(other_op->b)),              \
         cmp_result_ba = TensorExprNode(op->b).Equal(               \
                         TensorExprNode(other_op->a));              \
    if (cmp_result_ab && cmp_result_ba) {                           \
      return true;                                                  \
    }                                                               \
  }                                                                 \
  return false;                                                     \
}

#define DEFINE_BINARY_OP_NONCOMMUTATIVE_COMPARE(OpNodeType)         \
bool                                                                \
TensorExprNode::VisitExpr_(                                         \
    const OpNodeType* op,                                           \
    const TensorExprNode& other) {                                  \
  const OpNodeType* const other_op = other.expr_.as<OpNodeType>();  \
  CHECK(other_op != nullptr);                                       \
  if (!TensorExprNode(op->a).Equal(                                 \
       TensorExprNode(other_op->a))) {                              \
    return false;                                                   \
  }                                                                 \
  if (!TensorExprNode(op->b).Equal(                                 \
       TensorExprNode(other_op->b))) {                              \
    return false;                                                   \
  }                                                                 \
  return true;                                                      \
}

DEFINE_BINARY_OP_COMMUTATIVE_COMPARE(AddNode)
DEFINE_BINARY_OP_NONCOMMUTATIVE_COMPARE(SubNode)
DEFINE_BINARY_OP_COMMUTATIVE_COMPARE(MulNode)
DEFINE_BINARY_OP_NONCOMMUTATIVE_COMPARE(DivNode)

bool
TensorExprNode::Compare_(
    const CommReducer& lhs, const CommReducer& rhs) {
  if (lhs->result.size() == 0 ||
      lhs->result.size() != rhs->result.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs->result.size(); ++i) {
    if (!TensorExprNode(lhs->result[i]).Equal(
         TensorExprNode(rhs->result[i]))) {
      return false;
    }
  }
  return true;
}

bool
TensorExprNode::VisitExpr_(
    const ReduceNode* op,
    const TensorExprNode& other) {
  const ReduceNode* const other_op = other.expr_.as<ReduceNode>();
  CHECK(other_op != nullptr);
  if (op->value_index != 0 ||
      other_op->value_index != 0) {
    LOG(WARNING) << "Have not handled ReduceNode's whose value index is not 0";
    return false;
  }
  // We do NOT check the reduction axes here because they will be checked by operator==.
  bool is_same_combiner = Compare_(op->combiner, other_op->combiner),
       is_same_source = TensorExprNode(op->source[op->value_index]).Equal(
                        TensorExprNode(other_op->source[other_op->value_index])),
       is_same_condition = TensorExprNode(op->condition).Equal(
                           TensorExprNode(other_op->condition));
  if (is_same_combiner && is_same_source &&
      is_same_condition) {
    return true;
  } else {
    return false;
  }
}

#define DEFINE_IMM_COMPARE(OpNodeType)                              \
ConditionalBool                                                     \
TensorExprNode::VisitExpr_(                                         \
    const OpNodeType* op,                                           \
    const TensorExprNode& other) {                                  \
  const OpNodeType* const other_op = other.expr_.as<OpNodeType>();  \
  CHECK(other_op != nullptr);                                       \
  if (op->value == other_op->value) {                               \
    return ConditionalBool(true);                                   \
  } else {                                                          \
    return ConditionalBool(false);                                  \
  }                                                                 \
}

DEFINE_IMM_COMPARE(IntImmNode)
DEFINE_IMM_COMPARE(FloatImmNode)

ConditionalBool
TensorExprNode::Compare(const TensorExprNode& other) {
  if (!expr_.defined() ||
      !other.expr_.defined()) {
    return ConditionalBool(false);
  }
  if (expr_.get() == other.expr_.get()) {
    return ConditionalBool(true);
  }
  // If any of the LHS and/or RHS are ProducerLoadNode's, unpack them to
  // obtain the compute operation or the placeholder.
  if (const ProducerLoadNode* const
      op = expr_.as<ProducerLoadNode>()) {
    Tensor tensor = Downcast<Tensor>(op->producer);
    CHECK(tensor.defined());
    if (tensor->op->IsInstance<ComputeOpNode>()) {
      ComputeOp compute_op = Downcast<ComputeOp>(tensor->op);
      CHECK(compute_op->axis.size() ==
            op->indices.size());
      // Traverse through the compute axes. If the compute axes are kDataPar,
      // inline it directly by doing substitution, otherwise record them in the
      // <Var, ComputeOpAxis> mapping.
      Map<Var, PrimExpr> vmap;
      for (size_t i = 0; i < lhs_axis_.size(); ++i) {
        vmap.Set(lhs_axis_[i].first, lhs_axis_[i].second);
      }
      for (size_t i = 0; i < compute_op->axis.size(); ++i) {
        Var compute_op_axis_var = compute_op->axis[i]->var;
        if (compute_op->axis[i]->iter_type == kDataPar) {
          lhs_axis_.Set(
              i, std::make_pair(compute_op_axis_var,
                                analyzer_.Simplify(Substitute(op->indices[i], vmap))));
        } else {
          lhs_axis_.Set(
              i, std::make_pair(compute_op_axis_var, compute_op_axis_var));
          lhs_var_comp_op_axis_map_[compute_op_axis_var] = std::make_pair(compute_op.get(), i);
        }
      }  // for (i ∈ range(0, compute_op->axis.size()))
      return VisitExpr(compute_op->body[tensor->value_index], other);
    } else if (const PlaceholderOpNode* ph_op_node =
               tensor->op.as<PlaceholderOpNode>()) {
      // If LHS is a placeholder, we compare the LHS and RHS axes.
      if (const ProducerLoadNode* const
          other_op = other.expr_.as<ProducerLoadNode>()) {
        Tensor other_tensor = Downcast<Tensor>(other_op->producer);
        if (const PlaceholderOpNode* other_ph_op_node =
            other_tensor->op.as<PlaceholderOpNode>()) {
          if (ph_op_node == other_ph_op_node) {
            CHECK(op->indices.size() == other_op->indices.size())
                << "LHS and RHS are accessing the same tensor but have different sizes"
                << "(" << op->indices.size() << " vs. "
                       << other_op->indices.size() << ")";
            Map<Var, PrimExpr> vmap, other_vmap;
            for (size_t i = 0; i < lhs_axis_.size(); ++i) {
              vmap.Set(lhs_axis_[i].first, lhs_axis_[i].second);
            }
            for (size_t i = 0; i < rhs_axis_.size(); ++i) {
              other_vmap.Set(rhs_axis_[i].first, rhs_axis_[i].second);
            }
            VarMap var_map;
            for (size_t i = 0; i < op->indices.size(); ++i) {
              PrimExpr
                  lhs_index = analyzer_.Simplify(Substitute(op->indices[i], vmap)),
                  rhs_index = analyzer_.Simplify(Substitute(other_op->indices[i], other_vmap));
              RETURN_IF_FALSE_ELSE_UPDATE_VARMAP(
                  TensorExprNode(lhs_index).Compare(TensorExprNode(rhs_index)), var_map);
            }
            return ConditionalBool(true, var_map);
          }
        }
      }  // if (other_op = other.expr_.as<ProducerLoadNode>())
      return ConditionalBool(false);
    } else {
      LOG(WARNING) << "Unhandled tensor OpType: " << tensor->op;
      return ConditionalBool(false);
    }  // if (tensor->op->IsInstance<ComputeOpNode>())
  }  // if (op = expr_.as<ProducerLoadNode>())
  // repeat the same thing for the other tensor expression
  if (const ProducerLoadNode* const
      op = other.expr_.as<ProducerLoadNode>()) {
    Tensor tensor = Downcast<Tensor>(op->producer);
    if (tensor->op->IsInstance<ComputeOpNode>()) {
      ComputeOp compute_op = Downcast<ComputeOp>(tensor->op);
      // ditto
      Map<Var, PrimExpr> vmap;
      for (size_t i = 0; i < rhs_axis_.size(); ++i) {
        vmap.Set(rhs_axis_[i].first, rhs_axis_[i].second);
      }
      for (size_t i = 0; i < compute_op->axis.size(); ++i) {
        Var compute_op_axis_var = compute_op->axis[i]->var;
        if (compute_op->axis[i]->iter_type == kDataPar) {
          rhs_axis_.Set(
              i, std::make_pair(compute_op_axis_var,
                                analyzer_.Simplify(Substitute(op->indices[i], vmap))));
        } else {
          rhs_axis_.Set(
              i, std::make_pair(compute_op_axis_var, compute_op_axis_var));
          rhs_var_comp_op_axis_map_[compute_op_axis_var] = std::make_pair(compute_op.get(), i);
        }
      }  // for (i ∈ range(0, compute_op->axis.size()))
      return VisitExpr(expr_, TensorExprNode(compute_op->body[tensor->value_index]));
    } else {
      LOG(WARNING) << "Unhandled tensor OpType: " << tensor->op;
      return ConditionalBool(false);
    }  // if (tensor->op->IsInstance<ComputeOpNode>())
  }  // if (op = other.expr_.as<ProducerLoadNode>())
  // directly return false if the type index does not match
  if (expr_->type_index() !=
      other.expr_->type_index()) {
    return ConditionalBool(false);
  }
  return VisitExpr(expr_, other);
}

std::unordered_map<Var, ComputeOpAxis, ObjectPtrHash, ObjectPtrEqual>
    TensorExprNode::lhs_var_comp_op_axis_map_,
    TensorExprNode::rhs_var_comp_op_axis_map_;
Array<std::pair<Var, PrimExpr>> TensorExprNode::lhs_axis_;
Array<std::pair<Var, PrimExpr>> TensorExprNode::rhs_axis_;

/*******************************************************************************
 * CSE Optimizer
 *******************************************************************************/
std::pair<Tensor, Tensor>
CSEOptimizer::OptimizeInplace(const Tensor& tgt) {
  return std::make_pair(src_, tgt);
}

std::tuple<Tensor, std::vector<Tensor>, Tensor>
CSEOptimizer::Optimize(const Tensor& tgt) {
  return std::make_tuple(src_, std::vector<Tensor>{}, tgt);
}

ProducerLoad
CSEOptimizer::WrapProducerLoad(const PrimExpr& expr) {
  std::vector<PrimExpr> extents = InferExtents(expr);
  if (expr->IsInstance<ProducerLoad>()) {
    LOG(INFO) << expr << " is already in the format of a ProducerLoadNode";
    return Downcast<ProducerLoad>(expr);
  }
  std::vector<IterVar> axis;
  std::vector<PrimExpr> indices;
  for (const PrimExpr& extent : extents) {
    IterVar iv = IterVar(Range::FromMinExtent(PrimExpr(0), extent),
                         Var(),
                         kDataPar);
    axis.push_back(iv);
    indices.push_back(iv->var);
  }
  return ProducerLoad(Tensor(extents, expr.dtype(),
                             ComputeOp("feature_maps_" +
                                       std::to_string(feature_maps_cnt_++),
                                       "feature_maps",
                                       {}, axis, {expr}),
                             0),
                      indices);
}

std::pair<PrimExpr, PrimExpr>
CSEOptimizer::Optimize(const PrimExpr& tgt) {
  PrimExpr src = Search(tgt);
  if (!src.defined()) {
    return std::make_pair(PrimExpr(), tgt);
  }
  // If the optimization is inplace, then the target expression will be
  // optimized to a ProducerLoadNode. Otherwise, a PlaceholderOpNode. In either
  // case, a wrapper is needed on top of the src tensor expression.
  ProducerLoad wrapped_src = WrapProducerLoad(src);
  if (optimize_inplace_) {
    return std::make_pair(wrapped_src, wrapped_src);
  } else {
    Tensor tensor = Downcast<Tensor>(wrapped_src->producer);
    return std::make_pair(
        wrapped_src,
        ProducerLoad(Tensor(tensor->shape, tensor->dtype,
                            PlaceholderOp(tensor->op->name,
                                          tensor->shape, tensor->dtype),
                            0),
                     indices));
  }
}

}  // namespace te
}  // namespace tvm

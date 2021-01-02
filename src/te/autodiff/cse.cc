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
#include <tvm/node/functor.h>
#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>

#include <tuple>

#include "ad_utils.h"

namespace tvm {
namespace te {
namespace {


// Auxiliary class for comparing between tensor expressions.
class VarMap : public Map<Var, Var> {
  /*!
   *  \brief  Update this VarMap with all the entries from another VarMap.
   *  \return true if the insertion is successful, and there is no conflict
   *          between the two mappings, false otherwise
   */
 public:
  VarMap() : Map<Var, Var>() {}
  VarMap(std::initializer_list<std::pair<Var, Var>> init) : Map<Var, Var>(init) {}
  bool Update(const VarMap& other);
};  // class VarMap

/*!
 * \brief ConditionalBool is the return type of the compare method. It is a
 *        pair consists of a boolean variable and a variable mapping. While
 *        the former represents whether two tensor expressions MIGHT be
 *        equivalent or not, the latter represents the variable mapping that
 *        needs to satisfied for the equivalence relation.
 */
class ConditionalBool : public std::pair<bool, VarMap> {
 public:
  ConditionalBool() : std::pair<bool, VarMap>(false, {}) {}
  explicit ConditionalBool(const bool is_same)
      : std::pair<bool, VarMap>(is_same, {}) {}
  ConditionalBool(const bool is_same, VarMap var_map)
      : std::pair<bool, VarMap>(is_same, var_map) {}
  operator bool() { return first; }
};  // class ConditionalBool

class CSEOptimizer;  // Forward Declaration

using ComputeOpAxis = std::pair<const ComputeOpNode*, size_t>;

class TensorExprNode
    : public ExprFunctor<ConditionalBool(const PrimExpr&, const TensorExprNode&)> {
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
  using ExprFunctor<ConditionalBool(const PrimExpr&, const TensorExprNode&)>::VisitExpr;
  virtual ConditionalBool VisitExpr_(const VarNode* op,
                                     const TensorExprNode& other) override final;
  // The `ProducerLoadNode`'s will be handled by the high-level `Compare` method.
  virtual ConditionalBool VisitExpr_(const CallNode* op,
                                     const TensorExprNode& other) override final;
  virtual ConditionalBool VisitExpr_(const AddNode* op,
                                     const TensorExprNode& other) override final;
  virtual ConditionalBool VisitExpr_(const SubNode* op,
                                     const TensorExprNode& other) override final;
  virtual ConditionalBool VisitExpr_(const MulNode* op,
                                     const TensorExprNode& other) override final;
  virtual ConditionalBool VisitExpr_(const DivNode* op,
                                     const TensorExprNode& other) override final;
  virtual ConditionalBool VisitExpr_(const ReduceNode* op,
                                     const TensorExprNode& other) override final;
  virtual ConditionalBool VisitExpr_(const IntImmNode* op,
                                     const TensorExprNode& other) override final;
  virtual ConditionalBool VisitExpr_(const FloatImmNode* op,
                                     const TensorExprNode& other) override final;
  /*!
   * \brief  Compare two tensor expressions (internal use).
   * \return true if the tensor expressions MIGHT be equal, with the second item
   *         denotes the var mapping that has to be met for the equality, false otherwise
   */
  ConditionalBool Compare(const TensorExprNode& other);
  /*! \brief Auxiliary methods for comparing between reducer and placeholder nodes.
   */
  static ConditionalBool Compare_(const CommReducer& lhs, const CommReducer& rhs);
  PrimExpr expr_;
  // mapping from variable to axis of ComputeOp's
  static std::unordered_map<Var, ComputeOpAxis, ObjectPtrHash, ObjectPtrEqual>
      src_var_compute_op_axis_map,
      tgt_var_compute_op_axis_map;
  static Array<PrimExpr> src_axis, tgt_axis;
};


/*!
 * \brief The \p CSEOptimizer eliminates the common subexpressions between the
 *        source and target tensor.
 */
class CSEOptimizer : public ExprFunctor<bool(const PrimExpr&)> {
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
  bool VisitExpr_(const CallNode* op) override final;
  bool VisitExpr_(const AddNode* op) override final;
  bool VisitExpr_(const SubNode* op) override final;
  bool VisitExpr_(const MulNode* op) override final;
  bool VisitExpr_(const DivNode* op) override final;
  bool VisitExpr_(const ReduceNode* op) override final;
  /*!
   * \brief  Locate the target tensor expression within \p src .
   * \return true if the \p tgt tensor expression has been found, along with the
   *         located expression, false otherwise
   */
  std::pair<bool, PrimExpr> Find(const PrimExpr& tgt);

  Tensor src_;
  TensorExprNode tgt_expr_;
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
std::unordered_map<Var, ComputeOpAxis, ObjectPtrHash, ObjectPtrEqual>
    TensorExprNode::src_var_compute_op_axis_map,
    TensorExprNode::tgt_var_compute_op_axis_map;
Array<PrimExpr> TensorExprNode::src_axis;
Array<PrimExpr> TensorExprNode::tgt_axis;

bool VarMap::Update(
    const VarMap& other) {
  for (const std::pair<Var, Var>& var_pair : other) {
    iterator iter = find(var_pair.first);
    if (iter != end()) {
      if (!(*iter).second.same_as(var_pair.second)) {
        // In the case when previously created mapping contradicts with the
        // current one, return false and skip the insertion.
        return false;
      }
    } else {  // if (iter == this->end())
      // Create the variable mapping if it has not been created before.
      Set(var_pair.first, var_pair.second);
    }
  }  // for (var_pair ∈ other)
  return true;
}

#define RETURN_IF_FALSE_ELSE_UPDATE_VARMAP(cmp, var_map)  \
  do {                                                    \
    ConditionalBool cmp_result = (cmp);                   \
    if (!cmp_result) {                                    \
      return ConditionalBool(false);                      \
    } else {                                              \
      if (!var_map.Update(cmp_result.second)) {           \
        return ConditionalBool(false);                    \
      }                                                   \
    }                                                     \
  } while (0);


bool
TensorExprNode::Equal(const TensorExprNode& other) {
  src_var_compute_op_axis_map.clear();
  tgt_var_compute_op_axis_map.clear();
  src_axis.clear();
  tgt_axis.clear();
  ConditionalBool cmp_result = Compare(other);
  if (!cmp_result.first) {
    return false;
  }
  // Check whether two variable mappings conflict in terms of ComputeOpNode's.
  for (auto var_map_iter_i = cmp_result.second.begin();
       var_map_iter_i != cmp_result.second.end(); ++var_map_iter_i) {
    for (auto var_map_iter_j = var_map_iter_i;
         var_map_iter_j != cmp_result.second.end(); ++var_map_iter_j) {
      const std::pair<Var, Var>& vpair_i = *var_map_iter_i, & vpair_j = *var_map_iter_j;
      const ComputeOpAxis
          & src_compute_op_axis_i = src_var_compute_op_axis_map[vpair_i.first],
          & tgt_compute_op_axis_i = tgt_var_compute_op_axis_map[vpair_i.second],
          & src_compute_op_axis_j = src_var_compute_op_axis_map[vpair_j.first],
          & tgt_compute_op_axis_j = tgt_var_compute_op_axis_map[vpair_j.second];
      if (src_compute_op_axis_i.first == src_compute_op_axis_j.first) {
        if (tgt_compute_op_axis_i.first != tgt_compute_op_axis_j.first) {
          return false;
        }
        if (src_compute_op_axis_i.second == src_compute_op_axis_j.second) {
          if (tgt_compute_op_axis_i.second != tgt_compute_op_axis_j.second) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

ConditionalBool
TensorExprNode::VisitExpr_(
    const VarNode* op,
    const TensorExprNode& other) {
  const VarNode* const other_op = other.expr_.as<VarNode>();
  CHECK(other_op != nullptr);
  VarMap var_map{
        {GetRef<Var>(op),
         GetRef<Var>(other_op)}
      };
  return ConditionalBool(true, var_map);
}

ConditionalBool
TensorExprNode::VisitExpr_(
    const CallNode* op,
    const TensorExprNode& other) {
  const CallNode* const other_op = other.expr_.as<CallNode>();
  CHECK(other_op != nullptr);
  if (!op->op.same_as(other_op->op)) {
    return ConditionalBool(false);
  }
  if (op->args.size() != other_op->args.size()) {
    return ConditionalBool(false);
  }
  VarMap var_map;
  for (size_t i = 0; i < op->args.size(); ++i) {
    RETURN_IF_FALSE_ELSE_UPDATE_VARMAP(
        TensorExprNode(op->args[i]).Compare(
        TensorExprNode(other_op->args[i])), var_map);
  }
  return ConditionalBool(true, var_map);
}

#define DEFINE_BINARY_OP_COMMUTATIVE_COMPARE(OpNodeType)            \
ConditionalBool                                                     \
TensorExprNode::VisitExpr_(                                         \
    const OpNodeType* op,                                           \
    const TensorExprNode& other) {                                  \
  const OpNodeType* const other_op = other.expr_.as<OpNodeType>();  \
  CHECK(other_op != nullptr);                                       \
  VarMap var_map;                                                   \
  ConditionalBool                                                   \
      cmp_result_aa                                                 \
          = TensorExprNode(op->a).Compare(                          \
            TensorExprNode(other_op->a)),                           \
      cmp_result_bb                                                 \
          = TensorExprNode(op->b).Compare(                          \
            TensorExprNode(other_op->b));                           \
  if (cmp_result_aa && cmp_result_bb) {                             \
    if (var_map.Update(cmp_result_aa.second) &&                     \
        var_map.Update(cmp_result_bb.second)) {                     \
      return ConditionalBool(true, var_map);                        \
    } else {                                                        \
      return ConditionalBool(false);                                \
    }                                                               \
  } else {                                                          \
    ConditionalBool                                                 \
        cmp_result_ab                                               \
            = TensorExprNode(op->a).Compare(                        \
              TensorExprNode(other_op->b)),                         \
        cmp_result_ba                                               \
            = TensorExprNode(op->b).Compare(                        \
              TensorExprNode(other_op->a));                         \
    if (cmp_result_ab && cmp_result_ba) {                           \
      if (var_map.Update(cmp_result_ab.second) &&                   \
          var_map.Update(cmp_result_ba.second)) {                   \
        return ConditionalBool(true, var_map);                      \
      } else {                                                      \
        return ConditionalBool(false);                              \
      }                                                             \
    }                                                               \
  }                                                                 \
  return ConditionalBool(false);                                    \
}

#define DEFINE_BINARY_OP_NONCOMMUTATIVE_COMPARE(OpNodeType)         \
ConditionalBool                                                     \
TensorExprNode::VisitExpr_(                                         \
    const OpNodeType* op,                                           \
    const TensorExprNode& other) {                                  \
  const OpNodeType* const other_op = other.expr_.as<OpNodeType>();  \
  CHECK(other_op != nullptr);                                       \
  VarMap var_map;                                                   \
  RETURN_IF_FALSE_ELSE_UPDATE_VARMAP(                               \
      TensorExprNode(op->a).Compare(                                \
      TensorExprNode(other_op->a)), var_map);                       \
  RETURN_IF_FALSE_ELSE_UPDATE_VARMAP(                               \
      TensorExprNode(op->b).Compare(                                \
      TensorExprNode(other_op->b)), var_map);                       \
  return ConditionalBool(true, var_map);                            \
}

DEFINE_BINARY_OP_COMMUTATIVE_COMPARE(AddNode)
DEFINE_BINARY_OP_NONCOMMUTATIVE_COMPARE(SubNode)
DEFINE_BINARY_OP_COMMUTATIVE_COMPARE(MulNode)
DEFINE_BINARY_OP_NONCOMMUTATIVE_COMPARE(DivNode)

ConditionalBool
TensorExprNode::Compare_(
    const CommReducer& lhs, const CommReducer& rhs) {
  if (lhs->result.size() == 0 ||
      lhs->result.size() != rhs->result.size()) {
    return ConditionalBool(false);
  }
  VarMap var_map;
  for (size_t i = 0; i < lhs->result.size(); ++i) {
    RETURN_IF_FALSE_ELSE_UPDATE_VARMAP(
        TensorExprNode(lhs->result[i]).Compare(
        TensorExprNode(rhs->result[i])),
        var_map);
  }
  return ConditionalBool(true, var_map);
}

ConditionalBool
TensorExprNode::VisitExpr_(
    const ReduceNode* op,
    const TensorExprNode& other) {
  const ReduceNode* const other_op = other.expr_.as<ReduceNode>();
  CHECK(other_op != nullptr);
  if (op->value_index != 0 ||
      other_op->value_index != 0) {
    LOG(WARNING) << "Have not handled ReduceNode's whose value index is not 0";
    return ConditionalBool(false);
  }
  // We do NOT check the reduction axes here because they will be checked by operator==.
  ConditionalBool
      is_same_combiner
        = Compare_(op->combiner, other_op->combiner),
      is_same_source
        = TensorExprNode(op->source[op->value_index]).Compare(
          TensorExprNode(other_op->source[other_op->value_index])),
      is_same_condition
        = TensorExprNode(op->condition).Compare(
          TensorExprNode(other_op->condition));
  if (is_same_combiner && is_same_source &&
      is_same_condition) {
    VarMap var_map;
    var_map.Update(is_same_combiner.second);
    var_map.Update(is_same_source.second);
    var_map.Update(is_same_condition.second);
    return ConditionalBool(true, var_map);
  } else {
    return ConditionalBool(false);
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
  // If any of the LHS and/or RHS are `ProducerLoadNode`'s, unpack them to
  // obtain the compute operation or the placeholder.
  if (const ProducerLoadNode* const
      op = expr_.as<ProducerLoadNode>()) {
    Tensor tensor = Downcast<Tensor>(op->producer);
    if (tensor->op->IsInstance<ComputeOpNode>()) {
      ComputeOp compute_op = Downcast<ComputeOp>(tensor->op);
      Map<Var, PrimExpr> vmap;
      // Traverse through the compute axes. If the compute axes are kDataPar,
      // inline it directly, otherwise record them in the <Var, ComputeOpAxis> mapping.
      for (size_t i = 0; i < compute_op->axis.size(); ++i) {
        if (compute_op->axis[i]->iter_type == kDataPar) {
          vmap.Set(compute_op->axis[i]->var,
                   op->indices[i]);
        } else {
          src_var_compute_op_axis_map[compute_op->axis[i]->var]
              = std::make_pair(compute_op.get(), i);
        }
      }  // for (i ∈ [0, compute_op->axis.size()))
      for (size_t i = 0; i < TensorExprNode::src_axis.size(); ++i) {
        src_axis.Set(
            i, Substitute(TensorExprNode::src_axis[i], vmap));
      }
      return VisitExpr(compute_op->body[tensor->value_index], other);
    } else if (const PlaceholderOpNode* ph_op_node =
               tensor->op.as<PlaceholderOpNode>()) {


      /// \todo Finish the comparator for `PlaceholderOpNode`'s.


      return ConditionalBool(false);
    } else {
      LOG(WARNING) << "Unhandled tensor OpType: " << tensor->op;
      return ConditionalBool(false);
    }  // if (tensor->op->IsInstance<ComputeOpNode>())
  }  // if (const ProducerLoadNode* const
     //     op = opref_.as<ProducerLoadNode>())
  // repeat the same thing for the other tensor expression
  if (const ProducerLoadNode* const
      op = other.expr_.as<ProducerLoadNode>()) {
    Tensor tensor = Downcast<Tensor>(op->producer);
    if (tensor->op->IsInstance<ComputeOpNode>()) {
      ComputeOp compute_op = Downcast<ComputeOp>(tensor->op);
      Map<Var, PrimExpr> vmap;
      for (size_t i = 0; i < compute_op->axis.size(); ++i) {
        if (compute_op->axis[i]->iter_type == kDataPar) {
          vmap.Set(compute_op->axis[i]->var,
                   op->indices[i]);
        } else {
          tgt_var_compute_op_axis_map[compute_op->axis[i]->var]
              = std::make_pair(compute_op.get(), i);
        }
      }  // for (i ∈ [0, compute_op->axis.size()))
      for (size_t i = 0; i < TensorExprNode::tgt_axis.size(); ++i) {
        tgt_axis.Set(
            i, Substitute(TensorExprNode::tgt_axis[i], vmap));
      }
      return VisitExpr(expr_, TensorExprNode(compute_op->body[tensor->value_index]));
    } else {
      LOG(WARNING) << "Unhandled tensor OpType: " << tensor->op;
      return ConditionalBool(false);
    }  // if (tensor->op->IsInstance<ComputeOpNode>())
  }  // if (const ProducerLoadNode* const
     //     op = other.opref_.as<ProducerLoadNode>())
  // directly return false if the type index does not match
  if (expr_->type_index() !=
      other.expr_->type_index()) {
    return ConditionalBool(false);
  }
  return VisitExpr(expr_, other);
}


/*******************************************************************************
 * CSE Optimizer
 *******************************************************************************/
bool CSEOptimizer::VisitExpr_(const CallNode* op) {
  return false;
}

bool CSEOptimizer::VisitExpr_(const AddNode* op) {
  return false;
}

std::pair<bool, PrimExpr>
CSEOptimizer::Find(const PrimExpr& tgt) {
  tgt_expr_.expr_ = tgt;

  return std::make_pair(false, PrimExpr());
}

std::pair<Tensor, Tensor>
CSEOptimizer::OptimizeInplace(const Tensor& tgt) {
  return std::make_pair(src_, tgt);
}

std::tuple<Tensor, std::vector<Tensor>, Tensor>
CSEOptimizer::Optimize(const Tensor& tgt) {
  return std::make_tuple(src_, std::vector<Tensor>{}, tgt);
}

}  // namespace te
}  // namespace tvm

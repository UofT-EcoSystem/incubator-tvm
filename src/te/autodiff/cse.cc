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
#include "ad_utils.h"

#include <sstream>

#include <tvm/node/functor.h>
#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>


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
  ConditionalBool(const bool is_same)
      : std::pair<bool, VarMap>(is_same, {}) {}
  ConditionalBool(const bool is_same, VarMap var_map)
      : std::pair<bool, VarMap>(is_same, var_map) {}
  operator bool() { return first; }
};  // class ConditionalBool

using ComputeOpAxis = std::pair<const ComputeOpNode*, size_t>;

struct TensorExprNode {
  using FCompare = NodeFunctor<
      ConditionalBool(const ObjectRef&, const TensorExprNode&,
                      const TensorExprNode* const)
      >;
  static FCompare& cmptable() {
    static FCompare inst;
    return inst;
  }
  ConditionalBool Compare_(const CallNode* const opnode,
                           const TensorExprNode& other) const;
  ConditionalBool Compare_(const PlaceholderOpNode* const opnode,
                           const TensorExprNode& other) const;
  ConditionalBool Compare_(const VarNode* const opnode,
                           const TensorExprNode& other) const;
  ConditionalBool Compare_(const AddNode* const opnode,
                           const TensorExprNode& other) const;
  ConditionalBool Compare_(const SubNode* const opnode,
                           const TensorExprNode& other) const;
  ConditionalBool Compare_(const MulNode* const opnode,
                           const TensorExprNode& other) const;
  ConditionalBool Compare_(const DivNode* const opnode,
                           const TensorExprNode& other) const;
  ConditionalBool Compare_(const CommReducerNode* const opnode,
                           const TensorExprNode& other) const;
  ConditionalBool Compare_(const ReduceNode* const opnode,
                           const TensorExprNode& other) const;
  ConditionalBool Compare_(const IntImmNode* const opnode,
                           const TensorExprNode& other) const;
  ConditionalBool Compare_(const FloatImmNode* const opnode,
                           const TensorExprNode& other) const;
  /*!
   * \brief  Compare two tensor expressions (internal use).
   * \return 
   */
  ConditionalBool Compare(const TensorExprNode& other) const;
  explicit TensorExprNode(const ObjectRef& opref_) : opref(opref_) {}
  /*!
   * \brief  Compare two tensor expressions. 
   * \return true if the tensor expressions are deemed equal, false otherwise
   */
  bool operator==(const TensorExprNode& other) const;
  ObjectRef opref;
  // mapping from variable to axis of ComputeOp's
  static std::unordered_map<
      Var, ComputeOpAxis,
      ObjectPtrHash, ObjectPtrEqual> lhs_var_compute_op_axis_map, rhs_var_compute_op_axis_map;
  static Array<PrimExpr> lhs_axis, rhs_axis;
};


/*!
 * \brief The \c CSEOptimizer eliminates the common subexpressions between the
 *        source and target tensor.
 */
struct CSEOptimizer {
  CSEOptimizer(const Tensor& src);
  using FOptimize = NodeFunctor<
      std::pair<PrimExpr, PrimExpr>(const ObjectRef&, const PrimExpr&,
                                    CSEOptimizer* const)
      >;
  static FOptimize& optable() {
    static FOptimize inst;
    return inst;
  }
  std::pair<PrimExpr, PrimExpr> Optimize_(const CallNode* const opnode,
                                          const PrimExpr& tgt_expr);
  std::pair<PrimExpr, PrimExpr> Optimize_(const AddNode* const opnode,
                                          const PrimExpr& tgt_expr);
  std::pair<PrimExpr, PrimExpr> Optimize_(const SubNode* const opnode,
                                          const PrimExpr& tgt_expr);
  std::pair<PrimExpr, PrimExpr> Optimize_(const MulNode* const opnode,
                                          const PrimExpr& tgt_expr);
  std::pair<PrimExpr, PrimExpr> Optimize_(const DivNode* const opnode,
                                          const PrimExpr& tgt_expr);
  std::pair<PrimExpr, PrimExpr> Optimize_(const ReduceNode* const opnode,
                                          const PrimExpr& tgt_expr);
  std::pair<PrimExpr, PrimExpr> Optimize_(const IntImmNode* const opnode,
                                          const PrimExpr& tgt_expr);
  std::pair<PrimExpr, PrimExpr> Optimize_(const FloatImmNode* const opnode,
                                          const PrimExpr& tgt_expr);
};  // class CSEOptimizer


}  // namespace anonymous


std::pair<Tensor, std::vector<Tensor> >
CSE(const Tensor& output, const std::vector<Tensor>& input_grads) {
  // 1. Apply auto-inliner to inline the injective operations. The point is to
  //    simplify the tensor expressions, and particularly tensor indices.
  
  // 2. Remove the common subexpresssions between the input gradients.
  for (const Tensor& input_grad : input_grads) {

  }
  // 3. Remove the common subexpressions between the input gradients and output.
  //    This is in essence infering the backward dependency.
  return std::make_pair(output, input_grads);
}


/*******************************************************************************
 * TensorExprTree/Node
 *******************************************************************************/
std::unordered_map<
    Var, ComputeOpAxis,
    ObjectPtrHash, ObjectPtrEqual> TensorExprNode::var_compute_op_axis_map;
Array<PrimExpr> TensorExprNode::lhs_axis;
Array<PrimExpr> TensorExprNode::rhs_axis;

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
      return false;                                       \
    } else {                                              \
      if (!var_map.Update(cmp_result.second)) {           \
        return false;                                     \
      }                                                   \
    }                                                     \
  } while (0);


ConditionalBool
TensorExprNode::Compare_(
    const CallNode* const opnode,
    const TensorExprNode& other) const {
  const CallNode* const other_opnode = other.opref.as<CallNode>();
  CHECK(other_opnode != nullptr);
  if (!opnode->op.same_as(other_opnode->op)) {
    return false;
  }
  if (opnode->args.size() != other_opnode->args.size()) {
    return false;
  }
  VarMap var_map;
  for (size_t i = 0; i < opnode->args.size(); ++i) {
    RETURN_IF_FALSE_ELSE_UPDATE_VARMAP(
        TensorExprNode(opnode->args[i]).Compare(
        TensorExprNode(other_opnode->args[i])), var_map);
  }
  return ConditionalBool(true, var_map);
}

ConditionalBool
TensorExprNode::Compare_(
    const PlaceholderOpNode* const opnode,
    const TensorExprNode& other) const {
  const PlaceholderOpNode* const other_opnode
      = other.opref.as<PlaceholderOpNode>();
  CHECK(other_opnode != nullptr);
  return opnode == other_opnode;
}

ConditionalBool
TensorExprNode::Compare_(
    const VarNode* const opnode,
    const TensorExprNode& other) const {
  const VarNode* const other_opnode = other.opref.as<VarNode>();
  CHECK(other_opnode != nullptr);
  VarMap var_map{
        {GetRef<Var>(opnode),
         GetRef<Var>(other_opnode)}
      };
  return ConditionalBool(true, var_map);
}

#define DEFINE_BINARY_OP_COMMUTATIVE_COMPARE(OpNodeType)                \
ConditionalBool                                                         \
TensorExprNode::Compare_(                                               \
    const OpNodeType* const opnode,                                     \
    const TensorExprNode& other) const {                                \
  const OpNodeType* const other_opnode = other.opref.as<OpNodeType>();  \
  CHECK(other_opnode != nullptr);                                       \
  VarMap var_map;                                                       \
  ConditionalBool                                                       \
      cmp_result_aa                                                     \
          = TensorExprNode(opnode->a).Compare(                          \
            TensorExprNode(other_opnode->a)),                           \
      cmp_result_bb                                                     \
          = TensorExprNode(opnode->b).Compare(                          \
            TensorExprNode(other_opnode->b));                           \
  if (cmp_result_aa && cmp_result_bb) {                                 \
    if (var_map.Update(cmp_result_aa.second) &&                         \
        var_map.Update(cmp_result_bb.second)) {                         \
      return ConditionalBool(true, var_map);                            \
    } else {                                                            \
      return false;                                                     \
    }                                                                   \
  } else {                                                              \
    ConditionalBool                                                     \
        cmp_result_ab                                                   \
            = TensorExprNode(opnode->a).Compare(                        \
              TensorExprNode(other_opnode->b)),                         \
        cmp_result_ba                                                   \
            = TensorExprNode(opnode->b).Compare(                        \
              TensorExprNode(other_opnode->a));                         \
    if (cmp_result_ab && cmp_result_ba) {                               \
      if (var_map.Update(cmp_result_ab.second) &&                       \
          var_map.Update(cmp_result_ba.second)) {                       \
        return ConditionalBool(true, var_map);                          \
      } else {                                                          \
        return false;                                                   \
      }                                                                 \
    }                                                                   \
  }                                                                     \
  return false;                                                         \
}

#define DEFINE_BINARY_OP_NONCOMMUTATIVE_COMPARE(OpNodeType)             \
ConditionalBool                                                         \
TensorExprNode::Compare_(                                               \
    const OpNodeType* const opnode,                                     \
    const TensorExprNode& other) const {                                \
  const OpNodeType* const other_opnode = other.opref.as<OpNodeType>();  \
  CHECK(other_opnode != nullptr);                                       \
  VarMap var_map;                                                       \
  RETURN_IF_FALSE_ELSE_UPDATE_VARMAP(                                   \
      TensorExprNode(opnode->a).Compare(                                \
      TensorExprNode(other_opnode->a)), var_map);                       \
  RETURN_IF_FALSE_ELSE_UPDATE_VARMAP(                                   \
      TensorExprNode(opnode->b).Compare(                                \
      TensorExprNode(other_opnode->b)), var_map);                       \
  return ConditionalBool(true, var_map);                                \
}

DEFINE_BINARY_OP_COMMUTATIVE_COMPARE(AddNode)
DEFINE_BINARY_OP_NONCOMMUTATIVE_COMPARE(SubNode)
DEFINE_BINARY_OP_COMMUTATIVE_COMPARE(MulNode)
DEFINE_BINARY_OP_NONCOMMUTATIVE_COMPARE(DivNode)

ConditionalBool
TensorExprNode::Compare_(
    const CommReducerNode* const opnode,
    const TensorExprNode& other) const {
  const CommReducerNode* const other_opnode
      = other.opref.as<CommReducerNode>();
  CHECK(other_opnode != nullptr);
  if (opnode->result.size() == 0 ||
      opnode->result.size() != other_opnode->result.size()) {
    return false;
  }
  VarMap var_map;
  for (size_t i = 0; i < opnode->result.size(); ++i) {
    RETURN_IF_FALSE_ELSE_UPDATE_VARMAP(
        TensorExprNode(opnode->result[i]).Compare(
        TensorExprNode(other_opnode->result[i])), var_map
        );
  }
}

ConditionalBool
TensorExprNode::Compare_(
    const ReduceNode* const opnode,
    const TensorExprNode& other) const {
  const ReduceNode* const other_opnode = other.opref.as<ReduceNode>();
  CHECK(other_opnode != nullptr);
  if (opnode->value_index != 0 ||
      other_opnode->value_index != 0) {
    LOG(WARNING) << "Have not handled non-trivial ReduceNode's";
    return false;
  }
  // We do NOT check the reduction axes here because they will be checked by operator==.
  ConditionalBool
      is_same_combiner
        = TensorExprNode(opnode->combiner).Compare(
          TensorExprNode(other_opnode->combiner)),
      is_same_source
        = TensorExprNode(opnode->source[opnode->value_index]).Compare(
          TensorExprNode(other_opnode->source[other_opnode->value_index])),
      is_same_condition
        = TensorExprNode(opnode->condition).Compare(
          TensorExprNode(other_opnode->condition));
}

#define DEFINE_IMM_COMPARE(OpNodeType)                                  \
ConditionalBool                                                         \
TensorExprNode::Compare_(                                               \
    const OpNodeType* const opnode,                                     \
    const TensorExprNode& other) const {                                \
  const OpNodeType* const other_opnode = other.opref.as<OpNodeType>();  \
  CHECK(other_opnode != nullptr);                                       \
  if (opnode->value == other_opnode->value) {                           \
    return true;                                                        \
  } else {                                                              \
    return false;                                                       \
  }                                                                     \
}

DEFINE_IMM_COMPARE(IntImmNode)
DEFINE_IMM_COMPARE(FloatImmNode)

ConditionalBool
TensorExprNode::Compare(const TensorExprNode& other) const {
  static const FCompare& fcompare = cmptable();
  if (opref.defined() &&
      other.opref.defined()) {
    if (const ProducerLoadNode* const
        opnode = opref.as<ProducerLoadNode>()) {
      Tensor tensor = Downcast<Tensor>(opnode->producer);
      if (tensor->op->IsInstance<ComputeOpNode>()) {
        ComputeOp compute_op = Downcast<ComputeOp>(tensor->op);
        Map<Var, PrimExpr> vmap;
        // Traverse through the compute axes. If the compute axes are kDataPar,
        // inline it directly, otherwise record them in the <Var, ComputeOpAxis> mapping.
        for (size_t i = 0; i < compute_op->axis.size(); ++i) {
          if (compute_op->axis[i]->iter_type == kDataPar) {
            vmap.Set(compute_op->axis[i]->var,
                     opnode->indices[i]);
          } else {
            lhs_var_compute_op_axis_map[compute_op->axis[i]->var]
                = std::make_pair(compute_op.get(), i);
          }
        }  // for (i ∈ [0, compute_op->axis.size()))
        for (size_t i = 0; i < TensorExprNode::lhs_axis.size(); ++i) {
          lhs_axis.Set(
              i, Substitute(TensorExprNode::lhs_axis[i], vmap));
        }
        return fcompare(compute_op->body[tensor->value_index],
                        other, this);
      } else if (tensor->op->IsInstance<PlaceholderOpNode>()) {
        return fcompare(tensor->op, other, this);
      } else {
        LOG(WARNING) << "Unhandled tensor OpType: " << tensor->op;
        return false;
      }
    }
    if (const ProducerLoadNode* const
        opnode = other.opref.as<ProducerLoadNode>()) {
      Tensor tensor = Downcast<Tensor>(opnode->producer);
      if (tensor->op->IsInstance<ComputeOpNode>()) {
        ComputeOp compute_op = Downcast<ComputeOp>(tensor->op);
        Map<Var, PrimExpr> vmap;
        for (size_t i = 0; i < compute_op->axis.size(); ++i) {
          // Ditto.
          if (compute_op->axis[i]->iter_type == kDataPar) {
            vmap.Set(compute_op->axis[i]->var,
                     opnode->indices[i]);
          } else {
            rhs_var_compute_op_axis_map[compute_op->axis[i]->var]
                = std::make_pair(compute_op.get(), i);
          }
        }  // for (i ∈ [0, compute_op->axis.size()))
        for (size_t i = 0; i < TensorExprNode::rhs_axis.size(); ++i) {
          rhs_axis.Set(
              i, Substitute(TensorExprNode::rhs_axis[i], vmap));
        }
        return fcompare(opref, TensorExprNode(compute_op->body[tensor->value_index]), this);
      } else if (tensor->op->IsInstance<PlaceholderOpNode>()) {
        return fcompare(opref, TensorExprNode(tensor->op), this);
      } else {
        LOG(WARNING) << "Unhandled tensor OpType: " << tensor->op;
        return false;
      }
    }
    return fcompare(opref, other, this);
  }  // if (opref.defined() && other.opref.defined())
}

#define DISPATCH_TO_CMP(Op)                                              \
set_dispatch<Op>([](const ObjectRef& opref, const TensorExprNode& other, \
                    const TensorExprNode* const pthis)                   \
                   -> ConditionalBool {                                  \
  if (opref->type_index() != other.opref->type_index()) {                \
    return false;                                                        \
  }                                                                      \
  return pthis->Compare_(static_cast<const Op*>(opref.get()), other);    \
})

TVM_STATIC_IR_FUNCTOR(TensorExprNode, cmptable)
.DISPATCH_TO_CMP(CallNode)
.DISPATCH_TO_CMP(PlaceholderOpNode)
.DISPATCH_TO_CMP(VarNode)
.DISPATCH_TO_CMP(AddNode)
.DISPATCH_TO_CMP(SubNode)
.DISPATCH_TO_CMP(MulNode)
.DISPATCH_TO_CMP(DivNode)
.DISPATCH_TO_CMP(CommReducerNode)
.DISPATCH_TO_CMP(ReduceNode)
.DISPATCH_TO_CMP(IntImmNode)
.DISPATCH_TO_CMP(FloatImmNode);

bool
TensorExprNode::operator==(const TensorExprNode& other) const {

}


}  // namespace te
}  // namespace tvm

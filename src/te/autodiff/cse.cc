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


namespace tvm {
namespace te {
namespace {


class TensorExprNode;
typedef std::shared_ptr<TensorExprNode> TensorExprPtr;

using ComputeOpAxis = std::pair<const ComputeOpNode*, size_t>;

class TensorExprNode {
 private:
  ObjectRef opref_;
  // mapping from variable to axis of ComputeOp's
  /// @note ComputeOpAxis has to inherit from ObjectRef.
  //  static Map<Var, ComputeOpAxis> var_compute_op_axis_map_;

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

  using FCompare = NodeFunctor<
      ConditionalBool(const ObjectRef&, const TensorExprNode&,
                      const TensorExprNode* const)
      >;
  static FCompare & cmptable() {
    static FCompare instance;
    return instance; 
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
 public:
  TensorExprNode(const ObjectRef& opref) : opref_(opref) {}
  /*!
   * \brief  Compare two tensor expressions. 
   * \return true if the tensor expressions are deemed equal, false otherwise
   */
  bool operator==(const TensorExprNode& other) const;
};


class CSEOptimizer;

/*!
 * \brief The \c CSEOptimizer eliminates the common subexpressions between the
 *        source and target tensor.
 */
class CSEOptimizer {
 public:
  CSEOptimizer(const Tensor& src);
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
bool TensorExprNode::VarMap::Update(
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


TensorExprNode::ConditionalBool
TensorExprNode::Compare_(
    const CallNode* const opnode,
    const TensorExprNode& other) const {
  const CallNode* const other_opnode = other.opref_.as<CallNode>();
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
        TensorExprNode(opnode->args[i]).Compare(other_opnode->args[i]), var_map);
  }
  return ConditionalBool(true, var_map);
}

TensorExprNode::ConditionalBool
TensorExprNode::Compare_(
    const PlaceholderOpNode* const opnode,
    const TensorExprNode& other) const {
  const PlaceholderOpNode* const other_opnode = other.opref_.as<PlaceholderOpNode>();
  CHECK(other_opnode != nullptr);
  return opnode == other_opnode;
}

TensorExprNode::ConditionalBool
TensorExprNode::Compare_(
    const VarNode* const opnode,
    const TensorExprNode& other) const {
  const VarNode* const other_opnode = other.opref_.as<VarNode>();
  CHECK(other_opnode != nullptr);
  VarMap var_map{
        {GetRef<Var>(opnode),
         GetRef<Var>(other_opnode)}
      };
  return ConditionalBool(true, var_map);
}

#define DEFINE_BINARY_OP_COMMUTATIVE_COMPARE(OpNodeType)                     \
TensorExprNode::ConditionalBool                                              \
TensorExprNode::Compare_(                                                    \
    const OpNodeType* const opnode,                                          \
    const TensorExprNode& other) const {                                     \
  const OpNodeType* const other_opnode = other.opref_.as<OpNodeType>();      \
  CHECK(other_opnode != nullptr);                                            \
  VarMap var_map;                                                            \
  ConditionalBool                                                            \
      cmp_result_aa = TensorExprNode(opnode->a).Compare(other_opnode->a),    \
      cmp_result_bb = TensorExprNode(opnode->b).Compare(other_opnode->b);    \
  if (cmp_result_aa && cmp_result_bb) {                                      \
    if (var_map.Update(cmp_result_aa.second) &&                              \
        var_map.Update(cmp_result_bb.second)) {                              \
      return ConditionalBool(true, var_map);                                 \
    } else {                                                                 \
      return false;                                                          \
    }                                                                        \
  } else {                                                                   \
    ConditionalBool                                                          \
        cmp_result_ab = TensorExprNode(opnode->a).Compare(other_opnode->b),  \
        cmp_result_ba = TensorExprNode(opnode->b).Compare(other_opnode->a);  \
    if (cmp_result_ab && cmp_result_ba) {                                    \
      if (var_map.Update(cmp_result_ab.second) &&                            \
          var_map.Update(cmp_result_ba.second)) {                            \
        return ConditionalBool(true, var_map);                               \
      } else {                                                               \
        return false;                                                        \
      }                                                                      \
    }                                                                        \
  }                                                                          \
  return false;                                                              \
}

#define DEFINE_BINARY_OP_NONCOMMUTATIVE_COMPARE(OpNodeType)              \
TensorExprNode::ConditionalBool                                          \
TensorExprNode::Compare_(                                                \
    const OpNodeType* const opnode,                                      \
    const TensorExprNode& other) const {                                 \
  const OpNodeType* const other_opnode = other.opref_.as<OpNodeType>();  \
  CHECK(other_opnode != nullptr);                                        \
  VarMap var_map;                                                        \
  RETURN_IF_FALSE_ELSE_UPDATE_VARMAP(                                    \
      TensorExprNode(opnode->a).Compare(other_opnode->a), var_map);      \
  RETURN_IF_FALSE_ELSE_UPDATE_VARMAP(                                    \
      TensorExprNode(opnode->b).Compare(other_opnode->b), var_map);      \
  return ConditionalBool(true, var_map);                                 \
}

DEFINE_BINARY_OP_COMMUTATIVE_COMPARE(AddNode)
DEFINE_BINARY_OP_NONCOMMUTATIVE_COMPARE(SubNode)
DEFINE_BINARY_OP_COMMUTATIVE_COMPARE(MulNode)
DEFINE_BINARY_OP_NONCOMMUTATIVE_COMPARE(DivNode)

TensorExprNode::ConditionalBool
TensorExprNode::Compare_(
    const CommReducerNode* const opnode,
    )

TensorExprNode::ConditionalBool
TensorExprNode::Compare_(
    const ReduceNode* const opnode,
    const TensorExprNode& other) const {
  const ReduceNode* const other_opnode = other.opref_.as<ReduceNode>();
  CHECK(other_opnode != nullptr);
  if (opnode->value_index != 0 ||
      other_opnode->value_index != 0) {
    LOG(WARNING) << "Have not handled non-trivial ReduceNode's";
    return false;
  }
  ConditionalBool
      is_same_combiner
        = ConditionalBool(opnode->combiner).Compare(other_opnode->combiner),
      is_same_condition
        = ConditionalBool(opnode->condition).Compare(other_opnode->condition);
}

#define DEFINE_IMM_COMPARE(OpNodeType)                                   \
TensorExprNode::ConditionalBool                                          \
TensorExprNode::Compare_(                                                \
    const OpNodeType* const opnode,                                      \
    const TensorExprNode& other) const {                                 \
  const OpNodeType* const other_opnode = other.opref_.as<OpNodeType>();  \
  CHECK(other_opnode != nullptr);                                        \
  if (opnode->value == other_opnode->value) {                            \
    return true;                                                         \
  } else {                                                               \
    return false;                                                        \
  }                                                                      \
}

DEFINE_IMM_COMPARE(IntImmNode)
DEFINE_IMM_COMPARE(FloatImmNode)


}  // namespace te
}  // namespace tvm

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
 public:
  ObjectRef opref;
  std::vector<TensorExprPtr> operands;
  // The indices are specially reserved for the comparison between placeholders.
  Array<PrimExpr> indices;
  // mapping from variable to axis of ComputeOp's
  Map<Var, ComputeOpAxis> var_compute_op_axis_map;

  /*!
   * \brief Convert a \c TensorExprNode to string.
   */
  std::string toString(const unsigned indent = 0) const {
    std::ostringstream strout;
    strout << "\n";
    for (unsigned i = 0; i < indent; ++i) {
      strout << " ";
    }
    for (const std::shared_ptr<TensorExprNode>&
         operand : operands){
      strout << operand->toString(indent + 2);
    }
    return strout.str();
  }

  // Auxiliary class for comparing between tensor expressions.
  class VarMap : public Map<Var, Var> {
    /*!
     *  \brief  Update this VarMap with all the entries from another VarMap.
     *  \return true if the insertion is successful, and there is no conflict
     *          between the two mappings, false otherwise
     */
   public:
    bool Update(const VarMap& other);
  };  // class VarMap
  /*!
   * \brief ConditionalBool
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
  ConditionalBool Compare_(const CallNode* const opnode, const TensorExprNode& other) const;
  ConditionalBool Compare_(const PlaceholderOpNode* const opnode,
                           const TensorExprNode& other) const;
  ConditionalBool Compare_(const AddNode* const opnode, const TensorExprNode& other) const;
  ConditionalBool Compare_(const SubNode* const opnode, const TensorExprNode& other) const;
  ConditionalBool Compare_(const MulNode* const opndoe, const TensorExprNode& other) const;
  ConditionalBool Compare_(const DivNode* const opnode, const TensorExprNode& other) const;
  ConditionalBool Compare_(const ReduceNode* const opnode, const TensorExprNode& other) const;
  ConditionalBool Compare_(const IntImmNode* const opnode, const TensorExprNode& other) const;
  ConditionalBool Compare_(const FloatImmNode* const opnode, const TensorExprNode& other) const;
  ConditionalBool Compare(const TensorExprNode& other) const;
  /*!
   * \brief Compare two tensor expression subtree.
   */
  bool operator==(const TensorExprNode& other) const;
};


class CSEOptimizer;

/*!
 * \brief The \c TensorExprTree constructs a tree-like structure from a tensor expression.
 */
class TensorExprTree {
 public:
  using FConstruct = NodeFunctor<void(const ObjectRef&, TensorExprNode* const,
                                      TensorExprTree* const)>;
  static FConstruct& cstrtable() {
    static FConstruct instance;
    return instance;
  }
  void Construct_(const CallNode* const, TensorExprNode* const);
  void Construct_(const ProducerLoadNode* const, TensorExprNode* const);
  void Construct_(const AddNode* const, TensorExprNode* const);
  void Construct_(const SubNode* const, TensorExprNode* const);
  void Construct_(const MulNode* const, TensorExprNode* const);
  void Construct_(const DivNode* const, TensorExprNode* const);
  void Construct_(const ReduceNode* const, TensorExprNode* const);
  void Construct_(const IntImmNode* const, TensorExprNode* const) {}
  void Construct_(const FloatImmNode* const, TensorExprNode* const) {}
  /*!
   * \brief 
   */
  TensorExprPtr Construct(const ObjectRef& ref, const Array<IterVar>& axis);
 private:
};  // class TensorExprConstr


/*!
 * \brief The \c CSEOptimizer eliminates the common subexpressions between the
 *        source and target tensor.
 */
class CSEOptimizer {
 public:
  CSEOptimizer(const Tensor& src);
 private:
  TensorExprTree src_tensor_expr_tree_,
                 tgt_tensor_expr_tree_;
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
    iterator iter = this->find(var_pair.first);
    if (iter != this->end()) {
      if (!(*iter).second.same_as(var_pair.second)) {
        // In the case when previously created mapping contradicts with the
        // current one, return false and skip the insertion.
        return false;
      }
    } else {  // if (iter == this->end())
      // Create the variable mapping if it has not been created before.
      this->Set(var_pair.first, var_pair.second);
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
      var_map.Update(cmp_result.second);                  \
    }                                                     \
  } while (0);


TensorExprNode::ConditionalBool
TensorExprNode::Compare_(
    const CallNode* const opnode,
    const TensorExprNode& other) const {
  const CallNode* const other_opnode = other.opref.as<CallNode>();
  CHECK(other_opnode != nullptr);
  if (!opnode->op.same_as(other_opnode->op)) {
    return false;
  }
  if (this->operands.size() != other.operands.size()) {
    return false;
  }
  VarMap var_map;
  for (size_t i = 0; i < this->operands.size(); ++i) {
    RETURN_IF_FALSE_ELSE_UPDATE_VARMAP(this->Compare(*(other.operands[i])), var_map);
  }
  return true;
}

TensorExprNode::ConditionalBool
TensorExprNode::Compare_(
    const PlaceholderOpNode* const opnode,
    const TensorExprNode& other) const {
  const PlaceholderOpNode* const other_opnode
      = other.opref.as<PlaceholderOpNode>();
  CHECK(other_opnode != nullptr);
  return opnode == other_opnode;
}

/*
#define DEFINE_BINARY_OP_COMMUTATIVE_COMPARE(OpNode)        \
bool TensorExprNode::Compare_(                              \
    const OpNode* const opnode,                             \
    const TensorExprNode& other) const {                    \
  CHECK(this->operands.size() == 2);                        \
  CHECK(other.operands.size() == 2);                        \
  return ((*this->operands[0]) == (*other.operands[0]) &&   \
          (*this->operands[1]) == (*other.operands[1])) ||  \
         ((*this->operands[0]) == (*other.operands[1]) &&   \
          (*this->operands[1]) == (*other.operands[0]));    \
}
#define DEFINE_BINARY_OP_NONCOMMUTATIVE_COMPARE(OpNode)     \
bool TensorExprNode::Compare_(                              \
    const OpNode* const opnode,                             \
    const TensorExprNode & other) const {                   \
  CHECK(this->operands.size() == 2);                        \
  CHECK(other.operands.size() == 2);                        \
  return ((*this->operands[0]) == (*other.operands[0]) &&   \
          (*this->operands[1]) == (*other.operands[1]));    \
}
DEFINE_BINARY_OP_COMMUTATIVE_COMPARE(AddNode)
DEFINE_BINARY_OP_NONCOMMUTATIVE_COMPARE(SubNode)
DEFINE_BINARY_OP_COMMUTATIVE_COMPARE(MulNode)
DEFINE_BINARY_OP_NONCOMMUTATIVE_COMPARE(DivNode)

bool TensorExprNode::Compare_(
    const ReduceNode* const opnode,
    const TensorExprNode& other) const {
  const ReduceNode* const other_opnode = other.opref.as<ReduceNode>();
  CHECK(other_opnode != nullptr);
  if (opnode == other_opnode) {
    return true;
  }
  // To make sure that two reduce nodes are the same, the followings have to be
  // equal: (1) source, (2) commutative reducer, (3) condition.
  return ((*this->operands[0]) == (*other.operands[0]) &&
          (*this->operands[1]) == (*other.operands[1]) &&
          (*this->operands[2]) == (*other.operands[2]));
}


#define DEFINE_IMM_COMPARE(ImmNode)                                \
bool TensorExprNode::Compare_(                                     \
    const ImmNode* const immnode,                                  \
    const TensorExprNode& other) const {                           \
  const ImmNode* const other_immnode = other.opref.as<ImmNode>();  \
  CHECK(other_immnode != nullptr);                                 \
  return immnode->value == other_immnode->value;                   \
}
DEFINE_IMM_COMPARE(IntImmNode)
DEFINE_IMM_COMPARE(FloatImmNode)

#define DISPATCH_TO_CMP(OpNode)                                               \
set_dispatch<OpNode>([](const ObjectRef& opref, const TensorExprNode& other,  \
                        const TensorExprNode* const pthis) ->bool {           \
  if (opref->type_index() != other.opref->type_index()) {                     \
    return false;                                                             \
  }                                                                           \
  return pthis->Compare_(static_cast<const OpNode*>(opref.get()), other);     \
})

TVM_STATIC_IR_FUNCTOR(TensorExprNode, cmptable)
.DISPATCH_TO_CMP(CallNode)
.DISPATCH_TO_CMP(PlaceholderOpNode)
.DISPATCH_TO_CMP(AddNode)
.DISPATCH_TO_CMP(SubNode)
.DISPATCH_TO_CMP(MulNode)
.DISPATCH_TO_CMP(DivNode)
.DISPATCH_TO_CMP(ReduceNode)
.DISPATCH_TO_CMP(IntImmNode)
.DISPATCH_TO_CMP(FloatImmNode);


void TensorExprTree::Construct_(
    const CallNode* const opnode,
    TensorExprNode* const tenode) {
  CHECK(opnode->args.size() == 1)
      << "Current implementation only handles unary CallNode's";
  tenode->operands.push_back(Construct(opnode->args[0], tenode->axis));
}

#define DEFINE_BINARY_OP_CONSTRUCT(OpNode)  \
void TensorExprTree::Construct_(            \
    const OpNode* const opnode,             \
    TensorExprNode* const tenode) {         \
  tenode->operands.push_back(Construct(     \
      opnode->a,                            \
      tenode->axis));                       \
  tenode->operands.push_back(Construct(     \
      opnode->b,                            \
      tenode->axis));                       \
}

DEFINE_BINARY_OP_CONSTRUCT(AddNode)
DEFINE_BINARY_OP_CONSTRUCT(SubNode)
DEFINE_BINARY_OP_CONSTRUCT(MulNode)
DEFINE_BINARY_OP_CONSTRUCT(DivNode)

void TensorExprTree::Construct_(
    const ReduceNode* const opnode,
    TensorExprNode* const tenode) {
  // Array<IterVar> 
}
 */

}  // namespace te
}  // namespace tvm

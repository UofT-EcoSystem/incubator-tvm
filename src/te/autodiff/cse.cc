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


struct TensorExprNode;
typedef std::shared_ptr<TensorExprNode> TensorExprPtr;

struct TensorExprNode {
  ObjectRef opref;
  std::vector<TensorExprPtr> operands;
  Array<PrimExpr> indices;
  using ComputeOpAxis = std::pair<const ComputeOpNode*, size_t>;
  Map<Var, ComputeOpAxis> var_compute_op_axis_map;

  /*! @brief Convert a \c TensorExprNode to string.
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
  using FCompare = NodeFunctor<bool(const ObjectRef&, const TensorExprNode&,
                                    const TensorExprNode* const)>;
  static FCompare & cmptable() {
    static FCompare instance;
    return instance; 
  }
  bool Compare_(const CallNode* const, const TensorExprNode&) const;
  bool Compare_(const PlaceholderOpNode* const, const TensorExprNode&) const;
  bool Compare_(const AddNode* const, const TensorExprNode&) const;
  bool Compare_(const SubNode* const, const TensorExprNode&) const;
  bool Compare_(const MulNode* const, const TensorExprNode&) const;
  bool Compare_(const DivNode* const, const TensorExprNode&) const;
  bool Compare_(const ReduceNode* const, const TensorExprNode&) const;
  bool Compare_(const IntImmNode* const, const TensorExprNode&) const;
  bool Compare_(const FloatImmNode* const, const TensorExprNode&) const;

  /*! \brief Compare two tensor expression subtree.
   */
  bool operator==(const TensorExprNode& other) const {
    static const FCompare & fcompare = cmptable();
    if (this->opref.defined() &&
        other.opref.defined()) {
      return fcompare(this->opref, other, this);
    }
    return !this->opref.defined() && !other.opref.defined();
  }
  bool operator!=(const TensorExprNode& other) const {
    return !operator==(other);
  }
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

  /*! \brief 
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
bool TensorExprNode::Compare_(
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
  for (size_t i = 0; i < this->operands.size(); ++i) {
    if ((*(this->operands[i])) !=
        (*(other.operands[i]))) {
      return false;
    }
  }
  return true;
}

bool TensorExprNode::Compare_(
    const PlaceholderOpNode* const opnode,
    const TensorExprNode& other) const {
  const PlaceholderOpNode* const other_opnode
      = other.opref.as<PlaceholderOpNode>();
  CHECK(other_opnode != nullptr);
  return opnode == other_opnode;
}

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
  CHECK(this->operands.size() == 1);
  CHECK(other.operands.size() == 1);
  return false;
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



}  // namespace te
}  // namespace tvm

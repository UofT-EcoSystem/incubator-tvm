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
#include "ad_util.h"

#include <sstream>

#include <tvm/node/functor.h>
#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>


namespace tvm {
namespace te {
namespace {


/*! \brief The \c IndicesRemapNode remaps the current indices into another.
 */
class IndicesRemapNode : public Object {
 public:
  static constexpr char _type_key[] = "te.IndicesRemap";
  TVM_DECLARE_FINAL_OBJECT_INFO(IndicesRemapNode, Object);
};  // class IndicesRemapNode

class IndicesRemap : public ObjectRef {
 private:
 public:
  IndicesRemap(const ProducerLoadNode& op);
  TVM_DEFINE_OBJECT_REF_METHODS(IndicesRemap, ObjectRef,
                                IndicesRemapNode);
};  // class IndicesRemap


struct TensorExprNode;
typedef std::shared_ptr<TensorExprNode> TensorExprPtr;

struct TensorExprNode {
  ObjectRef opref;
  std::vector<TensorExprPtr> operands;

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
    static FCompare inst;
    return inst; 
  }
  bool Compare_(const CallNode* const op, const TensorExprNode& other) const;
  bool Compare_(const PlaceholderOpNode* const op,
                const TensorExprNode& other) const;
  bool Compare_(const AddNode* const op, const TensorExprNode& other) const;
  bool Compare_(const SubNode* const op, const TensorExprNode& other) const;
  bool Compare_(const MulNode* const op, const TensorExprNode& other) const;
  bool Compare_(const DivNode* const op, const TensorExprNode& other) const;
  bool Compare_(const ReduceNode* const op, const TensorExprNode& other) const;
  bool Compare_(const IntImmNode* const op, const TensorExprNode& other) const;
  bool Compare_(const FloatImmNode* const op, const TensorExprNode& other) const;

  /*! \brief Compare two tensor expression subtree.
   */
  bool operator==(const TensorExprNode & other) const {
    static const FCompare & fcompare = cmptable();
    if (this->opref.defined() &&
        other.opref.defined()) {
      return fcompare(this->opref, other, this);
    }
    return !this->opref.defined() && !other.opref.defined();
  }
};


class CSEOptimizer;

/*!
 * \brief The \c TensorExprTree constructs a tensor expression tree out of 
 * 
 * 
 */
class TensorExprTree {
 private:
  
 public:
  TensorExprTree(const Tensor& tensor) {

  }
};  // class TensorExprConstr


/*!
 * \brief The \c CSEOptimizer eliminates the common subexpressions between the source and target
 *        tensor. It is constructed from a source tensor expresssion
 */
class CSEOptimizer {
 private:
  TensorExprTree src, tgt;
 public:
  CSEOptimizer(const Tensor& src);
};  // class CSEOptimizer


}  // namespace anonymous


std::pair<Tensor, std::vector<Tensor> >
CSE(const Tensor& output, const std::vector<Tensor>& input_grads) {
  // 1. Apply auto-inliner to inline the injective operations. The point is to simplify the
  //    tensor expressions, and particularly tensor indices.
  
  // 2. Remove the common subexpresssions between the input gradients.
  for (const Tensor& input_grad : input_grads) {

  }
  return std::make_pair(output, input_grads);
}


IndicesRemap::IndicesRemap(const ProducerLoadNode& op) {
  
}


/**************************************************************************************************
 * TensorExprTree/Node
 **************************************************************************************************/
bool TensorExprNode::Compare_(
    const CallNode* const opnode,
    const TensorExprNode& other) const {
  const CallNode* const other_opnode = other.opref.as<CallNode>();
  CHECK(other_opnode != nullptr);
  return opnode->op.same_as(other_opnode->op) &&
         (*this->operands[0]) == (*other.operands[0]);
}

bool TensorExprNode::Compare_(
    const PlaceholderOpNode* const opnode,
    const TensorExprNode& other) const {
  const PlaceholderOpNode* const other_opnode = other.opref.as<PlaceholderOpNode>();
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
    .DISPATCH_TO_CMP(IntImmNode)
    .DISPATCH_TO_CMP(FloatImmNode);
}  // namespace te
}  // namespace tvm

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
#include "cse.h"

#include <sstream>

#include <tvm/node/functor.h>
#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>


using namespace ::tvm::tir;

namespace tvm {
namespace te {
namespace {


/*! \brief The \c IndicesRemapNode remaps the current indices into another.
 */
class IndicesRemapNode : public Object {
 public:
  static constexpr const char* _type_key
      = "te.IndicesRemap";
  TVM_DECLARE_FINAL_OBJECT_INFO(IndicesRemapNode, Object);
};  // class IndicesRemapNode

class IndicesRemap : public ObjectRef {
 public:
  IndicesRemap(const ProducerLoadNode& op);
  TVM_DEFINE_OBJECT_REF_METHODS(IndicesRemap, ObjectRef,
                                IndicesRemapNode);
};  // class IndicesRemap


struct TensorExprNode;
typedef std::shared_ptr<TensorExprNode> TensorExprPtr;

struct TensorExprNode {
  NodeRef op;
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
  bool Compare_(const PlaceholderOpNode* const op,
                const TensorExprNode& other);
  bool Compare_(const Add* const op, const TensorExprNode);
  bool Compare_(const Sub* const op, const TensorExprNode);
  bool Compare_(const Mul* const op, const TensorExprNode);
  bool Compare_(const Div* const op, const TensorExprNode);
  bool Compare_(const Reduce* const op, const TensorExprNode& other);
  bool Compare_(const IntImm* const op, const TensorExprNode& other);
  bool Compare_(const FloatImm* const op, const TensorExprNode& other);

  /*! \brief Compare two tensor expression subtree.
   */
  bool operator==(const TensorExprNode & other) const {
    static const FCompare & fcompare = cmptable();
    if (this->op.defined() &&
        other.op.defined()) {
      return fcompare(this->op, other, this);
    }
    return !this->op.defined() && !other.op.defined();
  }
};


class CSEOptimizer;

/*!
 * \brief The \c TensorExprConstr constructs a 
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
 *        tensor. It is constructed from a 
 */
class CSEOptimizer {
 private:
  TensorExprTree src, tgt;
 public:
  CSEOptimizer(const Tensor& src);
};  // class CSEOptimizer


}  // namespace anonymous


std::pair<Tensor, Array<Tensor> >
CSE(const Tensor& output, const Array<Tensor>& input_grads) {
  // First, we remove the common subexpresssions between the input gradients.
  for (const Tensor& input_grad : input_grads) {

  }
  return std::make_pair(output, input_grads);
}


namespace {



}  // namespace anonymous
}  // namespace te
}  // namespace tvm

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
 * \file autoinliner.h
 * \brief Tensor Auto-Inliner
 */
#ifndef TVM_TE_AUTODIFF_AUTOINLINER_H_
#define TVM_TE_AUTODIFF_AUTOINLINER_H_

#include <tvm/runtime/container.h>
#include <tvm/te/tensor.h>

#include <vector>
#include <unordered_map>
#include <unordered_set>


namespace tvm {
namespace te {

/*! \brief \c TensorAutoInliner automatically inlines injective tensor expressions into their
 *         respective consumers. The goal is to simplify the tensor expressions so as to reduce the
 *         complexity of certain optimizations (e.g., CSE).
 */
class TensorAutoInliner
{
 public:
  /*! \brief Inline all of \p tensors 's body statements.
   *  \return tensors which do the same compute, but with injective compute operations inlined
   */
  Array<Tensor> Mutate(const Array<Tensor>& tensors);
 private:
  /*! \brief Make a post-order walk-through of the tensors to initialize \c tensor_post_order ands
   *         \c tensor_reverse_map .
   */
  void VisitPostOrder_(const Array<Tensor>& tensors);

  // tensors sorted in post-order
  std::vector<Tensor> tensors_post_order_;
  // reverse mapping that maps each tensor to its consumers
  std::unordered_map<Tensor, std::unordered_set<Tensor> > tensor_reverse_map_;
  // mapping that maps each tensor to its respective compute operation
  std::unordered_map<Tensor, Operation> tensor_compute_op_map_;
};  // class TensorAutoInliner

}  // namespace te
}  // namespace tvm

#endif  // TVM_TE_AUTODIFF_AUTOINLINER_H_

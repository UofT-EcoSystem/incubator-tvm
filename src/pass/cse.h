#pragma once

#include <functional>
#include <queue>
#include <unordered_set>

#include <tvm/tensor.h>


namespace tvm {
namespace ir {


/// @brief  Common Subexpression Elimination (Top-Level Function Call)
void CSE(const Tensor & src, Tensor *  const ptgt);


}  // namespace ir
}  // namespace tvm

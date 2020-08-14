#pragma once


#include <tvm/tensor.h>


namespace tvm {
namespace ir {


/// @brief  Common Subexpression Elimination (Top-Level Function Call)
void CSE(const Tensor & src, Tensor *  const ptgt);


}  // namespace ir
}  // namespace tvm

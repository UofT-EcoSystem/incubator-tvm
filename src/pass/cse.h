#pragma once


#include <tvm/tensor.h>


namespace tvm {
namespace ir {


/// @brief  Common Subexpression Elimination (Top-Level Function Call)
/// @param  output   Output
/// @param  in_args  Input Gradients (from Gradient pass)
/// @return output and input gradients after CSE has been applied
std::pair < Tensor, Array < Tensor > >
CSE(const Tensor & output,
    const Array < Tensor > & in_args);


}  // namespace ir
}  // namespace tvm

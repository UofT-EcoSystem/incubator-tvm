// <bojian/TVM-SymbolicTuning>
#pragma once

#include <sstream>
#include <string>
#include <vector>

#include <tvm/runtime/container.h>


// #define SYMTUNE_DEBUG_TRACE
// #define SYMTUNE_SCHED_OPT_NO_COMPUTE_IF_CHECKS
// #define SYMTUNE_SCHED_OPT_NO_DUP_IF_CHECKS
// #define SYMTUNE_SCHED_OPT_SPLIT_BLOCKIDX


template<typename PrimExprT>
std::string exprs_tostr(
    const std::vector<PrimExprT>& exprs) {
  std::ostringstream strout;
  strout << "[";
  for (const PrimExprT& expr : exprs) {
    strout << expr << ", ";
  }
  strout << "]";

  return strout.str();
}

template<typename PrimExprT>
std::string exprs_tostr(
    const ::tvm::runtime::Array<PrimExprT>& exprs) {
  std::ostringstream strout;
  strout << "[";
  for (const PrimExprT& expr : exprs) {
    strout << expr << ", ";
  }
  strout << "]";

  return strout.str();
}

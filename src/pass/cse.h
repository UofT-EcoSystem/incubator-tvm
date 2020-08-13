#pragma once

#include <functional>
#include <queue>
#include <unordered_set>

#include <tvm/tensor.h>
#include <tvm/ir_visitor.h>
#include <tvm/tensor.h>


using ::tvm::PlaceholderOp;


namespace tvm {
namespace ir {


class IRComparator
{
private:
public:
        ~IRComparator() {}
        using FCompare = NodeFunctor < bool(const ObjectRef &,
                                            const ObjectRef &, IRComparator *) >;
        static FCompare & vtable();
        bool Compare(const NodeRef & lhs,
                     const NodeRef & rhs)
        {
                static const FCompare & f = vtable();
                if (lhs.defined() && rhs.defined())
                {
                        return f(lhs, rhs, this);
                }
                return false;
        }
        bool Compare_(const Variable * const lhs, const Variable * const rhs);
        bool Compare_(const Call * const lhs, const Call * const rhs);
        bool Compare_(const PlaceholderOp * const lhs,
                      const PlaceholderOp * const rhs);
        bool Compare_(const Add * const lhs, const Add * const rhs);
        bool Compare_(const Sub * const lhs, const Sub * const rhs);
        bool Compare_(const Mul * const lhs, const Mul * const rhs);
        bool Compare_(const Div * const lhs, const Div * const rhs);
};  // class IRComparator


/// @brief  Common Subexpression Elimination (Top-Level Function Call)
void CSE(const Tensor & src, Tensor *  const ptgt);


}  // namespace ir
}  // namespace tvm

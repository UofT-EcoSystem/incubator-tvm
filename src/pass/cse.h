#pragma once

#include <functional>
#include <queue>
#include <unordered_set>

#include <tvm/tensor.h>
#include <tvm/ir_visitor.h>


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
#define NOT_IMPLEMENTED                                                         \
        {                                                                       \
                LOG(FATAL) << "Comparator has not been implemented";            \
        }
        bool Compare_(const Variable * lhs, const Variable * rhs) NOT_IMPLEMENTED
        bool Compare_(const LetStmt * lhs, const LetStmt * rhs) NOT_IMPLEMENTED
        bool Compare_(const AttrStmt * lhs, const AttrStmt * rhs) NOT_IMPLEMENTED
        bool Compare_(const IfThenElse * lhs,
                      const IfThenElse * rhs) NOT_IMPLEMENTED
        bool Compare_(const For * lhs, const For * RHS) NOT_IMPLEMENTED
        bool Compare_(const Allocate * lhs, const Allocate * rhs) NOT_IMPLEMENTED
        bool Compare_(const Load * lhs, const Load * rhs) NOT_IMPLEMENTED
        bool Compare_(const Store * lhs, const Store * rhs) NOT_IMPLEMENTED
        bool Compare_(const Let * lhs, const Let * rhs) NOT_IMPLEMENTED
        bool Compare_(const Free * lhs, const Free * rhs) NOT_IMPLEMENTED
        bool Compare_(const Call * lhs, const Call * rhs) NOT_IMPLEMENTED
        bool Compare_(const Add * lhs, const Add * rhs);
        bool Compare_(const Sub * lhs, const Sub * rhs);
        bool Compare_(const Mul * lhs, const Mul * rhs);
        bool Compare_(const Div * lhs, const Div * rhs);
        bool Compare_(const Mod * lhs, const Mod * rhs);
        bool Compare_(const FloorDiv * lhs, const FloorDiv * rhs) NOT_IMPLEMENTED
        bool Compare_(const FloorMod * lhs, const FloorMod * rhs) NOT_IMPLEMENTED
        bool Compare_(const Min * lhs, const Min * rhs) NOT_IMPLEMENTED
        bool Compare_(const Max * lhs, const Max * rhs) NOT_IMPLEMENTED
        bool Compare_(const EQ * lhs, const EQ * rhs) NOT_IMPLEMENTED
        bool Compare_(const NE * lhs, const NE * rhs) NOT_IMPLEMENTED
        bool Compare_(const LT * lhs, const LT * rhs) NOT_IMPLEMENTED
        bool Compare_(const LE * lhs, const LE * rhs) NOT_IMPLEMENTED
        bool Compare_(const GT * lhs, const GT * rhs) NOT_IMPLEMENTED
        bool Compare_(const GE * lhs, const GE * rhs) NOT_IMPLEMENTED
        bool Compare_(const And * lhs, const And * rhs) NOT_IMPLEMENTED
        bool Compare_(const Or * lhs, const Or * rhs) NOT_IMPLEMENTED
        bool Compare_(const Reduce * lhs, const Reduce * rhs) NOT_IMPLEMENTED
        bool Compare_(const Cast * lhs, const Cast * rhs) NOT_IMPLEMENTED
        bool Compare_(const Not * lhs, const Not * rhs) NOT_IMPLEMENTED
        bool Compare_(const Select * lhs, const Select * rhs) NOT_IMPLEMENTED
        bool Compare_(const Ramp * lhs, const Ramp * rhs) NOT_IMPLEMENTED
        bool Compare_(const Shuffle * lhs, const Shuffle * rhs) NOT_IMPLEMENTED
        bool Compare_(const Broadcast * lhs, const Broadcast * rhs) NOT_IMPLEMENTED
        bool Compare_(const AssertStmt * lhs, const AssertStmt * rhs) NOT_IMPLEMENTED
        bool Compare_(const ProducerConsumer * lhs,
                      const ProducerConsumer * rhs) NOT_IMPLEMENTED
        bool Compare_(const Provide * lhs, const Provide * rhs) NOT_IMPLEMENTED
        bool Compare_(const Realize * lhs, const Realize * rhs) NOT_IMPLEMENTED
        bool Compare_(const Prefetch * lhs, const Prefetch * rhs) NOT_IMPLEMENTED
        bool Compare_(const Block * lhs, const Block * rhs) NOT_IMPLEMENTED
        bool Compare_(const Evaluate * lhs, const Evaluate * rhs) NOT_IMPLEMENTED
        bool Compare_(const IntImm * lhs, const IntImm * rhs) NOT_IMPLEMENTED
        bool Compare_(const UIntImm * lhs, const UIntImm * rhs) NOT_IMPLEMENTED
        bool Compare_(const FloatImm * lhs, const FloatImm * rhs) NOT_IMPLEMENTED
        bool Compare_(const StringImm * lhs, const StringImm * rhs) NOT_IMPLEMENTED
};  // class IRComparator


/// @brief  Common Subexpression Elimination (Top-Level Function Call)
void CSE(const Tensor & src, Tensor * const ptgt);


}  // namespace ir
}  // namespace tvm

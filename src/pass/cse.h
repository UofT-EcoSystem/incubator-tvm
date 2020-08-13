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
        bool Compare_(const Variable * lhs, const Variable * rhs);
        bool Compare_(const LetStmt * lhs, const LetStmt * rhs);
        bool Compare_(const AttrStmt * lhs, const AttrStmt * rhs);
        bool Compare_(const IfThenElse * lhs, const IfThenElse * rhs);
        bool Compare_(const For * lhs, const For * RHS);
        bool Compare_(const Allocate * lhs, const Allocate * rhs);
        bool Compare_(const Load * lhs, const Load * rhs);
        bool Compare_(const Store * lhs, const Store * rhs);
        bool Compare_(const Let * lhs, const Let * rhs);
        bool Compare_(const Free * lhs, const Free * rhs);
        bool Compare_(const Call * lhs, const Call * rhs);
        bool Compare_(const Add * lhs, const Add * rhs);
        bool Compare_(const Sub * lhs, const Sub * rhs);
        bool Compare_(const Mul * lhs, const Mul * rhs);
        bool Compare_(const Div * lhs, const Div * rhs);
        bool Compare_(const Mod * lhs, const Mod * rhs);
        bool Compare_(const FloorDiv * lhs, const FloorDiv * rhs);
        bool Compare_(const FloorMod * lhs, const FloorMod * rhs);
        bool Compare_(const Min * lhs, const Min * rhs);
        bool Compare_(const Max * lhs, const Max * rhs);
        bool Compare_(const EQ * lhs, const EQ * rhs);
        bool Compare_(const NE * lhs, const NE * rhs);
        bool Compare_(const LT * lhs, const LT * rhs);
        bool Compare_(const LE * lhs, const LE * rhs);
        bool Compare_(const GT * lhs, const GT * rhs);
        bool Compare_(const GE * lhs, const GE * rhs);
        bool Compare_(const And * lhs, const And * rhs);
        bool Compare_(const Or * lhs, const Or * rhs);
        bool Compare_(const Reduce * lhs, const Reduce * rhs);
        bool Compare_(const Cast * lhs, const Cast * rhs);
        bool Compare_(const Not * lhs, const Not * rhs);
        bool Compare_(const Select * lhs, const Select * rhs);
        bool Compare_(const Ramp * lhs, const Ramp * rhs);
        bool Compare_(const Shuffle * lhs, const Shuffle * rhs);
        bool Compare_(const Broadcast * lhs, const Broadcast * rhs);
        bool Compare_(const AssertStmt * lhs, const AssertStmt * rhs);
        bool Compare_(const ProducerConsumer * lhs,
                      const ProducerConsumer * rhs);
        bool Compare_(const Provide * lhs, const Provide * rhs);
        bool Compare_(const Realize * lhs, const Realize * rhs);
        bool Compare_(const Prefetch * lhs, const Prefetch * rhs);
        bool Compare_(const Block * lhs, const Block * rhs);
        bool Compare_(const Evaluate * lhs, const Evaluate * rhs);
        bool Compare_(const IntImm * lhs, const IntImm * rhs);
        bool Compare_(const UIntImm * lhs, const UIntImm * rhs);
        bool Compare_(const FloatImm * lhs, const FloatImm * rhs);
        bool Compare_(const StringImm * lhs, const StringImm * rhs);
};  // class IRComparator


/// @brief  Common Subexpression Elimination (Top-Level Function Call)
void CSE(const Tensor & src, Tensor * const ptgt);


}  // namespace ir
}  // namespace tvm

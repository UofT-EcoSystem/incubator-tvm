#pragma once

#include <functional>
#include <queue>
#include <unordered_set>

#include <tvm/ir_visitor.h>

namespace tvm {
namespace ir {


class IRComparator
{
private:
public:
        ~IRComparator() {}
        using FCompare = NodeFunctor < void(const ObjectRef &,
                                            const ObjectRef &, IRComparator *) >;
        static FCompare & vtable();
        bool Compare(const NodeRef & lhs,
                     const NodeRef & rhs)
        {
                static const FCompare & f = vtable();
                if (lhs.defined() && rhs.defined())
                {
                        f(lhs, rhs, this);
                }
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
        bool Compare_(const Min * op);
        bool Compare_(const Max * op);
        bool Compare_(const EQ * op);
        bool Compare_(const NE * op);
        bool Compare_(const LT * op);
        bool Compare_(const LE * op);
        bool Compare_(const GT * op);
        bool Compare_(const GE * op);
        bool Compare_(const And* op);
        bool Compare_(const Or * op);
        bool Compare_(const Reduce * op);
        bool Compare_(const Cast * op);
        bool Compare_(const Not * op);
        bool Compare_(const Select * op);
        bool Compare_(const Ramp * op);
        bool Compare_(const Shuffle * op);
        bool Compare_(const Broadcast * op);
        bool Compare_(const AssertStmt * op);
        bool Compare_(const ProducerConsumer * op);
        bool Compare_(const Provide * op);
        bool Compare_(const Realize * op);
        bool Compare_(const Prefetch * op);
        bool Compare_(const Block * op);
        bool Compare_(const Evaluate * op);
        bool Compare_(const IntImm * op);
        bool Compare_(const UIntImm * op);
        bool Compare_(const FloatImm * op);
        bool Compare_(const StringImm * op);
};  // class IRComparator


/// @brief  Common Subexpression Elimination (Top-Level Function Call)
void CSE(const Tensor & src, Tensor * const ptgt);


}  // namespace ir
}  // namespace tvm

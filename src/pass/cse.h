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
#define CMP_NOT_IMPLEMENTED                                                     \
        {                                                                       \
                LOG(FATAL) << "Comparator has not been implemented";            \
                return false;                                                   \
        }
        bool Compare_(const Variable * const lhs, const Variable * const rhs);
        bool Compare_(const LetStmt * const lhs, const LetStmt * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const AttrStmt * const lhs, const AttrStmt * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const IfThenElse * const lhs,
                      const IfThenElse * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const For * const lhs, const For * const RHS) CMP_NOT_IMPLEMENTED
        bool Compare_(const Allocate * const lhs, const Allocate * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const Load * const lhs, const Load * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const Store * const lhs, const Store * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const Let * const lhs, const Let * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const Free * const lhs, const Free * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const Call * const lhs, const Call * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const Add * const lhs, const Add * const rhs);
        bool Compare_(const Sub * const lhs, const Sub * const rhs);
        bool Compare_(const Mul * const lhs, const Mul * const rhs);
        bool Compare_(const Div * const lhs, const Div * const rhs);
        bool Compare_(const Mod * const lhs, const Mod * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const FloorDiv * const lhs, const FloorDiv * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const FloorMod * const lhs, const FloorMod * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const Min * const lhs, const Min * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const Max * const lhs, const Max * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const EQ * const lhs, const EQ * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const NE * const lhs, const NE * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const LT * const lhs, const LT * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const LE * const lhs, const LE * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const GT * const lhs, const GT * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const GE * const lhs, const GE * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const And * const lhs, const And * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const Or * const lhs, const Or * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const Reduce * const lhs, const Reduce * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const Cast * const lhs, const Cast * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const Not * const lhs, const Not * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const Select * const lhs, const Select * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const Ramp * const lhs, const Ramp * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const Shuffle * const lhs, const Shuffle * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const Broadcast * const lhs, const Broadcast * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const AssertStmt * const lhs, const AssertStmt * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const ProducerConsumer * const lhs,
                      const ProducerConsumer * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const Provide * const lhs, const Provide * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const Realize * const lhs, const Realize * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const Prefetch * const lhs, const Prefetch * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const Block * const lhs, const Block * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const Evaluate * const lhs, const Evaluate * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const IntImm * const lhs, const IntImm * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const UIntImm * const lhs, const UIntImm * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const FloatImm * const lhs, const FloatImm * const rhs) CMP_NOT_IMPLEMENTED
        bool Compare_(const StringImm * const lhs, const StringImm * const rhs) CMP_NOT_IMPLEMENTED
};  // class IRComparator


/// @brief  Common Subexpression Elimination (Top-Level Function Call)
void CSE(const Tensor & src, Tensor *  const ptgt);


}  // namespace ir
}  // namespace tvm

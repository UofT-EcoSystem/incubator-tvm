#include <tvm/operation.h>

#include "./cse.h"


namespace tvm {
namespace ir {


#define DISPATCH_TO_COMPARE(op)                                                 \
        set_dispatch < op > ([](const ObjectRef & lhs,                          \
                                const ObjectRef & rhs, IRComparator * v)        \
                               -> bool                                          \
        {                                                                       \
                v->Compare_(static_cast < const op * > (lhs.get()),             \
                            static_cast < const op * > (rhs.get()));            \
        })

TVM_STATIC_IR_FUNCTOR(IRComparator, vtable)
.DISPATCH_TO_COMPARE(Variable)
.DISPATCH_TO_COMPARE(LetStmt)
.DISPATCH_TO_COMPARE(AttrStmt)
.DISPATCH_TO_COMPARE(IfThenElse)
.DISPATCH_TO_COMPARE(For)
.DISPATCH_TO_COMPARE(Allocate)
.DISPATCH_TO_COMPARE(Load)
.DISPATCH_TO_COMPARE(Store)
.DISPATCH_TO_COMPARE(Let)
.DISPATCH_TO_COMPARE(Free)
.DISPATCH_TO_COMPARE(Call)
.DISPATCH_TO_COMPARE(Add)
.DISPATCH_TO_COMPARE(Sub)
.DISPATCH_TO_COMPARE(Mul)
.DISPATCH_TO_COMPARE(Div)
.DISPATCH_TO_COMPARE(Mod)
.DISPATCH_TO_COMPARE(FloorDiv)
.DISPATCH_TO_COMPARE(FloorMod)
.DISPATCH_TO_COMPARE(Min)
.DISPATCH_TO_COMPARE(Max)
.DISPATCH_TO_COMPARE(EQ)
.DISPATCH_TO_COMPARE(NE)
.DISPATCH_TO_COMPARE(LT)
.DISPATCH_TO_COMPARE(LE)
.DISPATCH_TO_COMPARE(GT)
.DISPATCH_TO_COMPARE(GE)
.DISPATCH_TO_COMPARE(And)
.DISPATCH_TO_COMPARE(Or)
.DISPATCH_TO_COMPARE(Reduce)
.DISPATCH_TO_COMPARE(Cast)
.DISPATCH_TO_COMPARE(Not)
.DISPATCH_TO_COMPARE(Select)
.DISPATCH_TO_COMPARE(Ramp)
.DISPATCH_TO_COMPARE(Shuffle)
.DISPATCH_TO_COMPARE(Broadcast)
.DISPATCH_TO_COMPARE(AssertStmt)
.DISPATCH_TO_COMPARE(ProducerConsumer)
.DISPATCH_TO_COMPARE(Provide)
.DISPATCH_TO_COMPARE(Realize)
.DISPATCH_TO_COMPARE(Prefetch)
.DISPATCH_TO_COMPARE(Block)
.DISPATCH_TO_COMPARE(Evaluate)
.DISPATCH_TO_COMPARE(IntImm)
.DISPATCH_TO_COMPARE(UIntImm)
.DISPATCH_TO_COMPARE(FloatImm)
.DISPATCH_TO_COMPARE(StringImm);


void CSE(const Tensor & src, Tensor * const ptgt)
{
        Tensor & tgt = *ptgt;

        // TODO: We limit the scope of analysis to compute.gamma.grad, but will
        //       remove this limitation in later stages.
        if (tgt->op->name != "compute.gamma.grad")
        {
                return;
        }

        std::queue < Tensor > worklist;
        std::unordered_set < Tensor > visited_workitems;
        worklist.push(tgt);
        for (; !worklist.empty(); worklist.pop())
        {
                const Tensor & workitem = worklist.front();

                if (visited_workitems.count(workitem) != 0)
                {
                        // continue;
                }
                visited_workitems.insert(workitem);

                LOG(INFO) << "Visiting Tensor " << workitem->op->name;

                if (const ComputeOpNode * compute_op =
                    workitem->op.as < ComputeOpNode > ()) 
                {
                        for (const Tensor & input : compute_op->InputTensors())
                        {
                                worklist.push(input);
                        }
                }
                if (const PlaceholderOpNode * placeholder_op = 
                    workitem->op.as < PlaceholderOpNode > ())
                {
                        LOG(INFO) << "Visiting Placeholder " << workitem->op;
                }
        }  // for (workitem ∈ worklist)
}


}  // namespace ir
}  // namespace tvm
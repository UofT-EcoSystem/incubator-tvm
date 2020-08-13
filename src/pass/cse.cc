#include <tvm/expr.h>
#include <tvm/operation.h>

#include "./cse.h"


namespace tvm {
namespace ir {


IRComparator::FCompare & IRComparator::vtable()
{
        static FCompare inst;
        return inst;
}


bool IRComparator::Compare_(const Variable * const lhs,
                            const Variable * const rhs)
{
        // > Each variable is UNIQUELY identified by its address.
        return lhs == rhs;
}


#define DISPATCH_TO_COMPARE(Op)                                                 \
        set_dispatch < Op > (                                                   \
                [](const ObjectRef & lhs,                                       \
                   const ObjectRef & rhs, IRComparator * v)                     \
                  -> bool                                                       \
                {                                                               \
                        if (lhs->type_index() != rhs->type_index())             \
                        {                                                       \
                                return false;                                   \
                        }                                                       \
                        return v->Compare_(static_cast < const Op * > (lhs.get()),  \
                                           static_cast < const Op * > (rhs.get())); \
                })

#define DEFINE_NONCOMMUTATIVE_BINARY_OP_COMPARE_(Op)                            \
        bool IRComparator::Compare_(const Op * lhs, const Op * rhs)             \
        {                                                                       \
                return this->Compare(lhs->a, rhs->a) &&                         \
                       this->Compare(lhs->b, rhs->b);                           \
        }

#define DEFINE_COMMUTATIVE_BINARY_OP_COMPARE_(Op)                               \
        bool IRComparator::Compare_(const Op * lhs, const Op * rhs)             \
        {                                                                       \
                return (this->Compare(lhs->a, rhs->a) &&                        \
                        this->Compare(lhs->b, rhs->b)) ||                       \
                       (this->Compare(lhs->a, rhs->b) &&                        \
                        this->Compare(lhs->b, rhs->a));                         \
        }


DEFINE_COMMUTATIVE_BINARY_OP_COMPARE_(Add)
DEFINE_NONCOMMUTATIVE_BINARY_OP_COMPARE_(Sub)
DEFINE_COMMUTATIVE_BINARY_OP_COMPARE_(Mul)
DEFINE_NONCOMMUTATIVE_BINARY_OP_COMPARE_(Div)


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


        Var x ("x");
        Var y ("y");
        IRComparator cmp;
        LOG(INFO) << "x + y == y + x?: " << cmp.Compare(x + y, y + x);
        LOG(INFO) << "x + y == y * x?: " << cmp.Compare(x + y, y * x);
        LOG(INFO) << "x * y + y == y + y * x?: " 
                  << cmp.Compare(x * y + y, y + y * x);

        if (const ComputeOpNode * compute_op =
            src->op.as < ComputeOpNode > ())
        {
                Expr body = compute_op->body[src->value_index];

                cmp.Compare(body, body);
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

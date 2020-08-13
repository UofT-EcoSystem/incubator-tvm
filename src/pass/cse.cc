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


bool IRComparator::_Compare(const Array < Expr > & lhs,
                            const Array < Expr > & rhs)
{
        if (lhs.size() != rhs.size())
        {
                return false;
        }
        for (size_t i = 0; i < lhs.size(); ++i)
        {
                if (Compare(lhs[i], rhs[i]) == false)
                {
                        return false;
                }
        }
        return true;
}


bool IRComparator::_Compare(const Variable * const lhs,
                            const Variable * const rhs)
{
        // > Each variable is UNIQUELY identified by its address.
        return lhs == rhs;
}


bool IRComparator::_Compare(const Call * const lhs, const Call * const rhs)
{
        if (lhs->type != rhs->type || 
            lhs->name != rhs->name ||
            lhs->call_type != rhs->call_type ||
            lhs->value_index != rhs->value_index)
        {
                return false;
        }
        if (lhs->call_type == Call::CallType::Halide)
        {
                if (lhs->func->GetTypeKey() !=
                    rhs->func->GetTypeKey())
                {
                        return false;
                }
                if (lhs->func->GetTypeKey() == 
                    ::tvm::PlaceholderOpNode::_type_key)
                {
                        return lhs->func == rhs->func && 
                               // Note that here we are invoking the private
                               // auxiliary function for arrays of expressions,
                               // instead of doing the dispatch.
                               _Compare(lhs->args, rhs->args);
                }
                else
                {
                        LOG(FATAL) << "Comparator has not been implemented "
                                      "for func=" << lhs->func;
                }
        }
        else if (lhs->call_type == Call::CallType::PureIntrinsic)
        {
                if (lhs->name == "exp" ||
                    lhs->name == "log" ||
                    lhs->name == "sigmoid" ||
                    lhs->name == "sqrt" ||
                    lhs->name == "tanh" ||
                    lhs->name == "pow" ||
                    lhs->name == "fabs")
                {
                        return Compare(lhs->args[0], rhs->args[0]);
                }
                else 
                {
                        LOG(FATAL) << "Comparator has not been implemented "
                                      "for name=" << lhs->name;
                }
        }
        LOG(FATAL) << "Comparator has not been implemented "
                      "for call_type=" << lhs->call_type;
        return false;
}


#define DEFINE_NONCOMMUTATIVE_BINARY_OP__COMPARE(Op)                            \
        bool IRComparator::_Compare(const Op * lhs, const Op * rhs)             \
        {                                                                       \
                return this->Compare(lhs->a, rhs->a) &&                         \
                       this->Compare(lhs->b, rhs->b);                           \
        }

#define DEFINE_COMMUTATIVE_BINARY_OP__COMPARE(Op)                               \
        bool IRComparator::_Compare(const Op * lhs, const Op * rhs)             \
        {                                                                       \
                return (this->Compare(lhs->a, rhs->a) &&                        \
                        this->Compare(lhs->b, rhs->b)) ||                       \
                       (this->Compare(lhs->a, rhs->b) &&                        \
                        this->Compare(lhs->b, rhs->a));                         \
        }

DEFINE_COMMUTATIVE_BINARY_OP__COMPARE(Add)
DEFINE_NONCOMMUTATIVE_BINARY_OP__COMPARE(Sub)
DEFINE_COMMUTATIVE_BINARY_OP__COMPARE(Mul)
DEFINE_NONCOMMUTATIVE_BINARY_OP__COMPARE(Div)


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
                        return v->_Compare(static_cast < const Op * > (lhs.get()),  \
                                           static_cast < const Op * > (rhs.get())); \
                })

TVM_STATIC_IR_FUNCTOR(IRComparator, vtable)
.DISPATCH_TO_COMPARE(Variable)
.DISPATCH_TO_COMPARE(Call)
.DISPATCH_TO_COMPARE(Add)
.DISPATCH_TO_COMPARE(Sub)
.DISPATCH_TO_COMPARE(Mul)
.DISPATCH_TO_COMPARE(Div);


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

                LOG(INFO) << "body == body?: " << cmp.Compare(body, body);
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

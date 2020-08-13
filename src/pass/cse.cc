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


bool IRComparator::_Compare(const Array < Expr > & lhs, const Array < Expr > & rhs)
{
        if (lhs.size() != rhs.size())
        {
                LOG(INFO) << "Returning false";
                return false;
        }
        for (size_t i = 0; i < lhs.size(); ++i)
        {
                if (Compare(lhs[i], rhs[i]) == false)
                {
                        LOG(INFO) << "Returning false";
                        return false;
                }
        }
        LOG(INFO) << "Returning true";
        return true;
}


bool IRComparator::_Compare(const IterVar & lhs, const IterVar & rhs)
{
        if (lhs->iter_type != rhs->iter_type)
        {
                LOG(INFO) << "Returning false";
                return false;
        }
        bool ret =
               Compare(lhs->dom->min, rhs->dom->min) &&
               Compare(lhs->dom->extent, rhs->dom->extent);
        LOG(INFO) << "Returning " << ret;
        return ret;
}


bool IRComparator::_Compare(const Array < IterVar > & lhs,
                            const Array < IterVar > & rhs)
{
        if (lhs.size() != rhs.size())
        {
                LOG(INFO) << "Returning false";
                return false;
        }
        for (size_t i = 0; i < lhs.size(); ++i)
        {
                if (_Compare(lhs[i], rhs[i]) == false)
                {
                        LOG(INFO) << "Returning false";
                        return false;
                }
        }
        LOG(INFO) << "Returning true";
        return true;
}


bool IRComparator::_Compare(const Variable * const lhs,
                            const Variable * const rhs)
{
        // > Each variable is UNIQUELY identified by its address.
        bool ret =
               lhs == rhs;
        LOG(INFO) << "Returning " << ret;
        return ret;
}


bool IRComparator::_Compare(const Call * const lhs, const Call * const rhs)
{
        if (lhs->type != rhs->type || 
            lhs->name != rhs->name ||
            lhs->call_type != rhs->call_type ||
            lhs->value_index != rhs->value_index)
        {
                LOG(INFO) << "Returning false";
                return false;
        }
        if (lhs->call_type == Call::CallType::Halide)
        {
                if (lhs->func->GetTypeKey() !=
                    rhs->func->GetTypeKey())
                {
                        LOG(INFO) << "Returning false";
                        return false;
                }
                if (lhs->func->GetTypeKey() == 
                    ::tvm::PlaceholderOpNode::_type_key)
                {
                        bool ret =
                               lhs->func == rhs->func && 
                               _Compare(lhs->args, rhs->args);
                        LOG(INFO) << "Returning " << ret;
                        return ret;
                }
                else if (lhs->func->GetTypeKey() == 
                         ::tvm::ComputeOpNode::_type_key)
                {
                        const ComputeOpNode 
                                * lhs_compute_op = lhs->func.as < ComputeOpNode > (),
                                * rhs_compute_op = rhs->func.as < ComputeOpNode > ();
                        bool ret =
                               _Compare(lhs_compute_op->axis, rhs_compute_op->axis) &&
                               _Compare(lhs_compute_op->reduce_axis,
                                        rhs_compute_op->reduce_axis) &&
                               Compare(lhs_compute_op->body[lhs->value_index],
                                       rhs_compute_op->body[rhs->value_index]);
                        LOG(INFO) << "Returning " << ret;
                        return ret;
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
                        bool ret = Compare(lhs->args[0], rhs->args[0]);
                        LOG(INFO) << "Returning " << ret;
                        return ret;
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


#define DEFINE_NONCOMMUTATIVE_BINARY_OP_COMPARE(Op)                             \
        bool IRComparator::_Compare(const Op * const lhs,                       \
                                    const Op * const rhs)                       \
        {                                                                       \
                bool ret =                                                      \
                       this->Compare(lhs->a, rhs->a) &&                         \
                       this->Compare(lhs->b, rhs->b);                           \
                LOG(INFO) << "Returning " << ret;                               \
                return ret;                                                     \
        }

#define DEFINE_COMMUTATIVE_BINARY_OP_COMPARE(Op)                                \
        bool IRComparator::_Compare(const Op * const lhs,                       \
                                    const Op * const rhs)                       \
        {                                                                       \
                bool ret =                                                      \
                       (this->Compare(lhs->a, rhs->a) &&                        \
                        this->Compare(lhs->b, rhs->b)) ||                       \
                       (this->Compare(lhs->a, rhs->b) &&                        \
                        this->Compare(lhs->b, rhs->a));                         \
                LOG(INFO) << "Returning " << ret;                               \
                return ret;                                                     \
        }

DEFINE_COMMUTATIVE_BINARY_OP_COMPARE(Add)
DEFINE_NONCOMMUTATIVE_BINARY_OP_COMPARE(Sub)
DEFINE_COMMUTATIVE_BINARY_OP_COMPARE(Mul)
DEFINE_NONCOMMUTATIVE_BINARY_OP_COMPARE(Div)

#define DEFINE_IMM_COMPARE(Imm)                                                 \
        bool IRComparator::_Compare(const Imm * const lhs,                      \
                                    const Imm * const rhs)                      \
        {                                                                       \
                bool ret =                                                      \
                       lhs->type == rhs->type &&                                \
                       lhs->value == rhs->value;                                \
                LOG(INFO) << "Returning " << ret;                               \
                return ret;                                                     \
        }

DEFINE_IMM_COMPARE(IntImm);
DEFINE_IMM_COMPARE(UIntImm);
DEFINE_IMM_COMPARE(FloatImm);

#define DISPATCH_TO_COMPARE(Op)                                                 \
        set_dispatch < Op > (                                                   \
                [](const ObjectRef & lhs,                                       \
                   const ObjectRef & rhs, IRComparator * v)                     \
                  -> bool                                                       \
                {                                                               \
                        if (lhs->type_index() != rhs->type_index())             \
                        {                                                       \
                                LOG(INFO) << "Returning false";                 \
                                return false;                                   \
                        }                                                       \
                        bool ret =                                              \
                                v->_Compare(static_cast < const Op * > (lhs.get()),  \
                                            static_cast < const Op * > (rhs.get())); \
                        LOG(INFO) << "Returning " << ret;                       \
                        return ret;                                             \
                })

TVM_STATIC_IR_FUNCTOR(IRComparator, vtable)
.DISPATCH_TO_COMPARE(Variable)
.DISPATCH_TO_COMPARE(Call)
.DISPATCH_TO_COMPARE(Add)
.DISPATCH_TO_COMPARE(Sub)
.DISPATCH_TO_COMPARE(Mul)
.DISPATCH_TO_COMPARE(Div)
.DISPATCH_TO_COMPARE(IntImm)
.DISPATCH_TO_COMPARE(UIntImm)
.DISPATCH_TO_COMPARE(FloatImm);


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

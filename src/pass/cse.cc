#include <functional>
#include <queue>
#include <unordered_set>

#include <tvm/expr.h>
#include <tvm/ir_visitor.h>
#include <tvm/operation.h>

#include "./cse.h"


#define CHECKPOINT_RETURN 1

#if CHECKPOINT_RETURN
#define RETURN(v)                                                               \
        do {                                                                    \
                bool ret = v;                                                   \
                LOG(INFO) << std::boolalpha << ret << std::noboolalpha;         \
                return ret;                                                     \
        } while(0)
#else
#define RETURN(v)  return v
#endif

namespace tvm {
namespace ir {


class IRComparator
{
private:
        bool _Compare(const Array < Expr > & lhs, const Array < Expr > & rhs);
        bool _Compare(const IterVar & lhs, const IterVar & rhs);
        bool _Compare(const Array < IterVar > & lhs,
                      const Array < IterVar > & rhs);
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
        bool _Compare(const Variable * const lhs, const Variable * const rhs);
        bool _Compare(const Call * const lhs, const Call * const rhs);
        bool _Compare(const Add * const lhs, const Add * const rhs);
        bool _Compare(const Sub * const lhs, const Sub * const rhs);
        bool _Compare(const Mul * const lhs, const Mul * const rhs);
        bool _Compare(const Div * const lhs, const Div * const rhs);
        bool _Compare(const Reduce * const lhs, const Reduce * const rhs);
        bool _Compare(const IntImm * const lhs, const IntImm * const rhs);
        bool _Compare(const UIntImm * const lhs, const UIntImm * const rhs);
        bool _Compare(const FloatImm * const lhs, const FloatImm * const rhs);
};  // class IRComparator


IRComparator::FCompare & IRComparator::vtable()
{
        static FCompare inst;
        return inst;
}


bool IRComparator::_Compare(const Array < Expr > & lhs, const Array < Expr > & rhs)
{
        if (lhs.size() != rhs.size())
        {
                RETURN(false);
        }
        for (size_t i = 0; i < lhs.size(); ++i)
        {
                if (Compare(lhs[i], rhs[i]) == false)
                {
                        RETURN(false);
                }
        }
        RETURN(true);
}


bool IRComparator::_Compare(const IterVar & lhs, const IterVar & rhs)
{
        if (lhs->iter_type != rhs->iter_type)
        {
                RETURN(false);
        }
        RETURN(Compare(lhs->dom->min, rhs->dom->min) &&
               Compare(lhs->dom->extent, rhs->dom->extent));
}


bool IRComparator::_Compare(const Array < IterVar > & lhs,
                            const Array < IterVar > & rhs)
{
        if (lhs.size() != rhs.size())
        {
                RETURN(false);
        }
        for (size_t i = 0; i < lhs.size(); ++i)
        {
                if (_Compare(lhs[i], rhs[i]) == false)
                {
                        RETURN(false);
                }
        }
        RETURN(true);
}


bool IRComparator::_Compare(const Variable * const lhs,
                            const Variable * const rhs)
{
        // > Each variable is UNIQUELY identified by its address.
        RETURN(lhs->type == rhs->type);
}


bool IRComparator::_Compare(const Call * const lhs, const Call * const rhs)
{
        if (lhs->type != rhs->type || 
            lhs->name != rhs->name ||
            lhs->call_type != rhs->call_type ||
            lhs->value_index != rhs->value_index)
        {
                RETURN(false);
        }
        if (lhs->call_type == Call::CallType::Halide)
        {
                if (lhs->func->GetTypeKey() !=
                    rhs->func->GetTypeKey())
                {
                        RETURN(false);
                }
                if (lhs->func->GetTypeKey() == 
                    ::tvm::PlaceholderOpNode::_type_key)
                {
                        RETURN(lhs->func == rhs->func && 
                               _Compare(lhs->args, rhs->args));
                }
                else if (lhs->func->GetTypeKey() == 
                         ::tvm::ComputeOpNode::_type_key)
                {
                        const ComputeOpNode 
                                * lhs_compute_op = lhs->func.as < ComputeOpNode > (),
                                * rhs_compute_op = rhs->func.as < ComputeOpNode > ();
                        RETURN(_Compare(lhs_compute_op->axis, rhs_compute_op->axis) &&
                               _Compare(lhs_compute_op->reduce_axis,
                                        rhs_compute_op->reduce_axis) &&
                               Compare(lhs_compute_op->body[lhs->value_index],
                                       rhs_compute_op->body[rhs->value_index]));
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
                        RETURN(Compare(lhs->args[0], rhs->args[0]));
                }
                else 
                {
                        LOG(FATAL) << "Comparator has not been implemented "
                                      "for name=" << lhs->name;
                }
        }
        LOG(FATAL) << "Comparator has not been implemented "
                      "for call_type=" << lhs->call_type;
        RETURN(false);
}


#define DEFINE_NONCOMMUTATIVE_BINARY_OP_COMPARE(Op)                             \
        bool IRComparator::_Compare(const Op * const lhs,                       \
                                    const Op * const rhs)                       \
        {                                                                       \
                RETURN(this->Compare(lhs->a, rhs->a) &&                         \
                       this->Compare(lhs->b, rhs->b));                          \
        }

#define DEFINE_COMMUTATIVE_BINARY_OP_COMPARE(Op)                                \
        bool IRComparator::_Compare(const Op * const lhs,                       \
                                    const Op * const rhs)                       \
        {                                                                       \
                RETURN((this->Compare(lhs->a, rhs->a) &&                        \
                        this->Compare(lhs->b, rhs->b)) ||                       \
                       (this->Compare(lhs->a, rhs->b) &&                        \
                        this->Compare(lhs->b, rhs->a)));                        \
        }

DEFINE_COMMUTATIVE_BINARY_OP_COMPARE(Add)
DEFINE_NONCOMMUTATIVE_BINARY_OP_COMPARE(Sub)
DEFINE_COMMUTATIVE_BINARY_OP_COMPARE(Mul)
DEFINE_NONCOMMUTATIVE_BINARY_OP_COMPARE(Div)

#define DEFINE_IMM_COMPARE(Imm)                                                 \
        bool IRComparator::_Compare(const Imm * const lhs,                      \
                                    const Imm * const rhs)                      \
        {                                                                       \
                RETURN(lhs->type == rhs->type &&                                \
                       lhs->value == rhs->value);                               \
        }


bool IRComparator::_Compare(const Reduce * const lhs,
                            const Reduce * const rhs)
{
        RETURN(_Compare(lhs->combiner->result, 
                        rhs->combiner->result) &&  // Array < Expr >
               _Compare(lhs->combiner->identity_element,
                        rhs->combiner->identity_element) &&  // Array < Expr >
               _Compare(lhs->source, rhs->source) &&  // Array < Expr >
               _Compare(lhs->axis, rhs->axis) &&      // Array < IterVar >
               Compare(lhs->condition, rhs->condition) &&  // Expr
               lhs->value_index == rhs->value_index);
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
                                RETURN(false);                                  \
                        }                                                       \
                        RETURN(v->_Compare(static_cast < const Op * > (lhs.get()),    \
                                           static_cast < const Op * > (rhs.get())));  \
                })

TVM_STATIC_IR_FUNCTOR(IRComparator, vtable)
.DISPATCH_TO_COMPARE(Variable)
.DISPATCH_TO_COMPARE(Call)
.DISPATCH_TO_COMPARE(Add)
.DISPATCH_TO_COMPARE(Sub)
.DISPATCH_TO_COMPARE(Mul)
.DISPATCH_TO_COMPARE(Div)
.DISPATCH_TO_COMPARE(Reduce)
.DISPATCH_TO_COMPARE(IntImm)
.DISPATCH_TO_COMPARE(UIntImm)
.DISPATCH_TO_COMPARE(FloatImm);


class IRPreOrderVisitor : public IRVisitor
{
private:
        std::function < void(const NodeRef &) > _f;
        std::unordered_set < const Node * > _visited_nodes;
public:
        IRPreOrderVisitor(std::function < void(const NodeRef &) > f) : _f(f) {}
        void Visit(const NodeRef & node) final
        {
                if (_visited_nodes.count(node.get()) != 0)
                {
                        return;
                }
                _visited_nodes.insert(node.get());
                _f(node);
                IRVisitor::Visit(node);
        }
};


/// TODO: This should be @c Mutator instead of @c Visitor .
class CSEVisitor : public IRVisitor
{
private:
        Expr _src_expr;
        std::unordered_set < const Node * > _visited_nodes;
        IRComparator _cmp;
public:
        CSEVisitor(const Expr & src_expr) : _src_expr(src_expr) {}
        void Visit(const NodeRef & node) final
        {
                if (_visited_nodes.count(node.get()) != 0)
                {
                        return;
                }
                _visited_nodes.insert(node.get());
                LOG(INFO) << "Visiting node " << node;
                IRPreOrderVisitor ir_pre_order_visitor (
                        [&node, this](const NodeRef & src_node)
                        {
                                LOG(INFO) << "Comparing with source node "
                                          << src_node;
                                if (this->_cmp.Compare(node, src_node))
                                {
                                        LOG(INFO) << node << " == " << src_node;
                                }
                        });
                IRVisitor::Visit(node);
        }
};  // class CSEVisitors


void CSE(const Tensor & src, Tensor * const ptgt)
{
        Tensor & tgt = *ptgt;

        // TODO: We limit the scope of analysis to compute.gamma.grad, but will
        //       remove this limitation in later stages.
        if (tgt->op->name != "compute.gamma.grad")
        {
                return;
        }

        Var x ("x"), y ("y"), z ("z");
        Integer _0 (0), _4 (4);
        IterVar i = reduce_axis(Range(_0, _4), "i");
        IRComparator cmp;
        LOG(INFO) << "x + y == y + x?: " << cmp.Compare(x + y, y + x);
        LOG(INFO) << "x + y == y * x?: " << cmp.Compare(x + y, y * x);
        LOG(INFO) << "x * y + y == y + y * x?: " 
                  << cmp.Compare(x * y + y, y + y * x);
        CommReducer combiner =
                CommReducerNode::make({x}, {y}, {ir::Mul::make(x, y)},
                                      {make_const(x->type, 1)});
        Expr reduce_z = Reduce::make(combiner, {z}, {i},
                                     make_const(Bool(1), true), 0);
        LOG(INFO) << "Reduce(z) == Reduce(z)?: " << cmp.Compare(reduce_z, reduce_z);

        if (const ComputeOpNode * compute_op =
            src->op.as < ComputeOpNode > ())
        {
                Expr body = compute_op->body[src->value_index];

                LOG(INFO) << "body == body?: " << cmp.Compare(body, body);
        }
        CSEVisitor cse_visitor (z * x * x);
        cse_visitor.Visit(x * x * y);

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

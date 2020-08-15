#include <functional>
#include <queue>
#include <unordered_set>

#include <tvm/expr.h>
#include <tvm/ir_visitor.h>
#include <tvm/operation.h>

#include "./_cse.h"


namespace tvm {
namespace ir {
namespace {

/*
#define CHECKPOINT_RETURN 0

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
        // RETURN(lhs->type == rhs->type);
        RETURN(lhs == rhs);
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


#define DEFINE_IMM_COMPARE(Imm)                                                 \
        bool IRComparator::_Compare(const Imm * const lhs,                      \
                                    const Imm * const rhs)                      \
        {                                                                       \
                RETURN(lhs->type  == rhs->type &&                               \
                       lhs->value == rhs->value);                               \
        }


DEFINE_IMM_COMPARE(IntImm)
DEFINE_IMM_COMPARE(UIntImm)
DEFINE_IMM_COMPARE(FloatImm)

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


class IRAssociativePreOrderVisitor : public IRVisitor
{
private:
        std::function < void(const NodeRef &) > _f;
protected:
        std::unordered_set < const Node * > _visited_nodes;
public:
        IRAssociativePreOrderVisitor(std::function < void(const NodeRef &) > f) : _f(f) {}
        void Visit_(const Call * op) override;
        /// @brief Override @c Add and @c Mul to take into account 
#define DEFINE_ASSOCIATIVE_VISIT(Op)                                            \
        void Visit_(const Op * op) override                                     \
        {                                                                       \
                IRVisitor::Visit_(op);                                          \
                if (const Op * lhs = op->a.as < Op > ())                        \
                {                                                               \
                        IRVisitor::Visit_(                                      \
                                Op::make(lhs->a,                                \
                                         Op::make(lhs->b, op->b))               \
                                .as < Op > ());                                 \
                }                                                               \
                else if (const Op * rhs = op->b.as < Op > ())                   \
                {                                                               \
                        IRVisitor::Visit_(                                      \
                                Op::make(Op::make(rhs->a, op->a),               \
                                         rhs->b).as < Op > ());                 \
                }                                                               \
        }
        DEFINE_ASSOCIATIVE_VISIT(Add)
        DEFINE_ASSOCIATIVE_VISIT(Mul)
        void Visit(const NodeRef & node) override
        {
                if (_visited_nodes.count(node.get()) != 0)
                {
                        return;
                }
                _visited_nodes.insert(node.get());
                _f(node);
                IRVisitor::Visit(node);
        }
};  // class IRAssociativePreOrderVisit


void IRAssociativePreOrderVisitor::Visit_(const Call * op) 
{
        IRVisitor::Visit_(op);
        if (const ComputeOpNode * compute_op =
            op->func.as < ComputeOpNode > ())
        {
                IRVisitor::Visit(compute_op->body[op->value_index]);
        }
}


/// TODO: This should be @c Mutator instead of @c Visitor .
class CSEVisitor : public IRAssociativePreOrderVisitor
{
private:
        Expr _src_expr;
        IRComparator _cmp;
public:
        CSEVisitor(const Expr & src_expr)
                : IRAssociativePreOrderVisitor(nullptr), 
                  _src_expr(src_expr)
        {}
        void Visit(const NodeRef & node) override final
        {
                if (_visited_nodes.count(node.get()) != 0)
                {
                        return;
                }
                _visited_nodes.insert(node.get());
                LOG(INFO) << "Visiting [" << node->GetTypeKey() << "] " 
                          << node;
                // IRAssociativePreOrderVisitor ir_pre_order_visitor (
                //         [&node, this](const NodeRef & src_node)
                //         {
                //                 // LOG(INFO) << "Comparing with source node "
                //                 //           << src_node;
                //                 if (this->_cmp.Compare(node, src_node))
                //                 {
                //                         LOG(INFO) << node << " == " << src_node;
                //                 }
                //         });
                // ir_pre_order_visitor.Visit(_src_expr);
                IRVisitor::Visit(node);
        }
};  // class CSEVisitors
 */

/*
class TensorVisitor : public IRVisitor
{
protected:
        std::unordered_set < const Node * > _visited_nodes;
public:
        TensorVisitor() {}
        void Visit(const NodeRef & node) override;
};


void TensorVisitor::Visit(const NodeRef & node)
{
        if (_visited_nodes.count(node.get()) != 0)
        {
                return;
        }
        _visited_nodes.insert(node.get());
        LOG(INFO) << "[" << node->GetTypeKey() << "] "
                  << node;
        if (const Call * call = node.as < Call > ()) 
        {
                if (call->call_type == Call::CallType::Halide)
                {
                        if (const ComputeOpNode * compute_op =
                            call->func.as < ComputeOpNode > ())
                        {
                                LOG(INFO) << "Visiting compute op " << compute_op->name;
                                _visited_nodes.insert(compute_op);
                                Visit(compute_op->body[call->value_index]);
                        }
                        if (const PlaceholderOpNode * placeholder_op = 
                            call->func.as < PlaceholderOpNode > ())
                        {
                                LOG(INFO) << "Visiting placeholder op "
                                          << placeholder_op->name;
                        }
                }
        }
        IRVisitor::Visit(node);
}
 */


class TensorPostOrderVisitor
{
private:
        std::unordered_set < Tensor > _visited_tensors;
public:
        void Visit(const Tensor & tensor)
        {
                if (_visited_tensors.count(tensor))
                {
                        return;
                }
                _visited_tensors.insert(tensor);
                if (const ComputeOpNode * compute_op =
                    tensor->op.as < ComputeOpNode > ()) 
                {
                        for (const Tensor & input_tensor : 
                             compute_op->InputTensors())
                        {
                                Visit(input_tensor);
                        }
                }
                LOG(INFO) << "Visiting [tensor] " << tensor;
        }
};


}   // namespace anonymous


/*
void _CSE(const Tensor & src, Tensor * const ptgt)
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
        CHECK_EQ(cmp.Compare(x + y, y + x), true);
        CHECK_EQ(cmp.Compare(x + y, y * x), false);
        CHECK_EQ(cmp.Compare(x * y + y, y + y * x), true);
        CommReducer combiner =
                CommReducerNode::make({x}, {y}, {ir::Mul::make(x, y)},
                                      {make_const(x->type, 1)});
        Expr reduce_z = Reduce::make(combiner, {z}, {i},
                                     make_const(Bool(1), true), 0);
        CHECK_EQ(cmp.Compare(reduce_z, reduce_z), true);

        if (const ComputeOpNode * compute_op =
            src->op.as < ComputeOpNode > ())
        {
                Expr body = compute_op->body[src->value_index];
                CHECK_EQ(cmp.Compare(body, body), true);
        }
        CSEVisitor(z * x * x).Visit(x * x * y);

        const ComputeOpNode * src_compute_op = src->op.as < ComputeOpNode > (),
                            * tgt_compute_op = (*ptgt)->op
                                               .as < ComputeOpNode > ();
        
        if (src_compute_op == nullptr ||
            tgt_compute_op == nullptr)
        {
                return;
        }
        CSEVisitor(src_compute_op->body[src->value_index])
            .Visit(tgt_compute_op->body[(*ptgt)->value_index]);
}
 */


void _CSE(const Tensor & src, Tensor * const ptgt)
{
        Tensor & tgt = *ptgt;

        // TODO: We limit the scope of analysis to compute.gamma.grad, but will
        //       remove this limitation in later stages.
        if (tgt->op->name != "compute.gamma.grad")
        {
                return;
        }

        TensorPostOrderVisitor().Visit(tgt);
        /*
        std::queue < Tensor > worklist;
        std::unordered_set < Tensor > visited_tensors; 
        worklist.push(tgt);
        for (; !worklist.empty(); worklist.pop())
        {
                const Tensor & workitem = worklist.front();

                if (visited_tensors.count(workitem) != 0)
                {
                        // continue;
                }
                visited_tensors.insert(workitem);

                LOG(INFO) << "Visiting Tensor " << workitem->op->name;
                if (const ComputeOpNode * compute_op =
                    workitem->op.as < ComputeOpNode > ()) 
                {
                        LOG(INFO) << "Visiting [ComputeOp] " << workitem->op;
                        for (const Tensor & input : compute_op->InputTensors())
                        {
                                worklist.push(input);
                        }
                }
                if (const PlaceholderOpNode * placeholder_op = 
                    workitem->op.as < PlaceholderOpNode > ())
                {
                        LOG(INFO) << "Visiting [PlaceholderOp] " << workitem->op;
                }
        }  // for (workitem ∈ worklist)
         */
}


}  // namespace ir
}  // namespace tvm

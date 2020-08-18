#include <functional>
#include <queue>
#include <unordered_set>

#include <dmlc/parameter.h>
#include <tvm/expr.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>
#include <tvm/operation.h>

#include "./_cse.h"
#include "./zero_elimination.h"


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


std::string Axis2Str(const Array < IterVar > & axis)
{
        std::ostringstream strout;
        strout << "[";
        for (const IterVar & iter_var : axis)
        {
                strout << iter_var << ", ";
        }
        strout << "]";
        return strout.str();
}


namespace {


struct TensorExpr
{
        NodeRef op;
        Array < IterVar > axis, ordered_axis;
        std::vector < std::shared_ptr < TensorExpr > > operands;
        
        std::string toString(const unsigned indent = 0)
        {
                std::ostringstream strout;
                strout << "\n";
                for (unsigned i = 0; i < indent; ++i)
                {
                        strout << " ";
                }
                strout << op << " @" << op.get();
                strout << " [axis=" << Axis2Str(axis) << ", " 
                          "ordered_axis="
                       << Axis2Str(ordered_axis) << "]";
                for (const auto & operand : operands)
                {
                        strout << operand->toString(indent + 2);
                }
                return strout.str();
        }
};
typedef std::shared_ptr < TensorExpr >  TensorExprPtr;


class TensorExprConstructor
{
private:
        std::unordered_map < const FunctionBaseNode *,
                             TensorExprPtr > _tensor_expr_map;
        std::unordered_map < const Node *, 
                             TensorExprPtr > _node_tensorexpr_map;


        void ArgsToOrderedAxis(const Array < Expr > & args,
                               const Array < IterVar > & axis,
                               Array < IterVar > * ordered_axis)
        {
                *ordered_axis = Array < IterVar > (args.size(), IterVar(nullptr));

                for (size_t arg_idx = 0; arg_idx < args.size(); ++arg_idx)
                {
                        const Variable * var
                                = args[arg_idx].as < Variable > ();
                        for (const IterVar & iv : axis)
                        {
                                if (args[arg_idx].as < Variable > () ==
                                    iv->var.get())
                                {
                                        (*ordered_axis).Set(arg_idx, iv);
                                }
                        }
                }
        }
public:
        using FConstruct
                = NodeFunctor < void(const ObjectRef &,
                                     TensorExpr * const,
                                     TensorExprConstructor * const)
                              >;
        static FConstruct & cstrtable() { static FConstruct inst; return inst; }

        void _Construct(const Call * const op,
                        TensorExpr * const expr)
        {
                if (op->call_type == Call::CallType::Halide)
                {
                        const FunctionBaseNode * call_func
                                = op->func.as < FunctionBaseNode > ();
                        CHECK(call_func != nullptr);
                        Array < IterVar > expr_axis = expr->axis;
                        *expr = *_tensor_expr_map.at(call_func);
                        expr->axis = expr_axis;
                        ArgsToOrderedAxis(op->args, expr_axis,
                                          &expr->ordered_axis);
                }
                else if (op->call_type == Call::CallType::PureIntrinsic)
                {
                        expr->operands.push_back(Construct(op->args[0], expr->axis));
                        expr->ordered_axis
                                = expr->operands[0]->ordered_axis;
                }
                else
                {
                        LOG(FATAL) << "NOT implemented for "
                                   << GetRef < Expr > (op);
                }
        }

#define DEFINE_BINARY_OP_CSTR(Op)                                               \
        void _Construct(const Op * const op,                                    \
                        TensorExpr * const expr)                                \
        {                                                                       \
                expr->operands.push_back(Construct(op->a, expr->axis));         \
                expr->operands.push_back(Construct(op->b, expr->axis));         \
                for (const TensorExprPtr & operand                              \
                     : expr->operands)                                          \
                {                                                               \
                        expr->ordered_axis =                                    \
                                operand->ordered_axis.size() >                  \
                                expr->ordered_axis.size() ?                     \
                                operand->ordered_axis : expr->ordered_axis;     \
                }                                                               \
        }
        DEFINE_BINARY_OP_CSTR(Add)
        DEFINE_BINARY_OP_CSTR(Sub)
        DEFINE_BINARY_OP_CSTR(Mul)
        DEFINE_BINARY_OP_CSTR(Div)

        void _Construct(const Reduce * const op,
                        TensorExpr * const expr)
        {
                Array < IterVar > source_axis = expr->axis;
                for (const IterVar & reduce_axis
                     : op->axis)
                {
                        source_axis.push_back(reduce_axis);
                }
                expr->operands.push_back(
                        Construct(op->source[op->value_index], source_axis));
                expr->ordered_axis = expr->axis;
        }

#define DEFINE_IMM_CSTR(Imm)                                                    \
        void _Construct(const Imm * const imm,                                  \
                        TensorExpr * const expr)                                \
        {}
        DEFINE_IMM_CSTR(IntImm)
        DEFINE_IMM_CSTR(UIntImm)
        DEFINE_IMM_CSTR(FloatImm)

                
        TensorExprPtr Construct(const NodeRef & node,
                                const Array < IterVar > & axis)
        {
                static const FConstruct & fconstruct = cstrtable();
                auto node_tensorexpr_map_iter
                        = _node_tensorexpr_map.find(node.get());
                if (node.defined() && 
                    node_tensorexpr_map_iter == _node_tensorexpr_map.end())
                {
                        TensorExprPtr & tensor_expr
                                = _node_tensorexpr_map.emplace(node.get(), new TensorExpr())
                                  .first->second;
                        tensor_expr->op = node;
                        tensor_expr->axis = axis;
                        fconstruct(node, tensor_expr.get(), this);
                        return tensor_expr;
                }
                return node_tensorexpr_map_iter->second;
        }


        void Visit(const Tensor & tensor)
        {
                const FunctionBaseNode * tensor_op_func
                        = tensor->op.as < FunctionBaseNode > ();
                CHECK(tensor_op_func != nullptr);

                if (_tensor_expr_map.count(tensor_op_func))
                {
                        return;
                }
                TensorExprPtr & tensor_expr
                        = _tensor_expr_map.emplace(
                                tensor_op_func, nullptr).first->second;
                
                if (const ComputeOpNode * compute_op =
                    tensor->op.as < ComputeOpNode > ()) 
                {
                        for (const Tensor & input_tensor : 
                             compute_op->InputTensors())
                        {
                                Visit(input_tensor);
                        }
                        const Expr & body_stmt
                                = compute_op->body[tensor->value_index];
                        tensor_expr = Construct(body_stmt, compute_op->axis);
                }  // if (tensor->op.as < ComputeOpNode > ())
                else if (tensor->op.as < PlaceholderOpNode > ())
                {
                        tensor_expr = std::make_shared < TensorExpr >();
                        tensor_expr->op = tensor->op;
                }
                else
                {
                        LOG(FATAL) << "Unknown tensor op type: "
                                   << tensor->op->GetTypeKey();
                }  // if (tensor->op.as < ComputeOpNode > ())
                LOG(INFO) << "Tensor [" << tensor << "]";
                LOG(INFO) << tensor_expr->toString();
        }
};


#define DISPATCH_TO_CSTR(Op)                                                    \
        set_dispatch < Op > ([](const ObjectRef & node,                         \
                                TensorExpr * const expr,                        \
                                TensorExprConstructor * const v)                \
                {                                                               \
                        v->_Construct(static_cast < const Op * > (node.get()),  \
                                      expr);                                    \
                })
TVM_STATIC_IR_FUNCTOR(TensorExprConstructor, cstrtable)
        .DISPATCH_TO_CSTR(Call)
        .DISPATCH_TO_CSTR(Add)
        .DISPATCH_TO_CSTR(Sub)
        .DISPATCH_TO_CSTR(Mul)
        .DISPATCH_TO_CSTR(Div)
        .DISPATCH_TO_CSTR(Reduce)
        .DISPATCH_TO_CSTR(IntImm)
        .DISPATCH_TO_CSTR(UIntImm)
        .DISPATCH_TO_CSTR(FloatImm);


typedef std::pair < Operation, int >  OpValueIdxPair;

Expr OpValueIdxPair2BodyStmt(const OpValueIdxPair & op_valueidx_pair)
{
        const ComputeOpNode * compute_op
                = op_valueidx_pair.first.as < ComputeOpNode > ();
        CHECK(compute_op != nullptr);
        return compute_op->body[op_valueidx_pair.second];
}



struct BodyStmtAutoInliner : public IRMutator
{
        FunctionRef func; Array < Var > args; 
        std::pair < Operation, int > op_valueidx_pair;

        Expr Mutate_(const Call * op, const Expr & e) final
        {
                Expr expr = IRMutator::Mutate_(op, e);
                op = expr.as < Call > ();

                if (op->func == this->func)
                {
                        CHECK_EQ(op->value_index, 0);
                        CHECK_EQ(op->call_type, Call::CallType::Halide);

                        const ComputeOpNode * compute_op
                                = op_valueidx_pair.first.as < ComputeOpNode > ();
                        if (compute_op->reduce_axis.empty())
                        {
                                expr = OpValueIdxPair2BodyStmt(op_valueidx_pair);
                                CHECK_EQ(args.size(), op->args.size());

                                Map < Var, Expr > vmap;
                                for (size_t i = 0; i < this->args.size(); ++i)
                                {
                                        vmap.Set(this->args[i], op->args[i]);
                                }
                                expr = Substitute(Evaluate::make(expr), vmap)
                                       .as < Evaluate > ()->value;
                                return expr;
                        }
                        else 
                        {
                                // LOG(INFO) << "Calling " << op_valueidx_pair.first;
                                return Call::make(op->type,
                                                  op->name,
                                                  op->args,
                                                  op->call_type,
                                                  op_valueidx_pair.first);
                        }
                }           
                else
                {
                        LOG(INFO) << op->func << " != " << this->func;
                        return expr;
                }
        }

};  // BodyStmtAutoInliner


class TensorAutoInliner
{
private:
        std::vector < Tensor > _tensor_postorder;
        std::unordered_map < Tensor, std::unordered_set < Tensor > >
                _tensor_reverse_map;
        std::unordered_map < Tensor, Operation > _tensor_bodystmt_map;

        std::pair < Operation, int > GetOpValueIdxPair(const Tensor & tensor)
        {
                auto iter = _tensor_bodystmt_map.find(tensor);

                if (iter != _tensor_bodystmt_map.end())
                {
                        // LOG(INFO) << "Body Stmt: " << iter->second;
                        return std::make_pair(iter->second, 0);
                }
                else 
                {
                        // LOG(INFO) << "Body Stmt: " << tensor->op;
                        return std::make_pair(tensor->op,
                                              tensor->value_index);
                }
        }
        void PostOrderVisit(const Tensor & tensor)
        {
                if (std::find(_tensor_postorder.begin(),
                              _tensor_postorder.end(), tensor) !=
                    _tensor_postorder.end())
                {
                        return;
                }
                if (const ComputeOpNode * compute_op =
                    tensor->op.as < ComputeOpNode > ()) 
                {
                        for (const Tensor & input_tensor :
                             compute_op->InputTensors())
                        {
                                PostOrderVisit(input_tensor);
                                _tensor_reverse_map[input_tensor].insert(tensor);
                        }
                }
                _tensor_postorder.push_back(tensor);
        }
public:
        void Mutate(Tensor * const ptensor)
        {
                PostOrderVisit(*ptensor);

                // for (const Tensor & t : _tensor_postorder)
                // {
                //         LOG(INFO) << t;
                // }

                for (const Tensor & i : _tensor_postorder)
                {
                        std::unordered_set < Tensor > reverse_input_tensors
                                = _tensor_reverse_map[i];
                        for (const Tensor & o : reverse_input_tensors)
                        {
                                const ComputeOpNode
                                        * icompute_op = i->op.as < ComputeOpNode > (),
                                        * ocompute_op = o->op.as < ComputeOpNode > ();
                                if (icompute_op != nullptr && ocompute_op != nullptr)
                                {
                                        Array < Var > args;
                                        for (const IterVar & iv : icompute_op->axis)
                                        {
                                                args.push_back(iv->var);
                                        }
                                        BodyStmtAutoInliner inliner;

                                        inliner.func = i->op,
                                        inliner.args = args,
                                        inliner.op_valueidx_pair = GetOpValueIdxPair(i);

                                        Expr new_body = Simplify(
                                                inliner.Mutate(Evaluate::make(
                                                       OpValueIdxPair2BodyStmt(GetOpValueIdxPair(o))
                                                )).as < Evaluate > ()->value);
                                        // LOG(INFO) << "Inlining " << icompute_op->name 
                                        //           << " @" << icompute_op << " into "
                                        //           << ocompute_op->name << " @" << ocompute_op;
                                        // LOG(INFO) << "New Body: " << new_body;
                                        _tensor_bodystmt_map[o] = ComputeOpNode::make(
                                                ocompute_op->name,
                                                ocompute_op->tag, 
                                                ocompute_op->attrs,
                                                ocompute_op->axis,
                                                {new_body});
                                        // LOG(INFO) << "Operation " << _tensor_bodystmt_map[o]
                                        //           << " @" << _tensor_bodystmt_map[o].get();
                                }
                        }
                }
                auto iter = _tensor_bodystmt_map.find(*ptensor);
                if (iter != _tensor_bodystmt_map.end())
                {
                        *ptensor = iter->second.output(0);
                }
        }
};


}  // namespace anonymous


void _CSE(const Tensor & src, Tensor * const ptgt)
{
        if (!dmlc::GetEnv("USE_CSE", 0))
        {
                return;
        }
        // LOG(INFO) << PrintTensorRecursively(*ptgt);

        // TODO: We limit the scope of analysis to compute.gamma.grad, but will
        //       remove this limitation in later stages.
        TensorAutoInliner().Mutate(ptgt);
        // TensorExprConstructor().Visit(src);
        TensorExprConstructor().Visit(*ptgt);

        // LOG(INFO) << PrintTensorRecursively(*ptgt);

}


}  // namespace ir
}  // namespace tvm

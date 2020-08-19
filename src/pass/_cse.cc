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


#define CHECKPOINT_RETURN 1

static bool s_enable_ret_checkpoint = false;

#if CHECKPOINT_RETURN
#define RETURN(v)                                                               \
        do {                                                                    \
                bool ret = (v);                                                 \
                if (!ret && s_enable_ret_checkpoint)                                       \
                {                                                               \
                        LOG(INFO) << this->op << " != " << other.op;            \
                }                                                               \
                return ret;                                                     \
        } while(0)
#else
#define RETURN(v)  return (v)
#endif


struct TensorExpr
{
        NodeRef op;
        Array < IterVar > axis, ordered_axis;
        std::vector < std::shared_ptr < TensorExpr > > operands;
        
        std::string toString(const unsigned indent = 0) const
        {
                std::ostringstream strout;
                strout << "\n";
                for (unsigned i = 0; i < indent; ++i)
                {
                        strout << " ";
                }
                strout << op << " @" << op.get() << " ["
                        //   "axis=" << Axis2Str(axis) << ", " 
                          "ordered_axis=" << Axis2Str(ordered_axis) 
                       << "]";
                for (const auto & operand : operands)
                {
                        strout << operand->toString(indent + 2);
                }
                return strout.str();
        }

        using FCompare = NodeFunctor < bool(const ObjectRef &, const TensorExpr &,
                                            TensorExpr * const) >;
        static FCompare & cmptable() { static FCompare inst; return inst; }

        bool _Compare(const Call * const op, const TensorExpr & other)
        {
                // s_enable_ret_checkpoint = true;
                CHECK(op->call_type == Call::CallType::PureIntrinsic);
                bool operand_equal = (*this->operands[0]) == (*other.operands[0]);
                // s_enable_ret_checkpoint = false;
                RETURN(operand_equal);
        }

        bool _Compare(const PlaceholderOpNode * const op,
                      const TensorExpr & other)
        {
                const PlaceholderOpNode * other_op
                        = other.op.as < PlaceholderOpNode > ();
                CHECK(other_op != nullptr);
                RETURN(op == other_op);
        }

#define DEFINE_BINARY_OP_COMMUTATIVE_COMPARE(Op)                                \
        bool _Compare(const Op * const op, const TensorExpr & other)            \
        {                                                                       \
                CHECK(this->operands.size() == 2);                              \
                CHECK(other.operands.size() == 2);                              \
                RETURN(((*this->operands[0]) == (*other.operands[0]) &&         \
                        (*this->operands[1]) == (*other.operands[1])) ||        \
                       ((*this->operands[0]) == (*other.operands[1]) &&         \
                        (*this->operands[1]) == (*other.operands[0])));         \
        }
#define DEFINE_BINARY_OP_NONCOMMUTATIVE_COMPARE(Op)                             \
        bool _Compare(const Op * const op, const TensorExpr & other)            \
        {                                                                       \
                CHECK(this->operands.size() == 2);                              \
                CHECK(other.operands.size() == 2);                              \
                RETURN(((*this->operands[0]) == (*other.operands[0]) &&         \
                        (*this->operands[1]) == (*other.operands[1])));         \
        }
        DEFINE_BINARY_OP_COMMUTATIVE_COMPARE(Add)
        DEFINE_BINARY_OP_NONCOMMUTATIVE_COMPARE(Sub)
        DEFINE_BINARY_OP_COMMUTATIVE_COMPARE(Mul)
        DEFINE_BINARY_OP_NONCOMMUTATIVE_COMPARE(Div)

private:
        enum class CommReducerOpType {C_Add, C_Mul, C_Unk};
        static CommReducerOpType
        InferCommReducerOpType(const CommReducer & comm_reducer)
        {
                if (comm_reducer->lhs.size() != 1 ||
                    comm_reducer->rhs.size() != 1 || 
                    comm_reducer->result.size() != 1 || 
                    comm_reducer->identity_element.size() != 1)
                {
                        return CommReducerOpType::C_Unk;
                }
#define CHECK_COMM_REDUCER_OP_TYPE(Op, identity_value)                          \
        {                                                                       \
                const Op * op = comm_reducer->result[0].as < Op > ();           \
                const Variable * op_a = op->a.as < Variable > (),               \
                               * op_b = op->b.as < Variable > (),               \
                               * reducer_lhs = comm_reducer->lhs[0].as < Variable > (),  \
                               * reducer_rhs = comm_reducer->rhs[0].as < Variable > ();  \
                const FloatImm * identity_element                               \
                        = comm_reducer->identity_element[0].as < FloatImm > (); \
                if (op != nullptr &&                                            \
                    op_a == reducer_lhs &&                                      \
                    op_b == reducer_rhs &&                                      \
                    identity_element != nullptr &&                              \
                    identity_element->value == identity_value)                  \
                {                                                               \
                        return CommReducerOpType::C_ ## Op;                     \
                }                                                               \
        }
                CHECK_COMM_REDUCER_OP_TYPE(Add, 0)
                CHECK_COMM_REDUCER_OP_TYPE(Mul, 1)
                LOG(INFO) << "Finished checking";
                return CommReducerOpType::C_Unk;
        }
        static std::vector < size_t >
        InferOrderedReduceAxis(const Array < IterVar > & source_ordered_axis,
                               const Array < IterVar > & reduce_axis)
        {
                std::vector < size_t > ret;

                for (size_t i = 0; i < source_ordered_axis.size(); ++i)
                {
                        for (const IterVar & iv : reduce_axis)
                        {
                                LOG(INFO) << source_ordered_axis << " vs. "
                                          << reduce_axis;
                                if (source_ordered_axis[i].same_as(iv))
                                {
                                        ret.push_back(i);
                                }
                        }
                }
                return ret;
        }
        enum class ReduceCondType {C_Always, C_Unk};
        static ReduceCondType
        InferReduceCondType(const Expr & cond)
        {
                const UIntImm * cond_as_uint = cond.as < UIntImm > ();
                if (cond_as_uint != nullptr &&
                    cond_as_uint->value == 1)
                {
                        return ReduceCondType::C_Always;
                }
                return ReduceCondType::C_Unk;
        }

public:
        bool _Compare(const Reduce * const op,
                      const TensorExpr & other)
        {
                // Two reductions are considered the same if they are the same 
                // in terms of ALL of the following aspects
                // 
                //   - CommReducer
                //     - OpType
                //     - IdentityElement
                //   - Source
                //   - Axis
                //   - Condition
                const Reduce * other_op = other.op.as < Reduce > ();
                CHECK(other_op != nullptr);

                if (op == other_op)
                {
                        RETURN(true);
                }

                CHECK(this->operands.size() == 1);
                CHECK(other.operands.size() == 1);
                bool same_combiner, same_source,
                     same_ordered_reduce_axis = true,
                     same_condition;

                CommReducerOpType
                        lhs_comm_reducer_op_type = InferCommReducerOpType(op->combiner),
                        rhs_comm_reducer_op_type = InferCommReducerOpType(other_op->combiner);
                same_combiner = lhs_comm_reducer_op_type == rhs_comm_reducer_op_type &&
                                lhs_comm_reducer_op_type != CommReducerOpType::C_Unk;
                same_source = (*this->operands[0]) == (*other.operands[0]);
                std::vector < size_t >
                        lhs_ordered_reduce_axis = InferOrderedReduceAxis(
                                this->operands[0]->ordered_axis, op->axis),
                        rhs_ordered_reduce_axis = InferOrderedReduceAxis(
                                other.operands[0]->ordered_axis, other_op->axis);
                if (lhs_ordered_reduce_axis.size() != rhs_ordered_reduce_axis.size() || 
                    lhs_ordered_reduce_axis.size() == 0)
                {
                        RETURN(false);
                }
                for (size_t i = 0; i < lhs_ordered_reduce_axis.size(); ++i)
                {
                        if (lhs_ordered_reduce_axis[i] !=
                            rhs_ordered_reduce_axis[i])
                        {
                                same_ordered_reduce_axis = false;
                        }
                }
                ReduceCondType
                        lhs_reduce_cond_type = InferReduceCondType(op->condition),
                        rhs_reduce_cond_type = InferReduceCondType(other_op->condition);
                same_condition = lhs_reduce_cond_type == rhs_reduce_cond_type &&
                                 lhs_reduce_cond_type != ReduceCondType::C_Unk;

                // LOG(INFO) << same_combiner;
                // LOG(INFO) << same_source;
                // LOG(INFO) << same_ordered_reduce_axis;
                // LOG(INFO) << same_condition;
                // if (same_combiner && 
                //     same_source && 
                //     same_ordered_reduce_axis &&
                //     same_condition)
                // {
                //         LOG(INFO) << GetRef < Expr > (op) << " == " << other.op;
                // }
                // else 
                // {
                //         LOG(INFO) << GetRef < Expr > (op) << " != " << other.op;
                // }
                RETURN(same_combiner && 
                       same_source && 
                       same_ordered_reduce_axis &&
                       same_condition);
        }

#define DEFINE_IMM_COMPARE(Imm)                                                 \
        bool _Compare(const Imm * const imm, const TensorExpr & other)          \
        {                                                                       \
                const Imm * other_imm = other.op.as < Imm > ();                 \
                CHECK(other_imm != nullptr);                                    \
                RETURN(imm->value == other_imm->value);                         \
        }
        DEFINE_IMM_COMPARE(IntImm)
        DEFINE_IMM_COMPARE(UIntImm)
        DEFINE_IMM_COMPARE(FloatImm)


        bool operator==(const TensorExpr & other)
        {
                static const FCompare & fcompare = cmptable();
                if (this->op.defined() &&
                    other.op.defined())
                {
                        return fcompare(this->op, other, this);
                }
                return false;
        }
};
typedef std::shared_ptr < TensorExpr >  TensorExprPtr;


#define DISPATCH_TO_CMP(Op)                                                     \
        set_dispatch < Op > ([](const ObjectRef & node, const TensorExpr & other,  \
                                TensorExpr * const _this)                       \
                             ->bool                                             \
                {                                                               \
                        if (node->type_index() != other.op->type_index())       \
                        {                                                       \
                                return false;                                  \
                        }                                                       \
                        return _this->_Compare(static_cast < const Op * > (node.get()),  \
                                               other);                                   \
                })
TVM_STATIC_IR_FUNCTOR(TensorExpr, cmptable)
        .DISPATCH_TO_CMP(Call)
        .DISPATCH_TO_CMP(PlaceholderOpNode)
        .DISPATCH_TO_CMP(Add)
        .DISPATCH_TO_CMP(Sub)
        .DISPATCH_TO_CMP(Mul)
        .DISPATCH_TO_CMP(Div)
        .DISPATCH_TO_CMP(Reduce)
        .DISPATCH_TO_CMP(IntImm)
        .DISPATCH_TO_CMP(UIntImm)
        .DISPATCH_TO_CMP(FloatImm);


class CSEMutator;

class TensorExprConstructor
{
private:
        friend class CSEMutator;

        std::unordered_map < const FunctionBaseNode *,
                             TensorExprPtr > _tensor_expr_map;
        std::unordered_map < const Node *, 
                             TensorExprPtr > _node_tensor_expr_map;


        static Array < IterVar >
        ArgsToOrderedAxis(const Array < Expr > & args,
                          const Array < IterVar > & axis)
        {
                Array < IterVar > ordered_axis (args.size(), IterVar(nullptr));

                for (size_t arg_idx = 0; arg_idx < args.size(); ++arg_idx)
                {
                        const Variable * var
                                = args[arg_idx].as < Variable > ();
                        for (const IterVar & iv : axis)
                        {
                                if (var == iv->var.get())
                                {
                                        ordered_axis.Set(arg_idx, iv);
                                }
                        }
                }
                return ordered_axis;
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
                        expr->ordered_axis = ArgsToOrderedAxis(op->args, expr_axis);
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
                auto node_tensor_expr_map_iter
                        = _node_tensor_expr_map.find(node.get());
                if (node.defined() && 
                    node_tensor_expr_map_iter == _node_tensor_expr_map.end())
                {
                        TensorExprPtr & tensor_expr
                                = _node_tensor_expr_map.emplace(node.get(), new TensorExpr())
                                  .first->second;
                        tensor_expr->op = node;
                        tensor_expr->axis = axis;
                        fconstruct(node, tensor_expr.get(), this);
                        return tensor_expr;
                }
                return node_tensor_expr_map_iter->second;
        }


        const TensorExprPtr & Visit(const Tensor & tensor)
        {
                const FunctionBaseNode * tensor_op_func
                        = tensor->op.as < FunctionBaseNode > ();
                CHECK(tensor_op_func != nullptr);
                auto tensor_expr_iter = _tensor_expr_map.find(tensor_op_func);

                if (tensor_expr_iter != _tensor_expr_map.end())
                {
                        return tensor_expr_iter->second;
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
                        tensor_expr = std::make_shared < TensorExpr > ();
                        tensor_expr->op = tensor->op;
                }
                else
                {
                        LOG(FATAL) << "Unknown tensor op type: "
                                   << tensor->op->GetTypeKey();
                }  // if (tensor->op.as < ComputeOpNode > ())
                return tensor_expr;
        }


        std::pair < const Node *, TensorExprPtr >
        Find(const TensorExpr & expr) const
        {
                for (const auto & node_tensor_expr_pair 
                     : _node_tensor_expr_map)
                {
                        if ((*node_tensor_expr_pair.second) == expr)
                        {
                                return node_tensor_expr_pair;
                        }
                }
                return std::make_pair < const Node *, TensorExprPtr > (nullptr, nullptr);
        }
};


#define DISPATCH_TO_CSTR(Op)                                                    \
        set_dispatch < Op > ([](const ObjectRef & node,                         \
                                TensorExpr * const expr,                        \
                                TensorExprConstructor * const _this)            \
                {                                                               \
                        _this->_Construct(static_cast < const Op * > (node.get()),  \
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


class CSEMutator
{
private:
        TensorExprConstructor _src_tensor_expr_constr, 
                              _tgt_tensor_expr_constr;
public:
        CSEMutator(const Tensor & src)
        {
                LOG(INFO) << _src_tensor_expr_constr.Visit(src)->toString();
        }

        using FOptimize = NodeFunctor < Expr(const ObjectRef &, const Expr &,
                                             CSEMutator * const) >;
        static FOptimize & optable() { static FOptimize inst; return inst; }

        Expr _Optimize(const Call * const op, const Expr & expr)
        {
                if (op->call_type == Call::CallType::PureIntrinsic)
                {
                        Expr new_arg = Optimize(op->args[0]);
                        if (new_arg.same_as(op->args[0])) { return expr; }
                        else 
                        {
                                return Call::make(op->type, op->name,
                                                  {new_arg},
                                                  op->call_type,
                                                  op->func,
                                                  op->value_index);
                        }
                }
                return expr;
        }
#define DEFINE_BINARY_OP_OPTIMIZE(Op)                                           \
        Expr _Optimize(const Op * const op, const Expr & expr)                  \
        {                                                                       \
                Expr a = Optimize(op->a);                                       \
                Expr b = Optimize(op->b);                                       \
                if (a.same_as(op->a) && b.same_as(op->b))                       \
                {                                                               \
                        return expr;                                            \
                }                                                               \
                else                                                            \
                {                                                               \
                        return Op::make(a, b);                                  \
                }                                                               \
        }
        DEFINE_BINARY_OP_OPTIMIZE(Add)
        DEFINE_BINARY_OP_OPTIMIZE(Sub)
        DEFINE_BINARY_OP_OPTIMIZE(Mul)
        DEFINE_BINARY_OP_OPTIMIZE(Div)

        Expr _Optimize(const Reduce * const op,
                       const Expr & expr)
        {
                Expr source = Optimize(op->source[op->value_index]);

                if (source.same_as(op->source[op->value_index]))
                {
                        return expr;
                }
                else
                {
                        return Reduce::make(op->combiner,
                                            {source},
                                            op->axis,
                                            op->condition,
                                            0);
                }
        }

#define DEFINE_IMM_OPTIMIZE(Imm)                                                \
        Expr _Optimize(const Imm * const imm, const Expr & expr)                \
        {                                                                       \
                return expr;                                                    \
        }
        DEFINE_IMM_OPTIMIZE(IntImm)
        DEFINE_IMM_OPTIMIZE(UIntImm)
        DEFINE_IMM_OPTIMIZE(FloatImm)

        Expr Optimize(const Expr & expr)
        {
                static const FOptimize & foptimize = optable();
                static size_t s_feature_map_counter = 0;

                TensorExprPtr & tgt_tensor_expr
                        = _tgt_tensor_expr_constr.
                          _node_tensor_expr_map.at(expr.get());
                std::pair < const Node *, TensorExprPtr > 
                        src_node_tensor_expr_pair
                        = _src_tensor_expr_constr.Find(*tgt_tensor_expr);
                if (src_node_tensor_expr_pair.first != nullptr)
                {
                        LOG(INFO) << "feature_map_" + std::to_string(s_feature_map_counter)
                                  << "=" << expr;
                        LOG(INFO) << src_node_tensor_expr_pair.second->toString();


                        Array < Expr > feature_map_shape, args;
                        for (const IterVar & iv : 
                             tgt_tensor_expr->ordered_axis)
                        {
                                if (!iv.defined())
                                {
                                        return expr;
                                }
                                feature_map_shape.push_back(iv->dom->extent);
                                args.push_back(iv->var);
                        }
                        Operation feature_map_placeholder
                                = PlaceholderOpNode::make(
                                        "feature_map_"
                                        + std::to_string(s_feature_map_counter++),
                                        feature_map_shape,
                                        Float(32));
                        return Call::make(Float(32), 
                                          feature_map_placeholder->name,
                                          args,
                                          Call::CallType::Halide,
                                          feature_map_placeholder,
                                          0);
                }
                return foptimize(expr, expr, this);
        }

        void Optimize(Tensor * const ptensor)
        {
                LOG(INFO) << _tgt_tensor_expr_constr.Visit(*ptensor)->toString();
                const ComputeOpNode * compute_op
                        = (*ptensor)->op.as < ComputeOpNode > ();
                if (compute_op != nullptr)
                {
                        Expr new_body_stmt = Optimize(
                                compute_op->body[(*ptensor)->value_index]);
                        LOG(INFO) << new_body_stmt;
                        *ptensor = ComputeOpNode::make(
                                compute_op->name,
                                compute_op->tag, 
                                compute_op->attrs,
                                compute_op->axis,
                                {new_body_stmt}).output(0);
                }
        }
};


#define DISPATCH_TO_OPT(Op)                                                     \
        set_dispatch < Op > ([](const ObjectRef & node,                         \
                                const Expr & expr,                              \
                                CSEMutator * const _this)                       \
                             -> Expr                                            \
                {                                                               \
                        return _this->_Optimize(static_cast < const Op * > (node.get()),  \
                                                expr);                                    \
                })
TVM_STATIC_IR_FUNCTOR(CSEMutator, optable)
        .DISPATCH_TO_OPT(Call)
        .DISPATCH_TO_OPT(Add)
        .DISPATCH_TO_OPT(Sub)
        .DISPATCH_TO_OPT(Mul)
        .DISPATCH_TO_OPT(Div)
        .DISPATCH_TO_OPT(Reduce)
        .DISPATCH_TO_OPT(IntImm)
        .DISPATCH_TO_OPT(UIntImm)
        .DISPATCH_TO_OPT(FloatImm);


// =============================================================================
// AutoInliner
// =============================================================================s

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
                        // LOG(INFO) << op->func << " != " << this->func;
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

                                        Expr new_body_stmt = Simplify(
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
                                                {new_body_stmt});
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


// TODO: Make sure that the argument is changed to a list of tensors.
void _CSE(const Tensor & src, Tensor * const ptgt)
{
        if (!dmlc::GetEnv("USE_CSE", 0))
        {
                return;
        }
        // LOG(INFO) << PrintTensorRecursively(*ptgt);

        Tensor src_inlined = src;

        TensorAutoInliner().Mutate(&src_inlined);
        TensorAutoInliner().Mutate(ptgt);
        // TensorExprConstructor().Visit(src);
        // TensorExprConstructor().Visit(*ptgt);
        CSEMutator(src_inlined).Optimize(ptgt);

        // LOG(INFO) << PrintTensorRecursively(*ptgt);
}


}  // namespace ir
}  // namespace tvm

#include <functional>
#include <queue>
#include <unordered_set>

#include <dmlc/parameter.h>
#include <tvm/expr.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>
#include <tvm/operation.h>

#include "cse.h"
#include "autoinliner.h"
#include "zero_elimination.h"


namespace tvm {
namespace ir {
namespace {


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


}  // namespace anonymous


// TODO: Make sure that the argument is changed to a list of tensors.
std::pair < Tensor, Array < Tensor > >
CSE(const Tensor & output, const Array < Tensor > & in_args)
{
        if (!dmlc::GetEnv("USE_CSE", 0))
        {
                return std::make_pair(output, in_args);
        }

        Tensor output_inlined 
                = TensorAutoInliner().Mutate({output})[0];
        Array < Tensor > in_args_inlined
                = TensorAutoInliner().Mutate(in_args);
        // CSEMutator(src_inlined).Optimize(ptgt);
        return std::make_pair(output_inlined, in_args_inlined);
}


}  // namespace ir
}  // namespace tvm
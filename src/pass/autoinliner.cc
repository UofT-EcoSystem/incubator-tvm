#include "autoinliner.h"


Expr BodyStmtAutoInliner::Mutate_(const Call * op, const Expr & e)
{
        Expr expr = IRMutator::Mutate_(op, e);

        if (op->func == src_op)
        {
                CHECK_EQ(op->value_index, 0);
                CHECK_EQ(op->call_type, Call::CallType::Halide);

                const ComputeOpNode * compute_op
                        = src_op.as < ComputeOpNode > ();
                CHECK(compute_op != nullptr);

                if (compute_op->reduce_axis.empty())
                {
                        CHECK_EQ(op->args.size(), src_axis.size());

                        Map < Var, Expr > vmap;
                        for (size_t i = 0; i < src_axis.size(); ++i)
                        {
                                vmap.Set(src_axis[i], op->args[i]);
                        }
                        expr = Substitute(Evaluate::make(expr), vmap)
                               .as < Evaluate > ()->value;
                        return expr;
                }
                else
                {
                        // In the case in which the source Op matches the
                        // function name, but NOT a reduction. We still have to
                        // make an udpate to the CallNode to point it to the
                        // latest version of the body statement.
                        return Call::make(op->type,
                                          op->name,
                                          op->args,
                                          op->call_type,
                                          src_op);
                }
        }           
        else
        {
                return expr;
        }
}

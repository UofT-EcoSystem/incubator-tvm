#include <tvm/api_registry.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>


namespace tvm {
namespace ir {
namespace {

class IRPreOrderVisitor : public IRVisitor
{
protected:
        std::unordered_set < const Node * > _visited_nodes;
public:
        IRPreOrderVisitor() {}
        void Visit(const NodeRef & node) override
        {
                if (_visited_nodes.count(node.get()) != 0)
                {
                        return;
                }
                _visited_nodes.insert(node.get());
                LOG(INFO) << "Visiting [" << node->GetTypeKey() << "]" << node;
                IRVisitor::Visit(node);
        }
};  // class IRPreOrderVisitor


}   // namespace anonymous


Stmt CSE(Stmt stmt, Stmt src)
{
        LOG(INFO) << "stmt: " << stmt;
        LOG(INFO) << "src: "  << src;

        IRPreOrderVisitor().Visit(stmt);
        IRPreOrderVisitor().Visit(src);

        return stmt;
}


TVM_REGISTER_API("ir_pass.CSE").set_body_typed(CSE);

}  // namespace ir
}  // namespace tvm

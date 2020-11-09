#include "search_cluster.h"


namespace tvm {
        namespace ansor {
                namespace symtuning {


SearchCluster::SearchCluster(Array < SearchTask > tasks,
                             Array < Array < State > > sketches,
                             const int repr_idx)
{
        for (const SearchTask & task : tasks)
        {
                CHECK(task->target->target_name == "cuda")
                        << "Cluster searching is currently limited to "
                           "CUDA tasks ONLY";
        }
        CHECK(tasks.size() == sketches.size()) 
                << "The number of search tasks should be equal to "
                   "the number of sketches";
        ObjectPtr < SearchClusterNode > node = make_object < SearchClusterNode > ();
        node->tasks = std::move(tasks);
        node->sketches = std::move(sketches);
        node->repr_idx = std::move(repr_idx);
        data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(SearchClusterNode);

TVM_REGISTER_GLOBAL("ansor.SearchCluster")
        .set_body_typed(
                [](Array < SearchTask > tasks,
                   Array < Array < State > > sketches, int repr_idx)
                {
                        return SearchCluster(tasks, sketches, repr_idx);
                });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
        .set_dispatch < SearchClusterNode > (
                [](const ObjectRef & ref, ReprPrinter * p)
                {
                        const SearchClusterNode * const node
                                = static_cast < const SearchClusterNode * > (ref.get());
                        p->stream << "class [SearchCluster] with "
                                  << node->tasks.size() << " search tasks (repr_idx="
                                  << node->repr_idx << "), "
                                  << "all with initial sketch state {"
                                  << node->sketches[node->repr_idx][0] << "}";
                }
        );


                }  // namespace symtuning
        }  // namespace ansor
}  // namespace tvm

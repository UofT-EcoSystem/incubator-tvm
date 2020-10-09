#include "selective_tuning.h"

#include <vector>


namespace tvm {
        namespace ansor {


SearchCluster::SearchCluster(Array < SearchTask > tasks,
                             SearchTask representative)
{
        auto node = make_object < SearchClusterNode > ();
        node->tasks = std::move(tasks);
        node->representative = std::move(representative);
}


TVM_REGISTER_GLOBAL("ansor.SearchCluster")
        .set_body_typed(
                [](Array < SearchTask > tasks,
                   SearchTask representative)
                {
                        return SearchCluster(tasks, representative);
                });

// Sketch Generation Rules, defined in sketch_search_policy.cc
class RuleSkipStage;
class RuleAlwaysInline;
class RuleMultiLevelTiling;

std::vector < State >
ClusterSearchPolicyNode::GenerateSketch() 
{
        // std::vector < State > 
}


void 
ClusterSearchPolicyNode::SearchOneRound(
        std::vector < std::vector < State > > * const best_states,
        const int num_random_states,
        std::vector < std::vector < State > > * const random_states)
{
        best_states->clear(); random_states->clear();

        for (std::vector < State > & sketch : _sketch_caches)
        {
                if (sketch.empty())
                {
                        sketch = GenerateSketch();
                }  // if (sketch.empty())
        }  // for (sketch ∈ sketch_caches)
        // sample the initial population
        std::vector < State > init_population;
}


Array < State >
ClusterSearchPolicyNode::Search(
        SearchCluster cluster, ProgramMeasurer measurer,
        const int num_trials,
        const int early_stopping,
        const int num_measures_per_iter,
        Array < SearchCallback > pre_search_callbacks)
{
        std::vector < std::vector < State > > 
                best_states  (cluster->tasks.size()),
                random_states(cluster->tasks.size());
        this->cur_cluster = cluster;

        // if (num_trials <= 1) 
        // {
                SearchOneRound(&best_states, 0, &random_states);

                std::vector < State > best_state_per_task(cluster->tasks.size());

                for (size_t tidx = 0;
                     tidx < cluster->tasks.size(); ++tidx)
                {
                        best_state_per_task[tidx] = best_states[tidx][0];
                }
                return best_state_per_task;
        // }
        // else  // if (n_trails > 1)
        // {
        //         LOG(FATAL) << "NOT Implemented";
        // }
}

        }  // namespace ansor
}  // namespace tvm

#include "selective_tuning.h"

#include <vector>

#include "transform_step.h"


namespace tvm {
        namespace ansor {


SearchCluster::SearchCluster(Array < SearchTask > tasks,
                             Array < Array < State > > sketches,
                             const int representative_idx)
{
        for (const SearchTask & task : tasks)
        {
                CHECK(task->target->target_name == "cuda")
                        << "Cluster searching is currently limited to "
                           "CUDA tasks ONLY";
        }
        ObjectPtr < SearchClusterNode > node = make_object < SearchClusterNode > ();
        node->tasks = std::move(tasks);
        node->sketches = std::move(sketches);
        node->representative_idx = std::move(representative_idx);
        data_ = std::move(node);
}


TVM_REGISTER_GLOBAL("ansor.SearchCluster")
        .set_body_typed(
                [](Array < SearchTask > tasks,
                   Array < Array < State > > sketches, int representative_idx)
                {
                        return SearchCluster(tasks, sketches, representative_idx);
                });


class ClusterSplitFactorMemo
{
public:
        using QueryKey = std::tuple < int, int, int >;

        const std::vector < std::vector < PrimExpr > > &
        GetFactorizationSchemes(int extent, int n_lengths, int max_innermost_factor);
        const std::vector < int > & GetFactors(int n);
};  // class ClusterSplitFactorizationMemo


int
ClusterSearchPolicyNode::InitPopulationFillTileSize(
        State * const state)
{
        for (size_t step_id = 0; step_id < (*state)->transform_steps.size();
             ++step_id)
        {
                if (const SplitStepNode * const split_step =
                    (*state)->transform_steps[step_id].as < SplitStepNode > ())
                {
                        bool defined = true;

                        for (const PrimExpr & len : split_step->lengths)
                        {
                                if (!len.defined())
                                {
                                        defined = false;
                                }
                        }
                        if (defined)
                        {
                                continue;
                        }
                        
                }
        }  // for (step_id ∈ range((*state)->transform_steps.size()))
}


void
ClusterSearchPolicyNode::SampleInitPopulation(
        const size_t out_size,
        std::vector < State > * const out_states)
{
        size_t failed_attempts = 0;
        while (out_states->size() < out_size && 
               failed_attempts < out_size)
        {
                State tmp_state = 
                        cur_cluster->shared_sketch[
                                _rng() % (cur_cluster->shared_sketch.size())
                        ];
                InitPopulationFillTileSize(&tmp_state);

                if (InitPopulationThreadBind(&tmp_state))
                {
                        failed_attempts += 1;
                        continue;
                }
                InitPopulationUnroll(&tmp_state);
                out_states->push_back(std::move(tmp_state));
        }
}


void 
ClusterSearchPolicyNode::SearchOneRound(
        std::vector < std::vector < State > > * const best_states,
        const int num_random_states,
        std::vector < std::vector < State > > * const random_states)
{
        best_states->clear(); random_states->clear();


        size_t num_use_measured
                = std::min(_measured_states_vec.size(),
                           static_cast < size_t > (
                                   C_EVOLUTIONARY_SEARCH_USE_MEASURED_RATIO *
                                   C_EVOLUTIONARY_SEARCH_POPULATION));
        // sample the initial population
        std::vector < State > init_population;
        SampleInitPopulation(C_EVOLUTIONARY_SEARCH_POPULATION - num_use_measured,
                             &init_population);
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

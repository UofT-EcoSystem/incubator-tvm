#include "selective_tuning.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "utils.h"
#include "transform_step.h"


namespace tvm {
        namespace ansor {


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


TVM_REGISTER_GLOBAL("ansor.SearchCluster")
        .set_body_typed(
                [](Array < SearchTask > tasks,
                   Array < Array < State > > sketches, int repr_idx)
                {
                        return SearchCluster(tasks, sketches, repr_idx);
                });

const ClusterSplitFactorizationCache::VT &
ClusterSplitFactorizationCache::GetFactorizationSchemes(
        const ClusterExtentsT & extents,
        const int num_lengths,
        const int max_innermost_factor)
{
        KT k = std::make_tuple(extents, num_lengths, max_innermost_factor);
        auto iter = _cache.find(k);
        if (iter != _cache.end())
        {
                return iter->second;
        }
        _num_lengths = num_lengths;
        _max_innermost_factor = max_innermost_factor;
        _working_stack.assign(
                num_lengths,
                std::vector < PrimExpr > (extents.size(), PrimExpr()));
        _ret = &_cache[k];
        DFSEnumerate(extents);
        return *_ret;
}


void
ClusterSplitFactorizationCache::DFSEnumerate(
        const ClusterExtentsT & extents,
        const int depth)
{
        if (depth == _num_lengths)
        {
                if (std::all_of(_working_stack.back().begin(),
                                _working_stack.back().end(),
                                [this](const PrimExpr & e)
                                {
                                        return e.as < IntImmNode > ()->value <= 
                                               this->_max_innermost_factor;
                                }))
                {
                        _ret->push_back(_working_stack);
                }
        }
        else  // if (depth != _num_lengths)
        {
                for (const std::vector < int > & cluster_factors :
                     GetFactors(extents))
                {
                        ClusterExtentsT remainder(extents.size());
                        for (size_t tidx = 0; tidx < cluster_factors.size(); ++tidx)
                        {
                                _working_stack[depth][tidx]
                                        = PrimExpr(cluster_factors[tidx]);
                                
                        }
                        DFSEnumerate(depth + 1, );
                }
        }  // if (depth == _num_lengths)
}


int
ClusterSearchPolicyNode::InitPopulationFillTileSize(
        State * const repr_state,
        std::vector < State > * const states)
{
        for (size_t step_id = 0; step_id < (*repr_state)->transform_steps.size();
             ++step_id)
        {
                if (const SplitStepNode * const repr_split_step =
                    (*repr_state)->transform_steps[step_id].as < SplitStepNode > ())
                {
                        bool defined = true;

                        for (const PrimExpr & len : repr_split_step->lengths)
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
                        std::vector < int > extents;
                        for (const State & state : (*states))
                        {
                                const SplitStepNode * const split_step
                                        = state->transform_steps[step_id].as < SplitStepNode > ();
                                CHECK(split_step != nullptr)
                                        << "Representative is performing a split step "
                                           "but its dependents are NOT";
                                extents.push_back(GetIntImm(split_step->extent));
                        }
                        const std::vector < std::vector < PrimExpr > > & candidate_lens =
                                split_memo->GetFactorizationSchemes(
                                        extents,
                                        repr_split_step->lengths.size(),
                                        cur_cluster->tasks[cur_cluster->repr_idx]->hardware_params->max_innermost_split_factor);
                }
        }  // for (step_id ∈ range((*state)->transform_steps.size()))
        return 0;
}


void
ClusterSearchPolicyNode::SampleInitPopulation(
        const size_t out_size,
        std::vector < std::vector < State > > * const out_states)
{
        size_t failed_attempts = 0;
        while (out_states->size() < out_size && 
               failed_attempts < out_size)
        {
                size_t rand_sketch_idx = _rng() % (cur_cluster->sketches[0].size());
                State tmp_repr_state
                        = cur_cluster->sketches[cur_cluster->repr_idx][rand_sketch_idx];
                std::vector < State > tmp_states(cur_cluster->tasks.size());
                for (const Array < State > & sketch : cur_cluster->sketches)
                {
                        tmp_states.push_back(sketch[rand_sketch_idx]);
                }
                InitPopulationFillTileSize(&tmp_repr_state,
                                           &tmp_states);
                if (InitPopulationThreadBind(&tmp_repr_state, &tmp_states))
                {
                        failed_attempts += 1;
                        continue;
                }
                InitPopulationUnroll(&tmp_repr_state, &tmp_states);
                out_states->push_back(std::move(tmp_states));
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
        std::vector < std::vector < State > > init_population;
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

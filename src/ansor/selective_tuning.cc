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

const ClusterSplitFactorCache::VT &
ClusterSplitFactorCache::GetFactorizationSchemes(
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
ClusterSplitFactorCache::DFSEnumerate(
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
                for (const int cluster_factor : GetFactors(extents))
                {
                        ClusterExtentsT remainder(extents.size());
                        for (size_t tidx = 0; tidx < extents.size(); ++tidx)
                        {
                                _working_stack[depth][tidx]
                                        = PrimExpr(cluster_factor);
                                remainder[tidx] = extents[tidx] / cluster_factor;
                        }
                        DFSEnumerate(remainder, depth + 1);
                }  // for (cluster_factor ∈ GetFactors(extents))
        }  // if (depth == _num_lengths)
}


const ClusterFactorsT &
ClusterSplitFactorCache::GetFactors(const ClusterExtentsT & extents)
{
        auto iter = _factor_cache.find(extents);
        if (iter != _factor_cache.end())
        {
                return iter->second;
        }
        ClusterFactorsT & cluster_factors = _factor_cache[extents];
        const int min_extent = *std::min_element(extents.begin(), extents.end());

        for (int f = 1; f < min_extent; ++f)
        {
                if (std::all_of(extents.begin(), extents.end(),
                                [f](const int extent)
                                {
                                        return extent % f;
                                }))
                {
                        cluster_factors.push_back(f);
                }
        }
        // There is no need to sort the factors because they are already
        // inserted in order.
        return cluster_factors;
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
                        const int num_lengths = repr_split_step->lengths.size();
                        for (const State & state : (*states))
                        {
                                const SplitStepNode * const split_step
                                        = state->transform_steps[step_id].as < SplitStepNode > ();
                                CHECK(split_step != nullptr)
                                        << "Representative is performing a split "
                                           "step but its dependents are NOT";
                                CHECK(split_step->lengths.size() == num_lengths)
                                        << "Representative does not share the "
                                           "same split lengths with its dependents";
                                extents.push_back(GetIntImm(split_step->extent));
                        }
                        
                        const ClusterSplitFactorCache::VT & candidates =
                                _split_factor_cache.GetFactorizationSchemes(
                                        extents, num_lengths,
                                        cur_cluster->tasks[cur_cluster->repr_idx]
                                                   ->hardware_params
                                                   ->max_innermost_split_factor);

                        // make sure that the dimensions are correct
                        for (const ClusterSplitFactorCache::StackT &
                             candidate : candidates)
                        {
                                CHECK(candidate.size() == num_lengths);
                                for (const std::vector < PrimExpr > &
                                     cluster_factor : candidate)
                                {
                                        CHECK(cluster_factor.size() == states->size());
                                }
                        }  // for (candidate ∈ candidates)

                        int rand_cidx = _rng() % candidates.size();
                        for (int tidx = 0; tidx < extents.size(); ++tidx)
                        {
                                std::vector < PrimExpr > lengths;
                                for (int lidx = 0; lidx < num_lengths; ++lidx)
                                {
                                        lengths.push_back(candidates[rand_cidx][lidx][tidx]);
                                }
                                StateNode * pstate = (*states)[tidx].CopyOnWrite();
                                const SplitStepNode * const split_step
                                        = (*states)[tidx]->transform_steps[step_id].as < SplitStepNode > ();
                                pstate->transform_steps[step_id]
                                        = SplitStep(split_step->stage_id,
                                                    split_step->iter_id,
                                                    split_step->extent, lengths,
                                                    split_step->inner_to_outer);
                        }
                }
        }  // for (step_id ∈ range((*state)->transform_steps.size()))
        return 0;
}


void
ClusterSearchPolicyNode::SampleInitPopulation(
        const int out_size,
        std::vector < std::vector < State > > * const out_states)
{
        int failed_attempts = 0;
        while (out_states->size() < out_size && 
               failed_attempts < out_size)
        {
                int rand_sketch_idx = _rng() % (cur_cluster->sketches[0].size());
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

        int num_use_measured
                = std::min(static_cast < int > (_measured_states_vec.size()),
                           static_cast < int > (
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

                for (int tidx = 0; tidx < cluster->tasks.size(); ++tidx)
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

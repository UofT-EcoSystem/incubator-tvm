#include "cluster_search_policy.h"

#include <algorithm>

#include "../search_policy/sketch_search_policy.h"
#include "../search_policy/utils.h"
#include "../transform_step.h"


namespace tvm {
        namespace ansor {


constexpr double ClusterSearchPolicyNode::C_EPS_GREEDY;
constexpr int ClusterSearchPolicyNode::C_EVOLUTIONARY_SEARCH_POPULATION;
constexpr int ClusterSearchPolicyNode::C_EVOLUTIONARY_SEARCH_NUM_ITERS;
constexpr double ClusterSearchPolicyNode::C_EVOLUTIONARY_SEARCH_MUTATION_PROB;
constexpr double ClusterSearchPolicyNode::C_EVOLUTIONARY_SEARCH_CROSS_OVER_RATIO;
constexpr double ClusterSearchPolicyNode::C_EVOLUTIONARY_SEARCH_USE_MEASURED_RATIO;
constexpr int ClusterSearchPolicyNode::C_GPU_AUTO_UNROLL_CONFIGS[];
constexpr const char * ClusterSearchPolicyNode::C_GPU_MULTI_LEVEL_TILING_STRUCTURE;
constexpr bool ClusterSearchPolicyNode::C_DISABLE_CHANGE_COMPUTE_LOCATION;


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
        
        DEBUG_LOG_VEC(extents);
        
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
                                [this](const PrimExpr & e) -> bool
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
                        for (size_t task_idx = 0; task_idx < extents.size(); ++task_idx)
                        {
                                _working_stack[depth][task_idx]
                                        = PrimExpr(cluster_factor);
                                remainder[task_idx] = extents[task_idx] / cluster_factor;
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

        for (int f = 1; f <= min_extent; ++f)
        {
                if (std::all_of(extents.begin(), extents.end(),
                                [f](const int extent) -> int
                                {
                                        return (extent % f) == 0;
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
        std::vector < State > * const states)
{
        DEBUG_LOG_VAR(cur_cluster->repr_idx);
        DEBUG_LOG_VAR(states->size());

        State & repr_state = (*states)[cur_cluster->repr_idx];
        
        // DEBUG_LOG_VAR(repr_state);
        DEBUG_LOG_VAR(repr_state->transform_steps.size());
        
        for (size_t step_idx = 0; step_idx < repr_state->transform_steps.size();
             ++step_idx)
        {
                if (const SplitStepNode * const repr_split_step =
                    repr_state->transform_steps[step_idx].as < SplitStepNode > ())
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
                                LOG(INFO) << "Directly returning as split length "
                                             "has already been defined";
                                continue;
                        }
                        std::vector < int > extents;
                        const size_t num_lengths = repr_split_step->lengths.size();
                        for (const State & state : (*states))
                        {
                                const SplitStepNode * const split_step
                                        = state->transform_steps[step_idx].as < SplitStepNode > ();
                                CHECK(split_step != nullptr)
                                        << "Representative is performing a split "
                                           "step but its dependents are NOT";
                                CHECK(split_step->lengths.size() == num_lengths)
                                        << "Representative does not share the "
                                           "same split lengths with its dependents";
                                extents.push_back(GetIntImm(split_step->extent));
                        }
                        
                        LOG(INFO) << "Attempting to get the factorization schemes "
                                  << "for extents=" << toString(extents) << ", "
                                  << "length=" << num_lengths;
                        
                        const ClusterSplitFactorCache::VT & candidates =
                                _split_factor_cache.GetFactorizationSchemes(
                                        extents, num_lengths,
                                        cur_cluster->tasks[cur_cluster->repr_idx]
                                                   ->hardware_params
                                                   ->max_innermost_split_factor);
                        
                        LOG(INFO) << "Finished getting the factorization schemes";
                        CHECK(candidates.size() != 0)
                                << "Failed to get any factorization scheme "
                                   "for extents=" << toString(extents) << ", "
                                << "length=" << num_lengths;

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

                        LOG(INFO) << "Randomly picking the factorization schemes";

                        size_t rand_candidate_idx = _rng() % candidates.size();
                        for (size_t task_idx = 0; task_idx < extents.size(); ++task_idx)
                        {
                                std::vector < PrimExpr > lengths;
                                for (size_t len_idx = 0; len_idx < num_lengths; ++len_idx)
                                {
                                        lengths.push_back(candidates[rand_candidate_idx][len_idx][task_idx]);
                                }

                                DEBUG_LOG_VEC(lengths);

                                StateNode * pstate = (*states)[task_idx].CopyOnWrite();
                                const SplitStepNode * const split_step
                                        = (*states)[task_idx]->transform_steps[step_idx].as < SplitStepNode > ();
                                pstate->transform_steps[step_idx]
                                        = SplitStep(split_step->stage_id,
                                                    split_step->iter_id,
                                                    split_step->extent, lengths,
                                                    split_step->inner_to_outer);
                        }
                }
        }  // for (step_idx ∈ range((*state)->transform_steps.size()))
        return 0;
}


int
ClusterSearchPolicyNode::InitPopulationThreadBind(
        std::vector < State > * const states)
{
        // The vector of ints is used to make sure that all search tasks undergo
        // exactly the same binding at the same stage index.
        std::vector < std::vector < int > > 
                fused_iter_ext_le_wrap_size(cur_cluster->tasks.size()),
                fused_iter_ext_gt_wrap_size(cur_cluster->tasks.size());
        /// @note The assumption that we have here is that every state will
        ///       always share the same thread binding.
        for (size_t task_idx = 0; task_idx < cur_cluster->tasks.size(); ++task_idx)
        {
                const SearchTask & task = cur_cluster->tasks[task_idx];
                State & state = (*states)[task_idx];
                std::set < int > multi_level_tiling_root_set;
                for (size_t stage_idx = 0; stage_idx < state->stages.size();
                     ++stage_idx)
                {
                        if (NeedsMultilevelTiling(task, state, stage_idx))
                        {
                                const Stage & stage = state->stages[stage_idx];
                                if (stage->compute_at != kIter)
                                {
                                        CHECK(HasCrossThreadReduction(state, stage_idx));
                                        continue;
                                }
                                CHECK_EQ(stage->compute_at, kIter);
                                const auto attached_iter
                                        = state->attach_map->stage_to_attach_iter.find(stage_idx);
                                CHECK(attached_iter != state->attach_map->stage_to_attach_iter.end());
                                multi_level_tiling_root_set.insert(attached_iter->second.first);
                        }
                }  // for (stage_idx ∈ range(state->stages.size()))
                for (size_t stage_idx = 0; stage_idx < state->stages.size();
                     ++stage_idx)
                {
                        const Stage & stage = state->stages[stage_idx];
                        if (stage->compute_at == kInlined || 
                            stage->op_type == kPlaceholder)
                        {
                                continue;
                        }
                        // skip if this stage has already been annotated with
                        // threadIdx.x or tensorized
                        if (HasAnnotatedIter(stage, IteratorAnnotation::kThreadX) ||
                            HasAnnotatedIter(stage, IteratorAnnotation::kTensorized))
                        {
                                continue;
                        }
                        if (stage->compute_at == kRoot)
                        {
                                if (!multi_level_tiling_root_set.count(stage_idx))
                                {
                                        Iterator fused_iter;
                                        state = FuseAllOuterSpaceIterators(state, stage_idx, &fused_iter);

                                        if (GetExtent(fused_iter) <= task->hardware_params->warp_size)
                                        {
                                                state.bind_thread(stage_idx, fused_iter, kThreadX);
                                                fused_iter_ext_le_wrap_size[task_idx].push_back(stage_idx);
                                        }
                                        else
                                        {
                                                const auto & split_iters = state.split(
                                                        stage_idx, fused_iter,
                                                        {task->hardware_params->warp_size});
                                                state.bind_thread(stage_idx, split_iters[0], kBlockX);
                                                state.bind_thread(stage_idx, split_iters[1], kThreadX);
                                                fused_iter_ext_gt_wrap_size[task_idx].push_back(stage_idx);
                                        }
                                        continue;
                                }  // if (!multi_level_tiling_root_set.count(stage_idx))

                                const te::ComputeOpNode * const compute_op
                                        = stage->op.as < te::ComputeOpNode > ();
                                std::vector < Iterator > to_fuse;

                                // The remaining part deals with the thread
                                // binding for multi-level tiled stages.
                                int total_space_extent = 1;
                                for (const auto & i : compute_op->root_iter_vars())
                                {
                                        CHECK(i->dom.defined());
                                        const IntImmNode * const extent
                                                = i->dom->extent.as < IntImmNode >();
                                        total_space_extent *= extent->value;
                                }
                                DEBUG_LOG_VAR(total_space_extent);
                                // Check if the total space extent is too small for multi-level thread binding.
                                if (total_space_extent <= task->hardware_params->warp_size)
                                {
                                        for (const auto & it : state->stages[stage_idx]->iters)
                                        {
                                                if (it->iter_type == kReduce)
                                                {
                                                        break;
                                                }
                                                to_fuse.push_back(it);
                                        }
                                        const auto & fused = state.fuse(stage_idx, to_fuse);
                                        state.bind_thread(stage_idx, fused, kThreadX);
                                        continue;
                                }

                                // 1. Fuse the outermost space tile as blockIdx.
                                for (size_t i = 0; i < compute_op->axis.size(); ++i)
                                {
                                        const auto & iter = state->stages[stage_idx]->iters[i];
                                        if (!StrEndsWith(iter->name, ".0"))
                                        {
                                                break;
                                        }
                                        to_fuse.push_back(iter);
                                }
                                const auto & blockidx_iter = state.fuse(stage_idx, to_fuse);
                                state.bind_thread(stage_idx, blockidx_iter, kBlockX);

                                // 2. Fuse the second outermost space tile as vthread.
                                to_fuse.clear();
                                for (size_t i = 1; i < compute_op->axis.size() + 1; ++i)
                                {
                                        const auto & it = state->stages[stage_idx]->iters[i];
                                        if (!StrEndsWith(it->name, ".1"))
                                        {
                                                break;
                                        }
                                        to_fuse.push_back(state->stages[stage_idx]->iters[i]);
                                }
                                const auto & vthread_iter = state.fuse(stage_idx, to_fuse);
                                if (GetExtent(vthread_iter) > 
                                    task->hardware_params->max_vthread_extent)
                                {
                                        LOG(WARNING) << "vthread.extent=" << GetExtent(vthread_iter) << " > "
                                                     << task->hardware_params->max_vthread_extent;
                                        return -1;
                                }
                                state.bind_thread(stage_idx, vthread_iter, kVThread);

                                // 3. Fuse the third outermost space tile as threadIdx.
                                to_fuse.clear();
                                for (size_t i = 2; i < compute_op->axis.size() + 2; ++i)
                                {
                                        const auto & iter = state->stages[stage_idx]->iters[i];
                                        if (!StrEndsWith(iter->name, ".2"))
                                        {
                                                break;
                                        }
                                        to_fuse.push_back(state->stages[stage_idx]->iters[i]);
                                        LOG(INFO) << "Fusing " << state->stages[stage_idx]->iters[i] << " "
                                                     "of " << state->stages[stage_idx];
                                }

                                DEBUG_LOG_VEC(to_fuse);
                                // std::string dummy_string;
                                // std::cin >> dummy_string;


                                const auto & threadidx_iter = state.fuse(stage_idx, to_fuse);
                                if (GetExtent(threadidx_iter) < 
                                    task->hardware_params->warp_size)
                                {
                                        if (threadidx_iter->range.defined())
                                        {
                                                LOG(INFO) << threadidx_iter->range;
                                        }
                                        else
                                        {
                                                LOG(INFO) << "threadidx_iter range has NOT been defined";
                                        }
                                        LOG(WARNING) << "threadIdx.extent=" << GetExtent(threadidx_iter) << " < "
                                                     << task->hardware_params->warp_size;
                                        return -1;
                                }

                                DEBUG_LOG_VAR(GetExtent(threadidx_iter));
                                // std::string dummy_string;
                                // std::cin >> dummy_string;

                                state.bind_thread(stage_idx, threadidx_iter, kThreadX);
                        }
                        else if (stage->compute_at == kIter &&
                                 StrEndsWith(stage->op->name, ".shared"))
                        {
                                CHECK(stage->compute_at == kIter);
                                const auto & attached_iter
                                        = state->attach_map->stage_to_attach_iter.find(stage_idx);
                                CHECK(attached_iter != state->attach_map->stage_to_attach_iter.end());
                                std::vector < int > spatial_split_step_idxs;
                                GetSpaceSplitStepIds(state, attached_iter->second.first,
                                                     &spatial_split_step_idxs);
                                // fuse all iterators to do cooperative fetching
                                Iterator fused = state.fuse(stage_idx, state->stages[stage_idx]->iters);
                                // split out an extra iterator for vectorization
                                const auto & iters0 = state.split(
                                        stage_idx, fused, {1});
                                state.vectorize(stage_idx, iters0[1]);
                                const auto & iters1 = state.follow_fused_split(
                                        stage_idx, iters0[0],
                                        spatial_split_step_idxs, 1, true);
                                state.bind_thread(stage_idx, iters1[1], kThreadX);
                        }
                }  // for (stage_idx ∈ range(state->stages.size()))
        }  // for (task_idx ∈ range(states->size()))
        auto are_stage_idxs_all_same = 
                [](const std::vector < int > & lhs,
                   const std::vector < int > & rhs)
                {
                        if (lhs.size() != rhs.size())
                        {
                                return false;
                        }
                        for (size_t i = 0; i < lhs.size(); ++i)
                        {
                                if (lhs[i] != rhs[i])
                                {
                                        return false;
                                }
                        }
                        return true;
                };
        CHECK(std::all_of(fused_iter_ext_le_wrap_size.begin(),
                         fused_iter_ext_le_wrap_size.end(),
                         [&fused_iter_ext_le_wrap_size, are_stage_idxs_all_same]
                         (const std::vector < int > & stage_idxs)
                         {
                                return are_stage_idxs_all_same(stage_idxs, fused_iter_ext_le_wrap_size[0]);
                         }));
        CHECK(std::all_of(fused_iter_ext_gt_wrap_size.begin(),
                         fused_iter_ext_gt_wrap_size.end(),
                         [&fused_iter_ext_gt_wrap_size, are_stage_idxs_all_same]
                         (const std::vector < int > & stage_idxs)
                         {
                                return are_stage_idxs_all_same(stage_idxs, fused_iter_ext_gt_wrap_size[0]);
                         }));
        return 0;
}


int
ClusterSearchPolicyNode::InitPopulationUnroll(std::vector < State > * const states)
{
        size_t rand_auto_unroll_config = C_GPU_AUTO_UNROLL_CONFIGS[_rng() % 5];
        for (size_t task_idx = 0; task_idx < cur_cluster->tasks.size(); ++task_idx)
        {
                const SearchTask & task = cur_cluster->tasks[task_idx];
                State & state = (*states)[task_idx];
                for (size_t stage_idx = 0; stage_idx < state->stages.size();
                     ++stage_idx)
                {
                        const Stage & stage = state->stages[stage_idx];
                        if (stage->compute_at == kInlined || 
                            stage->op_type == kPlaceholder)
                        {
                                continue;
                        }
                        /// @note Special unroll policy is ignored.
                        bool annotate_auto_unroll = HasReduceIter(stage);
                        if (!NeedsMultilevelTiling(task, state, stage_idx) ||
                            HasRfactorStage(state, stage_idx))
                        {
                                annotate_auto_unroll = false;
                        }
                        if (annotate_auto_unroll)
                        {
                                state.pragma(stage_idx, state->stages[stage_idx]->iters[0],
                                             std::string("auto_unroll_max_step") + "$" + 
                                             std::to_string(rand_auto_unroll_config));
                        }
                }  // for (stage_idx ∈ range(state->stages.size()))
        }  // for (task_idx ∈ range(states->size()))
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
                int rand_sketch_idx = _rng() % (cur_cluster->sketches[0].size());
                std::vector < State > tmp_states;
                DEBUG_LOG_VAR(cur_cluster->sketches.size());
                for (const Array < State > & sketch : cur_cluster->sketches)
                {
                        tmp_states.push_back(sketch[rand_sketch_idx]);
                }
                LOG(INFO) << "Finished selecting the random sketch states";
                InitPopulationFillTileSize(&tmp_states);
                for (size_t task_idx = 0; task_idx < cur_cluster->tasks.size();
                     ++task_idx)
                {
                        tmp_states[task_idx]
                                = cur_cluster->tasks[task_idx]
                                             ->compute_dag.InferBound(tmp_states[task_idx]);
                }
                LOG(INFO) << "Finished initializing the tile sizes";
                if (InitPopulationThreadBind(&tmp_states))
                {
                        failed_attempts += 1;
                        continue;
                }
                LOG(INFO) << "Finished initializing the thread bindings";
                InitPopulationUnroll(&tmp_states);
                LOG(INFO) << "Finished initializing the unrolling factors";

                bool all_valid = true;
                for (size_t task_idx = 0; task_idx < cur_cluster->tasks.size(); ++task_idx)
                {
                        std::vector < float > pop_scores;
                        tmp_states[task_idx]
                                = cur_cluster->tasks[task_idx]
                                             ->compute_dag.InferBound(tmp_states[task_idx]);
                        _program_cost_model->Predict(cur_cluster->tasks[task_idx],
                                                     {tmp_states[task_idx]},
                                                     &pop_scores);
                        DEBUG_LOG_VAR(pop_scores[0]);
                        // std::string dummy_string;
                        // std::cin >> dummy_string;
                        if (pop_scores[0] <= -1e10)
                        {
                                all_valid = false;
                        }
                }
                if (all_valid)
                {
                        out_states->push_back(std::move(tmp_states));
                }
                else
                {
                        LOG(INFO) << "State pruned for failing the lowering test";
                        ++failed_attempts;
                }
        }
        LOG(INFO) << "Finished sampling the initial population";
        LOG(INFO) << "Available # of candidates: " << out_states->size();
        CHECK(out_states->size() != 0)
                << "Failed to sample the initial population";
}


void
ClusterSearchPolicyNode::RandomSampleStates(
        const std::vector < std::vector < State > > & init_population,
        const int num_measures,
        std::vector < std::vector < State > > * const best_states)
{
        size_t rand_init_population_idx = _rng() % init_population.size();
        best_states->clear();

        for (int i = 0; i < num_measures; ++i)
        {
                best_states->emplace_back(cur_cluster->tasks.size());
                for (size_t task_idx = 0;
                     task_idx < cur_cluster->tasks.size(); ++task_idx)
                {
                        best_states->back()[task_idx]
                                = init_population[rand_init_population_idx][task_idx];

                        if (i == 0)
                        {
                                std::vector < float > pop_scores;
                                State init_population_state_wbound
                                        = cur_cluster->tasks[task_idx]
                                                     ->compute_dag.InferBound(init_population[rand_init_population_idx][task_idx]);
                                _program_cost_model->Predict(cur_cluster->tasks[task_idx],
                                                     {init_population_state_wbound},
                                                     &pop_scores);
                                DEBUG_LOG_VAR(pop_scores[0]);
                        }
                }
        }
}


void 
ClusterSearchPolicyNode::SearchOneRound(
        std::vector < std::vector < State > > * const best_states,
        std::vector < std::vector < State > > * const random_states,
        const int num_random_states)
{
        best_states->clear(); random_states->clear();

        int num_use_measured
                = std::min(static_cast < int > (_measured_states_vec.size()),
                           static_cast < int > (
                                   C_EVOLUTIONARY_SEARCH_USE_MEASURED_RATIO *
                                   C_EVOLUTIONARY_SEARCH_POPULATION));
        // sample the initial population [* × cluster_size]
        std::vector < std::vector < State > > init_population;
        LOG(INFO) << "Sampling the initial population";
        SampleInitPopulation(C_EVOLUTIONARY_SEARCH_POPULATION - num_use_measured,
                             &init_population);
        RandomSampleStates(init_population,  3 * _num_measures_per_iter, best_states);
        RandomSampleStates(init_population, 10 *  num_random_states, random_states);
}


void 
ClusterSearchPolicyNode::PickStatesWithEpsGreedy(
        std::vector < std::vector < MeasureInput > > * const inputs,
        const std::vector < std::vector < State > > & best_states,
        const std::vector < std::vector < State > > & random_states,
        const int remaining_num_trials)
{
        const size_t num_best_states
                = _num_measures_per_iter - C_EPS_GREEDY * _num_measures_per_iter;
        size_t best_idx = 0, random_idx = 0;

        for (size_t inputs_size = 0;
             inputs_size
                     < static_cast < size_t > (std::min(_num_measures_per_iter, remaining_num_trials));
             ++inputs_size)
        {
                const std::vector < State > * states;
                bool has_best = best_idx < best_states.size(),
                     has_random = random_idx < random_states.size();
                if (inputs_size < num_best_states)
                {
                        if (has_best)
                        {
                                states = &best_states[best_idx++];
                        }
                        else if (has_random)
                        {
                                states = &random_states[random_idx++];
                        }
                        else
                        {
                                break;
                        }
                }
                else
                {
                        if (has_random)
                        {
                                states = &random_states[random_idx++];
                        }
                        else if (has_best)
                        {
                                states = &best_states[best_idx++];
                        }
                        else
                        {
                                break;
                        }
                }
                // check if it has already been measured
                bool all_states_measured = true;
                for (size_t task_idx = 0;
                     task_idx < cur_cluster->tasks.size(); ++task_idx)
                {
                        std::string state_str = (*states)[task_idx].ToStr();
                        if (!_measured_states_set[task_idx].count(state_str))
                        {
                                all_states_measured = false;
                        }
                }
                if (all_states_measured)
                {
                        continue;
                }
                for (size_t task_idx = 0;
                     task_idx < cur_cluster->tasks.size(); ++task_idx)
                {
                        std::string state_str = (*states)[task_idx].ToStr();
                        _measured_states_set[task_idx].insert(state_str);
                        _measured_states_vec[task_idx].push_back((*states)[task_idx]);
                }
                for (size_t task_idx = 0;
                     task_idx < cur_cluster->tasks.size(); ++task_idx)
                {
                        (*inputs)[task_idx].emplace_back(cur_cluster->tasks[task_idx],
                                                      (*states)[task_idx]);
                }
        }  // while (ibatch->size() < min(num_measures_per_iter, remaining_num_trials))
}

using StatesScorePair = std::pair < std::vector < State >, float >;

bool operator>(const StatesScorePair & lhs,
               const StatesScorePair & rhs)
{
        return lhs.second > rhs.second;
}


void
ClusterSearchPolicyNode::EvolutionarySearch(
        // [* × cluster_size]
        const std::vector < std::vector < State > > & population,
        const int num_best_states,
        std::vector < std::vector < State > > * const best_states)
{
        // [cluster_size × *]
        std::vector < std::vector < State > >
                ping_buf(cur_cluster->tasks.size(),
                         std::vector < State > (population.size())),
                pong_buf;
        size_t ping_buf_size = population.size(), pong_buf_size = 0;
        // [num_best_states × (cluster_size, float)]
        std::vector < StatesScorePair > scoreboard;
        // [cluster_size × *]
        std::vector < std::unordered_set < std::string > > scoreboard_set(_measured_states_set);
        std::vector < std::vector < float > >
                scores(cur_cluster->tasks.size(), std::vector < float > (population.size()));
        std::vector < float > scores_per_population(cur_cluster->tasks.size()),
                              acc_scores(ping_buf_size);
        std::vector < State > states_per_population(cur_cluster->tasks.size());
        std::vector < std::string > state_strs_per_population(cur_cluster->tasks.size());
        std::vector < double > prefix_sum_probs(ping_buf_size);
        float max_score = 0.f;

        // cross over parameters
        const int c_num_cross_overs
                = C_EVOLUTIONARY_SEARCH_CROSS_OVER_RATIO *
                  C_EVOLUTIONARY_SEARCH_POPULATION;
        int mutation_succ_cnt = 0, crossover_succ_cnt = 0,
            mutation_fail_cnt = 0, corssover_fail_cnt = 0;
        std::vector < int > crossover_fail_counters = {0, 0, 0, 0, 0};

        // initialize the ping buffer to the population transposed
        for (size_t i = 0; i < population.size(); ++i)
        {
                CHECK(population[i].size() ==
                      cur_cluster->tasks.size());
                for (size_t j = 0; j < cur_cluster->tasks.size(); ++j)
                {
                        ping_buf[j][i] = population[i][j];
                }
        }
        // initialize the task indices, an auxiliary data structure for traversal
        std::vector < size_t > task_indices(cur_cluster->tasks.size());
        for (size_t task_idx = 0;
             task_idx < cur_cluster->tasks.size(); ++task_idx)
        {
                task_indices[task_idx] = task_idx;
        }
        for (int evo_search_iter = 0; evo_search_iter <= C_EVOLUTIONARY_SEARCH_NUM_ITERS;
             ++evo_search_iter)
        {
                // 1. Predict the performance numbers for all the search tasks
                //    in the search cluster.
                for (size_t task_idx = 0; task_idx < cur_cluster->tasks.size();
                     ++task_idx)
                {
                        cur_cluster->tasks[task_idx]->compute_dag.InferBound(&ping_buf[task_idx]);
                        PruneInvalidState(cur_cluster->tasks[task_idx],
                                          &ping_buf[task_idx]);
                        _program_cost_model->Predict(cur_cluster->tasks[task_idx], 
                                                     ping_buf[task_idx], &scores[task_idx]);
                        CHECK(scores[task_idx].size() == ping_buf[task_idx].size());
                }
                for (size_t pop_idx = 0; pop_idx < ping_buf_size;
                     ++pop_idx)
                {
                        // 2. Transpose the scores into [cluster_size × *]. Then
                        //    ∀ population, accumulate its scores.
                        for (size_t task_idx = 0;
                             task_idx < cur_cluster->tasks.size(); ++task_idx)
                        {
                                scores_per_population[task_idx] = scores[task_idx][pop_idx];
                                states_per_population[task_idx]
                                        = ping_buf[task_idx][pop_idx];
                                state_strs_per_population[task_idx]
                                        = ping_buf[task_idx][pop_idx].ToStr();
                        }
                        acc_scores[pop_idx] = 
                                std::accumulate(scores_per_population.begin(),
                                                scores_per_population.end(), 0.f);
                        auto state_recorded_on_scoreboard = 
                                [&scoreboard_set,
                                 &state_strs_per_population](const size_t task_idx)
                                -> bool
                                {
                                        return scoreboard_set[task_idx].count(state_strs_per_population[task_idx]) == 0;
                                };

                        if (std::any_of(task_indices.begin(), task_indices.end(),
                                        state_recorded_on_scoreboard))
                        {
                                // make sure that all search tasks are consistent
                                CHECK(std::all_of(task_indices.begin(), task_indices.end(),
                                                  state_recorded_on_scoreboard));
                                if (scoreboard.size() < static_cast < size_t > (num_best_states))
                                {
                                        scoreboard.emplace_back(states_per_population, acc_scores[pop_idx]);
                                        std::push_heap(scoreboard.begin(), scoreboard.end(),
                                                       std::greater < StatesScorePair > ());
                                        for (size_t task_idx = 0; task_idx < cur_cluster->tasks.size();
                                             ++task_idx)
                                        {
                                                scoreboard_set[task_idx].insert(state_strs_per_population[task_idx]);
                                        }
                                }
                                // Otherwise, push the states onto the scoreboard if
                                // the accumulated score is larger than the smallest
                                // score on board.
                                else if (acc_scores[pop_idx] > scoreboard.front().second)
                                {
                                        for (size_t task_idx = 0; task_idx < cur_cluster->tasks.size(); 
                                             ++task_idx)
                                        {
                                                scoreboard_set[task_idx].erase(
                                                        scoreboard.front().first[task_idx].ToStr());
                                                scoreboard_set[task_idx].insert(ping_buf[task_idx][pop_idx].ToStr());
                                        }
                                        // maintain the scoreboard
                                        std::pop_heap (scoreboard.begin(), scoreboard.end(),
                                                       std::greater < StatesScorePair > ());
                                        scoreboard.back() = StatesScorePair(states_per_population, acc_scores[pop_idx]);
                                        std::push_heap(scoreboard.begin(), scoreboard.end(),
                                                       std::greater < StatesScorePair > ());
                                }
                                if (acc_scores[pop_idx] > max_score)
                                {
                                        max_score = acc_scores[pop_idx];
                                }
                        }  // if (std::any_of(task_indices.begin(), task_indices.end(),
                           //                 state_recorded_on_scoreboard))
                }  // for (pop_idx ∈ [0, ping_buf_size))
                if (evo_search_iter == C_EVOLUTIONARY_SEARCH_NUM_ITERS)
                {
                        break;
                }
                // =============================================================
                // Crossover
                // =============================================================
                double sum = 0.;
                prefix_sum_probs.resize(acc_scores.size());
                for (size_t i = 0; i < acc_scores.size(); ++i)
                {
                        sum += std::max(acc_scores[i], 0.f);
                        prefix_sum_probs[i] = sum;
                }
                for (size_t i = 0; i < acc_scores.size(); ++i)
                {
                        prefix_sum_probs[i] = prefix_sum_probs[i] / sum;
                }
                for (size_t co = 0;
                     _cross_over_enabled &&
                     static_cast < int > (pong_buf_size) < c_num_cross_overs &&
                     co < c_num_cross_overs; ++co)
                {
                        int pop_idx1 = RandomChoose(prefix_sum_probs, &_rng),
                            pop_idx2 = RandomChoose(prefix_sum_probs, &_rng);
                        if (pop_idx1 == pop_idx2)
                        {
                                
                        }
                }
        }  // for (i ∈ range[0, C_EVOLUTIONARY_SEARCH_NUM_ITERS))
}


Array < State >
ClusterSearchPolicyNode::Search(
        SearchCluster cluster, ProgramMeasurer measurer,
        const int num_trials,
        const int early_stopping,  // early stopping is not used
        const int num_measures_per_iter,
        Array < SearchCallback > pre_search_callbacks)
        // pre-search callbacks are not used, for now
{
        // [* × cluster_size]
        std::vector < std::vector < State > > best_states, random_states;
        cur_cluster = cluster;
        _num_measures_per_iter = num_measures_per_iter;

        SplitFactorizationMemo split_memo;
        std::vector < std::vector < PrimExpr > > factor_schemes
                = split_memo.GetFactorizationSchemes(10, 4, 50);
        for (const auto & scheme : factor_schemes)
        {
                DEBUG_LOG_VEC(scheme);
        }
        Map < String, ObjectRef > params{
                {String("eps_greedy"), PrimExpr(0.05f)},
                {String("evolutionary_search_population"), PrimExpr(2048)},
                {String("evolutionary_search_num_iters"), PrimExpr(10)},
                {String("evolutionary_search_mutation_prob"), PrimExpr(0.85f)},
                {String("evolutionary_search_crossover_ratio"), PrimExpr(0.05f)},
                {String("evolutionary_search_use_measured_ratio"), PrimExpr(0.2f)},
                {String("cpu_multi_level_tiling_structure"), String("SSRSRS")},
                {String("gpu_multi_level_tiling_structure"), String("SSSRRSRS")},
                {String("disable_change_compute_location"), PrimExpr(0)}};
        SketchSearchPolicy sketch_search_policy(RandomModel(), params, 0);
        for (const SearchTask & task : cur_cluster->tasks)
        {
                sketch_search_policy->cur_task = task;
                std::vector < State > best_states_0, random_states_0;
                sketch_search_policy->SearchOneRound(&best_states_0, 0, &random_states_0);
                // std::string dummy_string;
                // std::cin >> dummy_string;
        }

        if (num_trials <= 1) 
        {
                LOG(INFO) << "Starting to search for one round";
                SearchOneRound(&best_states, &random_states, 0);
                return best_states[0];
        }
        else  // if (num_trials > 1)
        {
                // [cluster_size × *]
                std::vector < std::vector < MeasureInput > >  inputs;
                std::vector < std::vector < MeasureResult > > results;
                const int num_random_states = C_EPS_GREEDY * num_measures_per_iter;
                measurer->Reset();

                for (int num_trials_done = 0; num_trials_done < num_trials; num_trials_done += inputs.size())
                {
                        if (!inputs.empty())
                        {
                                for (size_t task_idx = 0;
                                     task_idx < cur_cluster->tasks.size(); ++task_idx)
                                {
                                        _program_cost_model->Update(inputs[task_idx], results[task_idx]);
                                }
                        }

                        SearchOneRound(&best_states, &random_states,
                                       num_random_states);
#define INFER_BOUND_FOREACH(states)                                             \
        for (std::vector < State > & states_per_cluster : states)               \
        {                                                                       \
                CHECK(states_per_cluster.size() == cur_cluster->tasks.size());  \
                for (size_t task_idx = 0;                                       \
                     task_idx < cur_cluster->tasks.size(); ++task_idx)          \
                {                                                               \
                        states_per_cluster[task_idx]                            \
                                = cur_cluster->tasks[task_idx]->compute_dag.InferBound(states_per_cluster[task_idx]);  \
                }                                                               \
        }
                        INFER_BOUND_FOREACH(best_states);
                        INFER_BOUND_FOREACH(random_states);
                        inputs .resize(cur_cluster->tasks.size());
                        results.resize(cur_cluster->tasks.size());
                        PickStatesWithEpsGreedy(&inputs, best_states, random_states,
                                                num_trials - num_trials_done);
                        if (inputs.empty())
                        {
                                LOG(INFO) << "All candidates in the search space "
                                             "have been measured";
                                break;
                        }
                        for (size_t task_idx = 0; task_idx < cur_cluster->tasks.size();
                             ++task_idx)
                        {
                                measurer->Measure(cur_cluster->tasks[task_idx], inputs[task_idx],
                                                  &results[task_idx]);
                        }

                        for (size_t task_idx = 0; task_idx < cur_cluster->tasks.size();
                             ++task_idx)
                        {
                                for (const MeasureResult & res : results[task_idx])
                                {
                                        _measured_states_thruput[task_idx]
                                                .push_back(1.0f / FloatArrayMean(res->costs));
                                }
                        }
                }  // for (trail_idx ∈ [0, num_trials))
                Array < State > best_states_from_measurer;
                for (size_t task_idx = 0;
                     task_idx < cur_cluster->tasks.size(); ++task_idx)
                {
                        best_states_from_measurer.push_back(
                                measurer->best_state[cur_cluster->tasks[task_idx]->workload_key]);
                }
                return best_states_from_measurer;
        }  // if (num_trials > 1)
}


ClusterSearchPolicy::ClusterSearchPolicy(
        CostModel program_cost_model,
        const int seed)
{
        ObjectPtr < ClusterSearchPolicyNode > node
                = make_object < ClusterSearchPolicyNode > ();
        node->_program_cost_model = program_cost_model;
        node->_rng = std::mt19937(seed);
        data_ = node;
}


TVM_REGISTER_NODE_TYPE(ClusterSearchPolicyNode);

TVM_REGISTER_GLOBAL("ansor.ClusterSearchPolicy")
        .set_body_typed([](CostModel program_cost_model, int seed)
                        {
                                return ClusterSearchPolicy(program_cost_model, seed);
                        });


        }  // namespace ansor
}  // namespace tvm

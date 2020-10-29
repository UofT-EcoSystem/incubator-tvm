#include "selective_tuning.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "auto_schedule.h"
#include "utils.h"
#include "search_policy/utils.h"
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
                                  << node->tasks.size() << " search tasks";
                }
        );


constexpr double ClusterSearchPolicyNode::C_EPS_GREEDY;
constexpr int ClusterSearchPolicyNode::C_EVOLUTIONARY_SEARCH_POPULATION;
constexpr int ClusterSearchPolicyNode::C_EVOLUTIONARY_SEARCH_NUM_ITERS;
constexpr double ClusterSearchPolicyNode::C_EVOLUTIONARY_SEARCH_MUTATION_PROB;
constexpr double ClusterSearchPolicyNode::C_EVOLUTIONARY_SEARCH_CROSSOVER_RATIO;
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

        for (int f = 1; f < min_extent; ++f)
        {
                if (std::all_of(extents.begin(), extents.end(),
                                [f](const int extent) -> int
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
        std::vector < State > * const pstates)
{
        State & repr_state = (*pstates)[cur_cluster->repr_idx];
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
                                continue;
                        }
                        std::vector < int > extents;
                        const size_t num_lengths = repr_split_step->lengths.size();
                        for (const State & state : (*pstates))
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
                                        CHECK(cluster_factor.size() == pstates->size());
                                }
                        }  // for (candidate ∈ candidates)

                        size_t rand_candidate_idx = _rng() % candidates.size();
                        for (size_t task_idx = 0; task_idx < extents.size(); ++task_idx)
                        {
                                std::vector < PrimExpr > lengths;
                                for (size_t len_idx = 0; len_idx < num_lengths; ++len_idx)
                                {
                                        lengths.push_back(candidates[rand_candidate_idx][len_idx][task_idx]);
                                }
                                StateNode * pstate = (*pstates)[task_idx].CopyOnWrite();
                                const SplitStepNode * const split_step
                                        = (*pstates)[task_idx]->transform_steps[step_idx].as < SplitStepNode > ();
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
        std::vector < State > * const pstates)
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
                State & state = (*pstates)[task_idx];
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
                                        return -1;
                                }
                                state.bind_thread(stage_idx, vthread_iter, kVThread);

                                // 3. Fuse the third outermost space tile as threadIdx.
                                for (size_t i = 2; i < compute_op->axis.size() + 2; i++)
                                {
                                        const auto & iter = state->stages[stage_idx]->iters[i];
                                        if (!StrEndsWith(iter->name, ".2"))
                                        {
                                                break;
                                        }
                                        to_fuse.push_back(state->stages[stage_idx]->iters[i]);
                                }
                                const auto & threadidx_iter = state.fuse(stage_idx, to_fuse);
                                if (GetExtent(threadidx_iter) < 
                                    task->hardware_params->warp_size)
                                {
                                        return -1;
                                }
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
        }  // for (task_idx ∈ range(pstates->size()))
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
ClusterSearchPolicyNode::InitPopulationUnroll(std::vector < State > * const pstates)
{
        size_t rand_auto_unroll_config = C_GPU_AUTO_UNROLL_CONFIGS[_rng() % 5];
        for (size_t task_idx = 0; task_idx < cur_cluster->tasks.size(); ++task_idx)
        {
                const SearchTask & task = cur_cluster->tasks[task_idx];
                State & state = (*pstates)[task_idx];
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
        }  // for (task_idx ∈ range(pstates->size()))
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
                State tmp_repr_state
                        = cur_cluster->sketches[cur_cluster->repr_idx][rand_sketch_idx];
                std::vector < State > tmp_states(cur_cluster->tasks.size());
                for (const Array < State > & sketch : cur_cluster->sketches)
                {
                        tmp_states.push_back(sketch[rand_sketch_idx]);
                }
                InitPopulationFillTileSize(&tmp_states);
                if (InitPopulationThreadBind(&tmp_states))
                {
                        failed_attempts += 1;
                        continue;
                }
                InitPopulationUnroll(&tmp_states);
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

                for (size_t task_idx = 0; task_idx < cluster->tasks.size(); ++task_idx)
                {
                        best_state_per_task[task_idx] = best_states[task_idx][0];
                }
                return best_state_per_task;
        // }
        // else  // if (n_trails > 1)
        // {
        //         LOG(FATAL) << "NOT Implemented";
        // }
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


void
AutoScheduleSearchCluster(SearchCluster cluster,
                          ClusterSearchPolicy cluster_search_policy,
                          TuneOption tune_option)
{
        LOG(INFO) << "Auto-scheduling " << cluster;
        // search for the best schedule
        ProgramMeasurer measurer =
                ProgramMeasurer(tune_option->builder,
                                tune_option->runner,
                                tune_option->measure_callbacks,
                                tune_option->verbose);
        Array < State > states = cluster_search_policy->Search(
                cluster, measurer,
                tune_option->n_trials,
                tune_option->early_stopping,
                tune_option->num_measure_per_iter,
                tune_option->pre_search_callbacks);
}


TVM_REGISTER_GLOBAL("ansor.AutoScheduleBySearchCluster")
        .set_body_typed([](SearchCluster cluster,
                           ClusterSearchPolicy cluster_search_policy,
                           TuneOption tune_option)
                        {
                                AutoScheduleSearchCluster(cluster, cluster_search_policy, tune_option);
                        });


        }  // namespace ansor
}  // namespace tvm

#include "cluster_search_policy.h"
#include "search_cluster.h"
#include "../auto_schedule.h"


namespace tvm {
        namespace ansor {


Array < Array < ObjectRef > >
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
        std::vector < Array < ObjectRef > > auto_sched_results(2);
        for (size_t task_idx = 0; task_idx < cluster->tasks.size();
             ++task_idx)
        {
                te::Schedule sched; Array < te::Tensor > tensors;
                std::tie(sched, tensors)
                        = cluster->tasks[task_idx]
                                 ->compute_dag.ApplySteps(states[task_idx]->transform_steps);
                auto_sched_results[0].push_back(sched);
                auto_sched_results[1].push_back(tensors);
        }
        return auto_sched_results;
}


TVM_REGISTER_GLOBAL("ansor.AutoScheduleBySearchCluster")
        .set_body_typed([](SearchCluster cluster,
                           ClusterSearchPolicy cluster_search_policy,
                           TuneOption tune_option)
                        -> Array < Array < ObjectRef > >
                        {
                                return AutoScheduleSearchCluster(cluster, cluster_search_policy, tune_option);
                        });


        }  // namespace ansor
}  // namespace tvm

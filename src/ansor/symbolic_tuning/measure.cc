#include "../measure.h"
#include "cluster_search_policy.h"
#include "search_cluster.h"


namespace tvm {
        namespace ansor {


void ProgramMeasurerNode::Measure(
        const SearchCluster & cluster,
        const ClusterSearchPolicy & policy,
        const std::vector < std::vector < MeasureInput > > & inputs,
        std::vector < std::vector < MeasureResult > > * const results)
{
        results->clear();
        results->resize(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i)
        {
                CHECK(inputs[i].size() == cluster->tasks.size());
                results[i].resize(cluster->tasks.size());
        }
        size_t batch_size = builder->n_parallel * 2;
        for (size_t i = 0; i < inputs.size() * cluster->tasks.size(); i += batch_size)
        {
                std::vector < MeasureInput >  ibatch;
                std::vector < MeasureResult > rbatch; 
                for (size_t j = 0;
                     j < batch_size && (i + j) < inputs.size() * cluster->tasks.size(); ++j)
                {
                        ibatch.push_back(inputs[(i + j) / (cluster->tasks.size())]
                                               [(i + j) % (cluster->tasks.size())]);
                }
                SilentMeasure(ibatch, &rbatch);
                for (size_t j = 0; j < ibatch.size(); ++j)
                {
                        size_t task_idx = (i + j) % (cluster->tasks.size());
                        double flops;
                        if (rbatch[j]->error_no == 0)
                        {
                                flops = cluster->tasks[task_idx]->compute_dag
                                               ->flop_ct / FloatArrayMean(rbatch[j]->costs);
                                error_ct = 0;
                        }
                        else
                        {
                                flops = 0.0;
                                error_ct++;
                        }
                        const std::string & workload_key = ibatch[j]->task->workload_key;
                        if (flops > best_flops[workload_key])
                        {
                                best_flops[workload_key] = flops;
                                best_state[workload_key] = ibatch[j]->state;
                                best_ct[workload_key] = ct;
                        }
                        ct++;
                }  // for (j ∈ range(ibatch.size()))

                // callback
                for (const auto & callback : callbacks)
                {
                        callback->callback(ibatch, rbatch);
                }

                // rbatch
                for (size_t j = 0; j < rbatch.size(); ++j)
                {
                        (*results)[(i + j) / (cluster->tasks.size())]
                                  [(i + j) % (cluster->tasks.size())] = rbatch[j];
                }

                if (error_ct > max_continous_error)
                {
                        LOG(FATAL) << "Too many errors happened during tuning";
                }
        }  // for (i ∈ [0, inputs.size() * cluster->tasks.size()))
}


        }  // namespace ansor
}  // namespace tvm

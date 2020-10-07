#include "selective_tuning.h"

#include <vector>


namespace tvm {
        namespace ansor {


SearchCluster::SearchCluster(Array < SearchTask > tasks,
                             SearchTask representative)
{
        SearchClusterNode node = make_object < SearchClusterNode > ();
        
}


Array < State >
ClusterSearchPolicyNode::Search(
        SearchCluster search_cluster,
        ProgramMeasurer measurer,
        const int n_trials,
        const int early_stopping,
        const int n_measures_per_iter,
        Array < SearchCallback > pre_search_callbacks)
{
        std::vector < State > best_states, random_states;
        this->cluster = cluster;
        this->_n_measures_per_iter = n_measures_per_iter;

        if (n_trails <= 1) 
        {
                SearchOneRound(&best_states, 0, &random_states);
        }
        else  // if (n_trails > 1)
        {
                LOG(FATAL) << "NOT Implemented";
        }
}

        }  // namespace ansor
}  // namespace tvm

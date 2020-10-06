#include "cluster_search_policy.h"

#include <vector>


namespace tvm {
        namespace ansor {

Array < State >
ClusterSearchPolicyNode::Search(
        SearchCluster search_cluster,
        ProgramMeasurer measurer,
        const int n_trials,
        const int early_stopping,
        const int n_msrs_per_iter,
        Array < SearchCallback > pre_search_callbacks)
{
        std::vector < State > best_states, random_states;
        this->cluster = cluster;
        this->_n_msrs_per_iter = n_msrs_per_iter;

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

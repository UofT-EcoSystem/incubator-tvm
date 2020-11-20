#include "../feature.h"
#include "search_cluster.h"



namespace tvm {
        namespace ansor {


extern TVMByteArray
SerializeFeatures(std::vector < std::vector < float > > && features,
                  std::vector < float > && normalized_throughputs,
                  std::vector < int > && task_ids,
                  std::vector < char > * out_data);


TVM_REGISTER_GLOBAL("ansor.GetPerStmtFeaturesFromStates")
        .set_body([](TVMArgs args, TVMRetValue *ret)
        {
                SearchCluster cluster;
                Array < Array < State > > states = args[1];
                const int max_n_bufs = args[2];

                std::vector < std::vector < float > > features;
                std::vector < float > normalized_throughputs;
                std::vector < int > task_ids;

                GetPerStmtFeaturesFromClusterStates(cluster, states, 0, max_n_bufs, &features);

                std::vector < char > byte_array;
                *ret = SerializeFeatures(std::move(features),
                                         std::vector < float > (),
                                         std::vector < int > (), 
                                         &byte_array);
});


        }  // namespace ansor
}  // namespace tvm

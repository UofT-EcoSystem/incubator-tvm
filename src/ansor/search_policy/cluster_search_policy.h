#pragma once


#include "./search_policy.h"


namespace tvm {
        namespace ansor {


using SearchCluster = Array < SearchTask >;

class ClusterSearchPolicyNode : public Object
{
private:
        int _n_msrs_per_iter;  ///< number of measurements per iteration
public:
        SearchCluster search_cluster;

        void VisitAttrs(AttrVisitor * v)
        {
                v->Visit("cluster", &cluster);
        }
        Array < State >
        Search(SearchCluster search_cluster,
               ProgramMeasurer measurer,
               const int n_trials,
               const int early_stopping,
               const int n_msrs_per_iter,
               Array < SearchCallback > pre_search_callbacks);
        static constexpr const char * const _type_key = "ansor.ClusterSearchPolicy";
        TVM_DECLARE_BASE_OBJECT_INFO(ClusterSearchPolicyNode, Object);
};  // class ClusterSearchPolicyNode

class ClusterSearchPolicy : public ObjectRef
{
public:
    TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ClusterSearchPolicy, ObjectRef,
                                          ClusterSearchPolicyNode);
};

        }  // namespace ansor
}  // namespace tvm

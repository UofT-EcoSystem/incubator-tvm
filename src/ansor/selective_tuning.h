#pragma once


#include "search_policy/search_policy.h"


namespace tvm {
        namespace ansor {

class SearchClusterNode : public Object
{
public:
        Array < SearchTask > tasks;
        SearchTask representative;

        void VisitAttrs(AttrVisitor * v)
        {
                v->Visit("tasks", &tasks);
                v->Visit("representative", &representative);
        }

        static constexpr const char* _type_key = "ansor.SearchTask";
        TVM_DECLARE_FINAL_OBJECT_INFO(SearchTaskNode, Object);
};  // class SearchClusterNode


class SearchCluster : public ObjectRef
{
public:
        SearchCluster(Array < SearchTask > tasks,
                      SearchTask representative);
        TVM_DEFINE_OBJECT_REF_METHODS(SearchCluster, ObjectRef,  
                                      SearchClusterNode);
};  // class SearchCluster

class ClusterSearchPolicyNode : public Object
{
private:
        int _n_measures_per_iter;
public:
        SearchCluster cur_cluster;

        void VisitAttrs(AttrVisitor * v)
        {
                v->Visit("cur_cluster", &cur_cluster);
        }
        Array < State >
        Search(SearchCluster cluster, ProgramMeasurer measurer,
               const int n_trials,
               const int early_stopping,
               const int n_measures_per_iter,
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

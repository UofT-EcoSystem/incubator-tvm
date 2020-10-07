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

        static constexpr const char * _type_key = "ansor.SearchTask";
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
        static constexpr double C_eps_greedy = 0.05;
        static constexpr size_t C_evolutionary_search_population = 2048;
        static constexpr size_t C_evolutionary_search_num_iters = 10;
        static constexpr double C_evolutionary_search_mutation_prob = 0.85;
        static constexpr double C_evolutionary_search_crossover_ratio = 0.05;
        static constexpr double C_evolutionary_search_use_measured_ratio = 0.2;
        static constexpr const char * C_gpu_multi_level_tiling_structure = "SSSRRSRS";
        static constexpr bool C_disable_change_compute_location = false;

        void SearchOneRound(
                std::vector < std::vector < State > > * const best_states,
                const int num_random_states,
                std::vector < std::vector < State > > * const random_states);
public:
        SearchCluster cur_cluster;

        void VisitAttrs(AttrVisitor * v)
        {
                v->Visit("cur_cluster", &cur_cluster);
        }
        Array < State >
        Search(SearchCluster cluster, ProgramMeasurer measurer,
               const int num_trials,
               const int early_stopping,
               const int num_measures_per_iter,
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

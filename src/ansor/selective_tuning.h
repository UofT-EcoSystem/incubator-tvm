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
        // Tuning Parameters (Constants)
        static constexpr double C_EPS_GREEDY = 0.05;
        static constexpr size_t C_EVOLUTIONARY_SEARCH_POPULATION = 2048;
        static constexpr size_t C_EVOLUTIONARY_SEARCH_NUM_ITERS = 10;
        static constexpr double C_EVOLUTIONARY_SEARCH_MUTATION_PROB = 0.85;
        static constexpr double C_EVOLUTIONARY_SEARCH_CROSSOVER_RATIO = 0.05;
        static constexpr double C_EVOLUTIONARY_SEARCH_USE_MEASURED_RATIO = 0.2;
        static constexpr const char * C_GPU_MULTI_LEVEL_TILING_STRUCTURE = "SSSRRSRS";
        static constexpr bool C_DISABLE_CHANGE_COMPUTE_LOCATION = false;

        /**
         * @brief Search for one round.
         */
        void SearchOneRound(
                std::vector < std::vector < State > > * const best_states,
                const int num_random_states,
                std::vector < std::vector < State > > * const random_states);
        std::vector < std::vector < State > > _sketch_caches;
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

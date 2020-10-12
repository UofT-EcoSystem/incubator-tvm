#pragma once


#include <random>
#include <vector>

#include "search_policy/search_policy.h"


namespace tvm {
        namespace ansor {

class SearchClusterNode : public Object
{
public:
        Array < SearchTask > tasks;
        Array < Array < State > > sketches;
        int representative_idx;

        void VisitAttrs(AttrVisitor * const v)
        {
                v->Visit("tasks", &tasks);
                v->Visit("sketches", &sketches);
                v->Visit("representative_idx", &representative_idx);
        }

        static constexpr const char * _type_key = "ansor.SearchCluster";
        TVM_DECLARE_FINAL_OBJECT_INFO(SearchTaskNode, Object);
};  // class SearchClusterNode


class SearchCluster : public ObjectRef
{
public:
        SearchCluster(Array < SearchTask > tasks,
                      SearchTask representative,
                      Array < State > shared_sketch);
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
        static constexpr const int C_GPU_AUTO_UNROLL_CONFIGS[] = {0, 16, 64, 512, 1024};
        static constexpr const char * C_GPU_MULTI_LEVEL_TILING_STRUCTURE = "SSSRRSRS";
        static constexpr bool C_DISABLE_CHANGE_COMPUTE_LOCATION = false;

        std::mt19937 _rng;
        std::vector < State > _measured_states_vec;
        /**
         * @brief Samples the initial population.
         */
        void SampleInitPopulation(
                const size_t out_size,
                std::vector < State > * const out_states);
        /**
         * @brief Search for one round.
         */
        void SearchOneRound(
                std::vector < std::vector < State > > * const best_states,
                const int num_random_states,
                std::vector < std::vector < State > > * const random_states);
        /**
         * @brief  Initialize the population's (tile sizes/thread binding/unroll).
         * @return 0 if the initialization is successful, nonzero otherwise
         */
        int InitPopulationFillTileSize(State * const state);
        int InitPopulationThreadBind  (State * const state);
        int InitPopulationUnroll      (State * const state);

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

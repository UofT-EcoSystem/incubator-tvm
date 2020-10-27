#pragma once


#include <random>
#include <stack>
#include <tuple>
#include <vector>
#include <unordered_map>

#include "cost_model/cost_model.h"
#include "search_policy/search_policy.h"


namespace tvm {
        namespace ansor {

class SearchClusterNode : public Object
{
public:
        Array < SearchTask > tasks;
        Array < Array < State > > sketches;
        int repr_idx;

        void VisitAttrs(AttrVisitor * const v)
        {
                v->Visit("tasks", &tasks);
                v->Visit("sketches", &sketches);
                v->Visit("repr_idx", &repr_idx);
        }

        static constexpr const char * _type_key = "ansor.SearchCluster";
        TVM_DECLARE_FINAL_OBJECT_INFO(SearchClusterNode, Object);
};  // class SearchClusterNode


class SearchCluster : public ObjectRef
{
public:
        SearchCluster(Array < SearchTask > tasks,
                      Array < Array < State > > sketches,
                      const int repr_idx);
        TVM_DEFINE_OBJECT_REF_METHODS(SearchCluster, ObjectRef,  
                                      SearchClusterNode);
};  // class SearchCluster


// Vectorized Extents and Factors
// [cluster_size]
using ClusterExtentsT = std::vector < int >;
// [sizeof(factors)]
using ClusterFactorsT = std::vector < int >;

class ClusterSplitFactorCache
{
public:
        using KT = std::tuple  < ClusterExtentsT, int, int >;
        using StackT = std::vector < std::vector < PrimExpr > >;
        using VT = std::vector < StackT >;
private:
        std::unordered_map < KT, VT > _cache;
        /**
         * @brief 
         */
        void DFSEnumerate(const ClusterExtentsT & extents,
                          const int depth = 0);
        /**
         * @brief Get the factors based on the extents.
         * @todo  Because currently the returned factors are of size
         *        [sizeof(factors) × cluster size], it is possible that a valid
         *        factor might not be found (imagine the case in which there are
         *        3 search tasks whose extents are given by [10, 17, 20]). This
         *        case is currently NOT handled.
         */
        const ClusterFactorsT &
        GetFactors(const ClusterExtentsT & extents);
        // internal variables to avoid passing parameters around different methods
        VT * _ret;
        StackT _working_stack;  // [num_lengths × cluster_size]
        int _num_lengths;
        int _max_innermost_factor;
        std::unordered_map < ClusterExtentsT, ClusterFactorsT > _factor_cache;
public:
        /**
         * @brief  Get the factorization schemes based on the (extents,
         *         num_lengths, max_innermost_factor) tuple.
         * @return all possible factorization schemes
         */
        const VT & GetFactorizationSchemes(
                const ClusterExtentsT & extents,
                const int num_lengths,
                const int max_innermost_factor);
};


class ClusterSearchPolicy;


class ClusterSearchPolicyNode : public Object
{
private:
        // Tuning Parameters (Constants)
        static constexpr double C_EPS_GREEDY = 0.05;
        static constexpr int C_EVOLUTIONARY_SEARCH_POPULATION = 2048;
        static constexpr int C_EVOLUTIONARY_SEARCH_NUM_ITERS = 10;
        static constexpr double C_EVOLUTIONARY_SEARCH_MUTATION_PROB = 0.85;
        static constexpr double C_EVOLUTIONARY_SEARCH_CROSSOVER_RATIO = 0.05;
        static constexpr double C_EVOLUTIONARY_SEARCH_USE_MEASURED_RATIO = 0.2;
        static constexpr int C_GPU_AUTO_UNROLL_CONFIGS[] = {0, 16, 64, 512, 1024};
        static constexpr const char * C_GPU_MULTI_LEVEL_TILING_STRUCTURE = "SSSRRSRS";
        static constexpr bool C_DISABLE_CHANGE_COMPUTE_LOCATION = false;

        CostModel _program_cost_model;
        std::mt19937 _rng;
        std::vector < State > _measured_states_vec;

        /**
         * @brief Samples the initial population.
         */
        void SampleInitPopulation(const size_t out_size,
                                  std::vector < std::vector < State > > * const out_states);
        /**
         * @brief Search for one round.
         */
        void SearchOneRound(
                std::vector < std::vector < State > > * const best_states,
                const int num_random_states,
                std::vector < std::vector < State > > * const random_states);
        /**
         * @brief  Initialize the population's (tile sizes/thread bindings/
         *         unrolling factors).
         * @return 0 if the initialization is successful, nonzero otherwise
         */
        ClusterSplitFactorCache _split_factor_cache;
        int InitPopulationFillTileSize(
                std::vector < State > * const pstates);
        int InitPopulationThreadBind(
                std::vector < State > * const pstates);
        int InitPopulationUnroll(std::vector < State > * const pstates);
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

        friend class ClusterSearchPolicy;
};  // class ClusterSearchPolicyNode

class ClusterSearchPolicy : public ObjectRef
{
public:
        ClusterSearchPolicy(CostModel program_cost_model, int seed);
        TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ClusterSearchPolicy, ObjectRef,
                                              ClusterSearchPolicyNode);
};

        }  // namespace ansor
}  // namespace tvm

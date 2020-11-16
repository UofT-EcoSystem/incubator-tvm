#pragma once

#include "search_cluster.h"

#include "../measure.h"


namespace tvm {
        namespace ansor {


class ParamCostModelNode : public Object
{
private:
        PackedFunc _update_func, _predict_func, _predict_stages_func;
public:
        void Update (const Array < Array < MeasureInput > > & inputs,
                     const Array < Array < MeasureResult > > & results);
        void Predict(const SearchCluster & cluster,
                     const std::vector < std::vector < State > > & states,
                     std::vector < std::vector < float > > * const scores);
        void PredictStages(const SearchCluster & cluster,
                           const std::vector < std::vector < State > > & states,
                           std::vector < std::vector < float > > * const scores,
                           std::vector < std::vector < std::vector < float > > > * const stage_scores);
        static constexpr const char * _type_key = "ansor.ParamCostModel";
        TVM_DECLARE_FINAL_OBJECT_INFO(ParamCostModelNode, Object);
};


class ParamCostModel : public ObjectRef
{
public:
        ParamCostModel(PackedFunc update_func, PackedFunc predict_func,
                       PackedFunc predict_stages_func);
        TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ParamCostModel, ObjectRef,
                                              ParamCostModelNode);
};  // class ParamCostModel


        }  // namespace ansor
}  // namespace tvm

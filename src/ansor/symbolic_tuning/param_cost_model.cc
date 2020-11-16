#include "param_cost_model.h"


namespace tvm {
        namespace ansor {




TVM_REGISTER_OBJECT_TYPE(ParamPythonBasedCostModelNode);

TVM_REGISTER_GLOBAL("ansor.ParamPythonBasedCostModel")
        .set_body_typed([](PackedFunc update_func, PackedFunc predict_func,
                           PackedFunc predict_stages_func)
        {
                
        })


        }  // namespace ansor
}  // namespace tvm

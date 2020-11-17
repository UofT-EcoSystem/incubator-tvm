import ctypes
import numpy as np

from tvm.runtime import Object
from .. import _ffi_api


@tvm._ffi.register_object("ansor.VectorizedPythonBasedModel")
class VectorizedPythonBasedModel(Object):
    def __init__(self):
        def update_func(inputs, results):
            self.update(inputs, results)

        def predict_func(task, states, return_ptr):
            return_ptr = ctypes.cast(return_ptr, ctypes.POINTER(ctypes.c_float))
            array_wrapper = np.ctypeslib.as_array(return_ptr, shape=(len(states),))
            array_wrapper[:] = self.predict(task, states)

        def predict_stages_func(task, states, return_ptr):
            ret = self.predict_stages(task, states)
            return_ptr = ctypes.cast(return_ptr, ctypes.POINTER(ctypes.c_float))
            array_wrapper = np.ctypeslib.as_array(return_ptr, shape=ret.shape)
            array_wrapper[:] = ret

        self.__init_handle_by_constructor__(_ffi_api.PythonBasedModel, update_func,
                                            predict_func, predict_stages_func)

    def update(self, inputs, results):
        raise NotImplementedError

    def predict(self, cluster, states):
        raise NotImplementedError

    def predict_stages(self, cluster, states):
        raise NotImplementedError

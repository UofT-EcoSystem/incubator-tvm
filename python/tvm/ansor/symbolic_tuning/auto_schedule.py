"""
<bojian/TVM-SymbolicTuning> Cluster search tasks for selective tuning (Ansor implementation).
"""
import tvm._ffi

from ...runtime import Object
from .. import _ffi_api
from ..auto_schedule import SketchSearchPolicy


import logging
logger = logging.getLogger(__name__)


@tvm._ffi.register_object("ansor.SearchCluster")
class SearchCluster(Object):
    def __init__(self, tasks, sketches, repr_idx):
        self.__init_handle_by_constructor__(
                _ffi_api.SearchCluster, 
                tasks, sketches, repr_idx)


@tvm._ffi.register_object("ansor.ClusterSearchPolicy")
class ClusterSearchPolicy(Object):
    def __init__(self, program_cost_model, seed):
        self.__init_handle_by_constructor__(
                _ffi_api.ClusterSearchPolicy,
                program_cost_model, seed)


def auto_schedule_cluster(search_cluster, cluster_search_policy,
                          tune_option):
    scheds, tensors = _ffi_api.AutoScheduleBySearchCluster(
            search_cluster, cluster_search_policy,
            tune_option)
    return scheds, tensors

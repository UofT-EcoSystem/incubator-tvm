"""
<bojian/TVM-SymbolicTuning> Cluster search tasks for selective tuning (Ansor implementation).
"""
import tvm._ffi

from ..autotvm import SelectiveTuningABC
from ..runtime import Object
from . import _ffi_api

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


class SelectiveTuning(SelectiveTuningABC):
    @classmethod
    def ComputePairwiseSimilarity(cls, taskA, taskB):
        """
        Compute the similarity between two tasks.

        In the original implementation, we are allowed to compute the similarity
        between two tasks if they share the same schedule space map. However,
        since the notion of schedule space map no longer exists in Ansor, we
        have to turn to sketches (?) (i.e., two search tasks are allowed to be
        compared for similarity if they share the same initial sketch). This
        heuristic is subject to change in hte future.
        """
        if _ffi_api.StateCmp(taskA.initial_sketch_state,
                             taskB.initial_sketch_state):
            return 1.
        else:
            return 0.
    
    @classmethod
    def MakeSearchCluster


def auto_schedule_cluster(search_cluster, cluster_search_policy,
                          tune_option):
    sched, tensors = _ffi_api.AutoScheduleBySearchCluster(
            search_cluster, cluster_search_policy,
            tune_option)
    return sched, tensors

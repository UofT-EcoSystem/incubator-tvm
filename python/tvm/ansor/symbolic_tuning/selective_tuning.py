"""
<bojian/TVM-SymbolicTuning> Cluster search tasks for selective tuning (Ansor implementation).
"""
from ...autotvm import SelectiveTuningABC
from .. import _ffi_api
from ..auto_schedule import SketchSearchPolicy
from ..cost_model.xgb_model import XGBModel
from .auto_schedule import SearchCluster

import logging
logger = logging.getLogger(__name__)


class SelectiveTuning(SelectiveTuningABC):
    __slots__ = 'search_policy'

    def __init__(self):
        self.search_policy = SketchSearchPolicy(
                program_cost_model=XGBModel(seed=0),
                seed=0)

    def ComputePairwiseSimilarity(self, taskA, taskB):
        """
        Compute the similarity between two tasks.

        In the original implementation, we are allowed to compute the similarity
        between two tasks if they share the same schedule space map. However,
        since the notion of schedule space map no longer exists in Ansor, we
        have to turn to sketches (?) (i.e., two search tasks are allowed to be
        compared for similarity if they share the same initial sketch). This
        heuristic is subject to change in hte future.
        """
        if _ffi_api.StateCmp(taskA.sketch_states[0],
                             taskB.sketch_states[0]):
            return 1.
        else:
            return 0.

    def Annotate(self, search_task):
        """
        Annotate a search task with its own sketch states.
        """
        sketch_states = self.search_policy.generate_sketches(task=search_task)
        logger.info("Search Task={}, Initial Sketch State={}"
                    .format(search_task, sketch_states[0]))
        search_task.sketch_states = sketch_states

    def MakeSearchClusters(self, search_tasks, clusters, centroids):
        search_clusters = []
        for cidx, cluster in enumerate(clusters):
            if cluster:
                tasks, sketches = [], []
                for tidx in cluster:
                    tasks.append(search_tasks[tidx])
                    sketches.append(
                            search_tasks[tidx].sketch_states)
                search_clusters.append(SearchCluster(tasks, sketches, centroids[cidx][0]))
        return search_clusters

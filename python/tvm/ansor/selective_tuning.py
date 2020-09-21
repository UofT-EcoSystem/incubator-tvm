"""
<bojian/TVM-SymbolicTuning> Cluster search tasks for selective tuning (Ansor implementation).
"""
from ..autotvm import SelectiveTuningABC

from .cost_model    import XGBModel
from .auto_schedule import SketchSearchPolicy

import logging
logger = logging.getLogger(__name__)


search_policy = SketchSearchPolicy(program_cost_model=XGBModel(seed=0),
                                   seed=0)


class SelectiveTuning(SelectiveTuningABC):
    def ComputePairwiseSimilarity(self, sketch_stateA, sketch_stateB):
        """
        Compute the similarity between two tasks.

        In the original implementation, we are allowed to compute the similarity
        between two tasks if they share the same schedule space map. However,
        since the notion of schedule space map no longer exists in Ansor, we
        have to turn to sketches (?) (i.e., two search tasks are allowed to be
        compared for similarity if they share the same sketch).
        """
        pass

    def ComputePSM(self, sketch_states):
        pass

    def ClusterPSM(self, sketch_states):
        pass

    def MarkDepend(self, search_tasks):
        logger.info("Marking dependent tasks")
        def _SearchTask2SketchState(search_task):
            """
            Transform a search task to sketch.
            """
            sketch = search_policy.generate_sketches(task=search_task)
            assert len(list(sketch)) == 1, \
                   "Not implemented for cases where there are more than 1 state"
            state = sketch[0]
            logger.info("Search Task={}, State={}".format(search_task, state))
            return state
        sketch_states = [_SearchTask2SketchState(task) for task in search_tasks]
        centroids, labels = self.ClusterPSM(sketch_states)

"""
<bojian/TVM-SymbolicTuning> Cluster search tasks for selective tuning (Ansor implementation).
"""
from ..autotvm import SelectiveTuningABC

import logging
logger = logging.getLogger(__name__)


class SelectiveTuning(SelectiveTuningABC):
    @classmethod
    def ComputePairwiseSimilarity(cls, sketch_stateA, sketch_stateB):
        """
        Compute the similarity between two tasks.

        In the original implementation, we are allowed to compute the similarity
        between two tasks if they share the same schedule space map. However,
        since the notion of schedule space map no longer exists in Ansor, we
        have to turn to sketches (?) (i.e., two search tasks are allowed to be
        compared for similarity if they share the same sketch).
        """
        return 0.

"""
<bojian/TVM-SymbolicTuning> Cluster search tasks for selective tuning (Ansor implementation).
"""
from ..autotvm import SelectiveTuningABC
from . import _ffi_api

import logging
logger = logging.getLogger(__name__)


class SelectiveTuning(SelectiveTuningABC):
    @classmethod
    def ComputePairwiseSimilarity(cls, taskA, taskB):
        """
        Compute the similarity between two tasks.

        In the original implementation, we are allowed to compute the similarity
        between two tasks if they share the same schedule space map. However,
        since the notion of schedule space map no longer exists in Ansor, we
        have to turn to sketches (?) (i.e., two search tasks are allowed to be
        compared for similarity if they share the same sketch).
        """
        stages_cacheA, transform_stepsA = \
                _ffi_api.StateGetStages(taskA), \
                _ffi_api.StateGetTransformSteps(taskA)
        logger.info("StagesCacheA={}, TransformStepsA={}"
                    .format(stages_cacheA, transform_stepsA))
        return 0.

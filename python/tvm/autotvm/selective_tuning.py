"""
<bojian/TVM-SymbolicTuning> Cluster search tasks for selective tuning (AutoTVM implementation).
"""
import logging
import numpy as np
logger = logging.getLogger(__name__)


def ComputePairwiseSimilarity(taskA, taskB):
    """
    Compare the pairwise similarity metric between two tasks.
    """
    if taskA.name != taskB.name:
        return 0.
    return 1.
    

def ComputePSM(search_tasks):
    psm = np.zeros(shape=(len(search_tasks), len(search_tasks)),
                   dtype=np.float32)
    for i, _ in enumerate(search_tasks):
        for j in range(i + 1, len(search_tasks)):
            psm[i, j] = ComputePairwiseSimilarity(search_tasks[i], search_tasks[j])
    logger.info("psm={}".format(psm))


def ClusterPSM(search_tasks):
    ComputePSM(search_tasks)
    return None, None


def MarkDepend(search_tasks):
    logger.info("Marking dependent tasks")
    centroids, labels = ClusterPSM(search_tasks)

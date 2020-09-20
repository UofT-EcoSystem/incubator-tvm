"""
<bojian/TVM-SymbolicTuning> Cluster search tasks for selective tuning (AutoTVM implementation).
"""
import logging
logger = logging.getLogger(__name__)


def ComputeSimilarity(taskA, taskB):
    pass


def PSMClustering(search_tasks):
    return None, None


def MarkDepend(search_tasks):
    logger.info("Marking dependent tasks")
    centroids, labels = PSMClustering(search_tasks)

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
    for task in search_tasks:
        logger.info("Config Space={}".format(task.config_space))
        logger.info("Config Space.Space Map={}".format(task.config_space.space_map))
    centroids, labels = PSMClustering(search_tasks)

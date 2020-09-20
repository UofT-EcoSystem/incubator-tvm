"""
<bojian/TVM-SymbolicTuning> Cluster search tasks for selective tuning (Ansor implementation).
"""
from .cost_model    import XGBModel
from .auto_schedule import SketchSearchPolicy

import logging
logger = logging.getLogger(__name__)


search_policy = SketchSearchPolicy(program_cost_model=XGBModel(seed=0),
                                   seed=0)


def ComputeSimilarity(taskA, taskB):
    """
    Compute the similarity between two tasks.

    In the original implementation, we are allowed to compute the similarity
    between two tasks if they share the same schedule space map. However, since
    the notion of schedule space map no longer exists in Ansor, we have to turn
    to sketches (?) (i.e., two search tasks are allowed to be compared for
    similarity if they share the same sketch).
    """
    pass


def PSMClustering(sketches):
    return None, None


def SearchTask2Sketch(search_task):
    """
    Transform a search task to sketch.
    """
    sketch = search_policy.generate_sketches(task=search_task)
    assert len(list(sketch)) == 1, \
           "Not implemented for cases where there are more than 1 state"
    state = sketch[0]
    logger.info("Search Task={}, State={}".format(search_task, state))
    return state


def MarkDepend(search_tasks):
    logger.info("Marking dependent tasks")
    sketches = [SearchTask2Sketch(task) for task in search_tasks]
    centroids, labels = PSMClustering(sketches)

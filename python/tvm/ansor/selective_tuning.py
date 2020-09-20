import logging

from .cost_model    import XGBModel
from .auto_schedule import SketchSearchPolicy

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
    logging.info("Search Task={}, Sketch={}"
                 .format(search_task, sketch))
    return sketch


def MarkDepend(search_tasks):
    logging.info("Marking dependent tasks")
    sketches = [SearchTask2Sketch(task) for task in search_tasks]
    centroids, labels = PSMClustering(sketches)

"""
<bojian/TVM-SymbolicTuning> Cluster search tasks for selective tuning (AutoTVM implementation).
"""
import logging
import numpy as np
logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod


class SelectiveTuningABC(ABC):
    @classmethod
    @abstractmethod
    def ComputePairwiseSimilarity(cls, taskA, taskB):
        pass

    @classmethod
    def ComputePSM(cls, search_tasks):
        cls.psm = np.zeros(shape=(len(search_tasks), len(search_tasks)),
                           dtype=np.float32)
        for i, _ in enumerate(search_tasks):
            cls.psm[i, i] = 1.
            for j in range(i + 1, len(search_tasks)):
                cls.psm[i, j] = cls.psm[j, i] = \
                        cls.ComputePairwiseSimilarity(search_tasks[i], search_tasks[j])
        logger.info("psm=\n{}".format(cls.psm))

    @classmethod
    def ClusterPSM(cls, search_tasks):
        cls.ComputePSM(search_tasks)
        import networkx as nx
        # create a graph with task index as nodes and PSM as edge weights
        graph = nx.Graph()
        graph.add_nodes_from(range(len(search_tasks)))
        graph.add_edges_from([(i, j) for i in range(len(search_tasks))
                                     for j in range(i + 1, len(search_tasks))
                                     if cls.psm[i, j] > 0.])
        # cluster assignment for each task
        assigned_cluster = [([], None) for _ in range(len(search_tasks))]
        # find cliques and initailize clusters
        cliques = list(nx.find_cliques(graph))
        logger.info("cliques={}".format(cliques))
        clusters = []
        for cidx, clique in enumerate(cliques):
            clusters.append(set())
            for tidx in clique:
                assigned_cluster[tidx][0].append(cidx)
        # assign the tasks that only belong to one clique to the cluster
        for tidx in range(len(search_tasks)):
            if len(assigned_cluster[tidx]) == 1:
                cidx = assigned_cluster[tidx][0][0]
                clusters[cidx].add(tidx)
                assgined_cluster[tidx][1] = cidx

        def _weight_sum(primary_tidx, target_tidxs):
            return sum([cls.psm[primary_tidx][target_tidx] for target_tidx in target_tidxs])

        changed = True
        while changed:
            changed = False
            for tidx in range(len(search_tasks)):
                if len(assigned_cluster[tidx]) == 1:
                    continue
                assigned_cidx = max(assigned_cluster[tidx][0],
                                    key=lambda cidx: _weight_sum(tidx, clusters[cidx]))
                if assigned_cidx != assigned_cluster[tidx][1]:
                    changed = True
                    clusters[assigned_cidx].add(tidx)
                    if assigned_cluster[tidx][1] is not None:
                        clusters[assigned_cluster[tidx][1]].remove(tidx)
                    assigned_cluster[tidx] = (assigned_cluster[tidx][0], assigned_cidx)
        labels = [label for _, label in assigned_cluster]

        # ∀cluster, select the task that has the maximum weight sum to other
        # tasks in cluster
        centroids = []
        for cluster in clusters:
            if cluster:
                centroids.append(max(cluster, key=lambda tidx: _weight_sum(tidx, cluster)))
            else:  # empty cluster
                centroids.append(-1)
        logger.info("centroids={}, labels={}".format(centroids, labels))
        return centroids, labels

    @classmethod
    def MarkDepend(cls, search_tasks):
        logger.info("Marking dependent tasks")
        centroids, labels = cls.ClusterPSM(search_tasks)
        search_task_dependent_map = {}
        for tidx, task in enumerate(search_tasks):
            if labels[tidx] != -1:
                repr_idx = centroids[labels[tidx]]
                representative = search_tasks[repr_idx]
                logger.info("centroid={} (ReprTaskIdx={}) <= Task={} (TaskIdx={})"
                            .format(representative, repr_idx, task, tidx))
                search_task_dependent_map[task] = representative
            else:
                logger.warning("Task={} does not have dependent".format(task))
        logger.info("Select {} tasks over {} tasks"
                    .format(sum([1 if search_task_dependent_map[task] == task else 0
                                 for task in search_tasks]),
                            len(search_tasks)))
        return search_task_dependent_map


class SelectiveTuning(SelectiveTuningABC):
    @classmethod
    def ComputePairwiseSimilarity(cls, taskA, taskB):
        """
        Compare the pairwise similarity metric between two tasks.
        """
        if taskA.name != taskB.name:
            return 0.
        config_space_mapA = taskA.config_space.space_map
        config_space_mapB = taskB.config_space.space_map

        logger.info("Merging {} with {}"
                    .format(config_space_mapA, config_space_mapB))

        config_space_mapU = set(config_space_mapA)
        config_space_mapU.update(config_space_mapB)
        if len(config_space_mapA) != len(config_space_mapB) or \
           len(config_space_mapA) != len(config_space_mapU):
            logger.info("len(ConfigSpaceMapA)={config_space_mapA_len} != len(ConfigSpaceMapB)={} or "
                        "len(ConfigSpaceMapA)={config_space_mapA_len} != len(ConfigSpaceMapU)={}"
                        .format(len(config_space_mapB), len(config_space_mapU),
                                config_space_mapA_len=len(config_space_mapA)))
            return 0.
        similarity_vec = [config_space_mapA[name].similar(config_space_mapB[name])
                          for name in config_space_mapU]
        logger.info("similarity vec={}".format(similarity_vec))
        return np.prod(similarity_vec)

"""Use a measurement simulator to debug and test task scheduler"""

from collections import namedtuple, defaultdict

import numpy as np
import random

import tvm
from tvm import ansor
from common import get_workload_keys, get_workload_weights

FakeMeasureResult = namedtuple("FakeMeasureResult", ['costs'])
FakeFloatImm = namedtuple("FakeFloatImm", ['value'])

class MeasurementSimulator:
    """
    A measurement simulator for debugging
    This class will generate artificial tuning curve for search tasks
    """
    def __init__(self, tasks, max_n_trials=20000):
        self.tasks = tasks
        self.task_id_dict = {self.tasks[i].workload_key : i for i in range(len(self.tasks))}
        self.max_n_trials = max_n_trials
        self.data = self.ct = None

    def init_data(self, case_no):
        self.data = []

        def power_curve(rate, scale, vibration=20.0, stop_improve=1200):
            costs = []
            for t in range(stop_improve):
                costs.append(1 / (np.power(t+1, rate)))
            costs = np.array(costs)
            costs = np.concatenate((costs, costs[-1] * np.ones(self.max_n_trials - len(costs))))
            costs = costs.flatten()
            costs *= np.random.uniform(1.0, vibration, len(costs))
            costs = costs / costs.min() * scale
            return costs

        for i in range(len(tasks)):
            if case_no in [0, 1]:     # all workloads are the same
                costs = power_curve(0.2, 3e-4)
            elif case_no in [2, 5]:   # some workloads runs much faster
                scale = [0.000744, 0.000438, 0.000067, 0.000117, 0.000238, 0.000368, 0.000058, 0.000503, 0.000172, 0.000199, 0.000273, 0.000449, 0.000654, 0.000146, 0.000273, 0.000294, 0.000395, 0.000104, 0.000219, 0.000479]
                assert len(scale) == len(self.tasks)
                costs = power_curve(0.2, scale[i])
            elif case_no == 3:   # random workload
                costs = power_curve(0.2, random.uniform(1e-4, 3e-4))
            elif case_no == 4:   # two workloads are the best
                if i in [2, 3]:
                    costs = power_curve(0.2, 1e-2, stop_improve=2000)
                else:
                    costs = power_curve(0.2, random.uniform(1e-4, 3e-4))
            elif case_no == 6:
                costs = power_curve(0.2, tasks[i].compute_dag.flop_ct / 1000e9)
            else:
                assert False
            self.data.append(costs)

    def reset_counter(self):
        self.ct = [0 for _ in range(len(tasks))]

    def get_next_batch(self, task, num_measure):
        task_idx = self.task_id_dict[task.workload_key]
        costs = self.data[task_idx][self.ct[task_idx]:self.ct[task_idx] + num_measure]
        self.ct[task_idx] += num_measure

        inputs = []
        results = []
        for c in costs:
            inputs.append(None)
            results.append(FakeMeasureResult(costs=[FakeFloatImm(c)]))

        return inputs, results

    def plot_costs(self, costs):
        import matplotlib.pyplot as plt
        plt.plot(1 / costs)
        plt.show()
        exit()


if __name__ == "__main__":
    wkl = 'resnet-50.C+'
    n_trails = 15000
    num_measure_per_iter = 48
    case_no = 0
    num_repeat = 20
    seed = 2

    np.random.seed(seed)
    random.seed(seed)

    # build search tasks
    target = tvm.target.create('llvm')
    wkl_keys = get_workload_keys(wkl)
    task_weights = get_workload_weights(wkl)
    tasks = []
    for wkl_key in wkl_keys:
        dag = ansor.workload_key_to_dag(wkl_key)
        tasks.append(ansor.SearchTask(dag, wkl_key, target))
    tune_option = ansor.TuneOption(n_trials=n_trails, num_measure_per_iter=num_measure_per_iter)
    measure_simulator = MeasurementSimulator(tasks)

    strategies = ['gradient', 'round-robin']

    for case_no in [0, 1, 2, 3, 4, 5, 6]:
        # set objective
        if case_no == 0:
            objective_func = None
        elif case_no == 5:
            def objective_func(costs):
                return sum(max(costs[i], 0.0004) * task_weights[i] for i in range(len(costs)))
        else:
            def objective_func(costs):
                return sum(costs[i] * task_weights[i] for i in range(len(costs)))

        # set the parameter to control the weight of similarity term
        if case_no == 6:
            beta = 2
        else:
            beta = 1e30

        score_dict = defaultdict(list)
        action_dict = defaultdict(list)

        for _ in range(num_repeat):
            measure_simulator.init_data(case_no)
            for task_scheduler in strategies:
                measure_simulator.reset_counter()
                if task_scheduler != 'rl':
                    tuner = ansor.SimpleTaskScheduler(tasks, objective_func,
                                                      strategy=task_scheduler,
                                                      verbose=0,
                                                      beta=beta,
                                                      use_debug_measurement_simulator=measure_simulator)
                else:
                    raise ValueError("Invalid task scheduler: " + task_scheduler)

                tuner.tune(tune_option)

                score_dict[task_scheduler].append(1 / tuner.cur_score)
                action_dict[task_scheduler].append(tuner.task_cts)

        scores = [-np.mean(score_dict[key]) for key in strategies]
        indices = np.argsort(scores)

        print("================ Case %d ================ " % case_no)
        for idx in indices:
            key = strategies[idx]
            print("%-20s%.2f (%.2f) \t%s" %
                  (key,
                   np.mean(score_dict[key]), np.std(score_dict[key]),
                   ", ".join(["%4.0f" % x for x in np.mean(action_dict[key], axis=0)])))


# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=unused-argument, no-self-use, invalid-name
"""Base class of tuner"""
import logging

import numpy as np

from ..measure import MeasureInput, create_measure_batch
from ..util import format_si_prefix

from ..env import GLOBAL_SCOPE

logger = logging.getLogger('autotvm')

class Tuner(object):
    """Base class for tuners

    Parameters
    ----------
    task: autotvm.task.Task
        Tuning Task
    """

    def __init__(self, task, **kwargs):
        self.param = kwargs
        self.recorder = None

        self.task = task

        # <bojian/TVM-SymbolicTuning>
        # self.best_config = None
        self.record_cache = []  # best_config -> record_cache

        self.best_flops = 0
        self.best_measure_pair = None
        self.best_iter = 0

        # time to leave
        self.ttl = None
        self.n_trial = None
        self.early_stopping = None

    def has_next(self):
        """Whether has next untried config in the space

        Returns
        -------
        has_next: bool
        """
        raise NotImplementedError()

    def next_batch(self, batch_size):
        """get the next batch of configs to be measure on real hardware

        Parameters
        ----------
        batch_size: int
            The size of the batch

        Returns
        -------
        a batch of configs
        """
        raise NotImplementedError()

    def update(self, inputs, results):
        """Update parameters of the tuner according to measurement results

        Parameters
        ----------
        inputs: Array of autotvm.measure.MeasureInput
            The input for measurement
        results: Array of autotvm.measure.MeasureResult
            result for measurement
        """


    def tune(self, n_trial, measure_option, early_stopping=None, callbacks=(), si_prefix='G'
             # <bojian/TVM-SymbolicTuning>
           , depend_mode='top1'
             ):
        """Begin tuning

        Parameters
        ----------
        n_trial: int
            Maximum number of configs to try (measure on real hardware)
        measure_option: dict
            The options for how to measure generated code.
            You should use the return value ot autotvm.measure_option for this argument.
        early_stopping: int, optional
            Early stop the tuning when not finding better configs in this number of trials
        callbacks: List of callable
            A list of callback functions. The signature of callback function is
            (Tuner, List of MeasureInput, List of MeasureResult)
            with no return value. These callback functions will be called on
            every measurement pair. See autotvm/tuner/callback.py for some examples.
        si_prefix: str
            One of tvm.autotvm.util.SI_PREFIXES. The SI prefix to use when reporting FLOPS.
        """
        measure_batch = create_measure_batch(self.task, measure_option)
        n_parallel = getattr(measure_batch, 'n_parallel', 1)
        early_stopping = early_stopping or 1e9
        self.n_trial = n_trial
        self.early_stopping = early_stopping

        # Validate si_prefix arg
        format_si_prefix(0, si_prefix)

        old_level = logger.level

        GLOBAL_SCOPE.in_tuning = True

        def _parse_depend_mode(depend_mode, total_num_tuned_configs):
            """
            Parse the depend mode to return the number of the best dependent configs.
            """
            try:
                if depend_mode.startswith('top'):
                    return int(depend_mode[3:])
                elif depend_mode[-1] == '%':
                    return int(total_num_tuned_configs *
                               float(depend_mode[:-1]) / 100))
                else:
                    raise ValueError
            except ValueError:
                logger.warning('Unknown mode of using the dependent best configs: %s', mode)
            return 1
        n_best_dependent_configs = _parse_depend_mode(
                depend_mode, len(self.task.dependent.tuned_configs))
        dependent_configs = iter(self.task.dependent.tuned_configs)


        i = error_ct = 0
        while i < n_trial:
            if not self.has_next():
                break

            # <bojian/TVM-SymbolicTuning>
            # Comment: `next_batch` gets the next batch of configs to be measured on real hardware
            # configs = self.next_batch(min(n_parallel, n_trial - i))
            configs = []
            tune_dependent_configs_only = (self.task.dependent != self.task) and \
                                          (self.task.dependent.tuned_configs)
                                          
            if tune_dependent_configs_only:
                for _ in range(min(n_parallel, n_trail - i,
                                   n_best_dependent_configs - i)):
                    try:
                        configs.append(next(dependent_configs))
                    except StopIteration:
                        break
            if not configs:
                if tune_dependent_configs_only:
                    logger.warning('Fallback because all dependent configs are not working')
                    return
                configs = self.next_batch(min(n_parallel, n_trial - i))

            inputs = [MeasureInput(self.task.target, self.task, config) for config in configs]
            results = measure_batch(inputs)

            # keep best config
            for k, (inp, res) in enumerate(zip(inputs, results)):
                config = inp.config
                if res.error_no == 0:
                    flops = inp.task.flop / np.mean(res.costs)
                    error_ct = 0
                else:
                    flops = 0
                    error_ct += 1

                # <bojian/TVM-SymbolicTuning>
                if res.error_no == 0:
                    self.record_cache.append((inp, res))

                if flops > self.best_flops:
                    self.best_flops = flops

                    # <bojian/TVM-SymbolicTuning>
                    # self.best_config = config

                    self.best_measure_pair = (inp, res)
                    self.best_iter = i + k

                logger.debug("No: %d\t%sFLOPS: %.2f/%.2f\tresult: %s\t%s",
                             i + k + 1, si_prefix, format_si_prefix(flops, si_prefix),
                             format_si_prefix(self.best_flops, si_prefix), res, config)

            i += len(results)
            self.ttl = min(early_stopping + self.best_iter, n_trial) - i

            self.update(inputs, results)
            for callback in callbacks:
                callback(self, inputs, results)

            if i >= self.best_iter + early_stopping:
                logger.debug("Early stopped. Best iter: %d.", self.best_iter)
                break

            if error_ct > 150:
                logging.basicConfig()
                logger.warning("Too many errors happen in the tuning. Now is in debug mode")
                logger.setLevel(logging.DEBUG)
            else:
                logger.setLevel(old_level)

            # <bojian/TVM-SymbolicTuning>
            if tune_dependent_configs_only and self.best_flops > 0 and \
               i >= n_best_dependent_configs:
                break
        # <bojian/TVM-SymbolicTunning>
        for record in sorted(self.record_cache, 
                             key=lambda r: np.mean(r[1].costs)):
            self.task.tuned_configs.append(record[0].config)

        GLOBAL_SCOPE.in_tuning = False
        del measure_batch

    def reset(self):
        """reset the status of tuner"""

        # <bojian/TVM-SymbolicTuning>
        # self.best_config = None
        self.record_cache = []  # best_config -> record_cache

        self.best_flops = 0
        self.best_measure_pair = None

    def load_history(self, data_set):
        """load history data for transfer learning

        Parameters
        ----------
        data_set: Array of (MeasureInput, MeasureResult) pair
            Previous tuning records
        """
        raise NotImplementedError()

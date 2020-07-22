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

"""Serialization and other I/O support for tuning logs (measurement records)"""

import numpy as np

import tvm._ffi
from tvm.runtime import Object
from .measure import MeasureCallback, MeasureErrorNo
from .loop_state import State
from . import _ffi_api


@tvm._ffi.register_object("ansor.LogToFile")
class LogToFile(MeasureCallback):
    """
    A measurement callback that writes measurement records into a file

    Parameters
    ----------
    filename : Str
    """

    def __init__(self, filename="ansor_tuning.json"):
        self.__init_handle_by_constructor__(_ffi_api.LogToFile, filename)


@tvm._ffi.register_object("ansor.LogReader")
class LogReader(Object):
    """
    Reader of the json log file

    Parameters
    ----------
    filename : Str
    """
    def __init__(self, filename):
        self.__init_handle_by_constructor__(_ffi_api.LogReader, filename)

    def read_lines(self, max_size=-1, skip_size=0):
        inputs, results = _ffi_api.LogReaderReadLines(
            self, max_size, skip_size)
        return inputs, results

    def __iter__(self):
        while True:
            ret = _ffi_api.LogReaderReadNext(self)
            if ret is None or not len(ret):
                break
            yield ret[0], ret[1]  # (input, result)


def load_from_file(filename: str):
    """Load measurement records from a file"""
    return zip(*LogReader(filename).read_lines())


def write_measure_records_to_file(filename, inputs, results):
    """Write(append) measure records to file"""
    _ffi_api.WriteMeasureRecordsToFile(filename, inputs, results)


def get_states_from_measure_inputs(inputs, task):
    """Get states from measure inputs"""
    state_objects = _ffi_api.GetStatesFromMeasureInputs(inputs, task)
    return [State(s, task.compute_dag) for s in state_objects]



def ckpt_measure_pair_in_file(
        log_file,
        ckpt_file_prefix,
        ckpt_period=10,
        target=tvm.target.cuda()):
    log_reader = LogReader(log_file)
    best_cost, best_input, best_result = 1e30, None, None

    ckpt_costs = []

    for i, (input, result) in enumerate(log_reader):
        if ((i + 1) % ckpt_period) == 0:
            from .workload_registry import workload_key_to_dag

            ckpt_costs.append(best_cost)
            dag = workload_key_to_dag(best_input.task.workload_key)
            sched, in_args = dag.apply_steps_from_state(best_input.state)
            cuda_kernel = tvm.build(sched, in_args, target=target)
            with open(ckpt_file_prefix + ('%d_sched.log' % (i + 1)), 'w') as fout:
                fout.write('{}'.format(
                        tvm.lower(sched, in_args, simple_mode=True)))
            with open(ckpt_file_prefix + ('%d_cuda_kernel.log' % (i + 1)), 'w') as fout:
                fout.write('{}'.format(cuda_kernel.get_source()))

        if result.error_no != MeasureErrorNo.NO_ERROR:
            continue
        costs = []
        for value in result.costs:
            costs.append(value.value)
        cost = np.mean(costs)

        if cost < best_cost:
            best_cost, best_input, best_result = cost, input, result
    with open(ckpt_file_prefix + 'costs.log', 'w') as fout:
        fout.write('{}'.format(ckpt_costs))


def best_measure_pair_in_file(filename, workload_key=None, target=None):
    """ Return the best measurement pair form a log file

    Parameters
    ----------
    filename : Str
    workload_key : Str
    target : Str

    Returns
    -------
    inp : MeasureInput
    res : MeasureResult
    """
    log_reader = LogReader(filename)
    best_cost = 1e30
    best_inp = None
    best_res = None

    for inp, res in log_reader:
        if res.error_no != MeasureErrorNo.NO_ERROR:
            continue
        if workload_key and inp.task.workload_key != workload_key:
            continue
        if target and inp.task.target.target_name != target.target_name:
            continue

        costs = []
        for value in res.costs:
            costs.append(value.value)
        cost = np.mean(costs)
        if cost < best_cost:
            best_cost = cost
            best_inp = inp
            best_res = res

    return best_inp, best_res

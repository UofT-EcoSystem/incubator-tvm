"""Distill a log file: Save the best logs from the original file into a new file
Usage:
python3 distill_log.py log.json
"""
import argparse
import os

import numpy as np

from tvm import ansor, relay

from common import LogFileDatabase, measure_schedule
import tune_op_subgraph
from tune_network import get_network

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=str)
    parser.add_argument("--out-file", type=str, default=None)
    parser.add_argument("--n-lines", type=int, default=-1, help='Only load first n lines')
    parser.add_argument("--remeasure", action='store_true', help='Replay the log to get more accurate measurement')
    parser.add_argument("--load-network", type=str)
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    if args.out_file is None:
        out_file = args.log_file
        out_file = out_file.replace(".json", ".best.json")
    else:
        out_file = args.out_file

    if args.load_network:
        os.environ['TVM_AUTO_CACHE_FLUSH'] = "1"
        os.environ['TVM_BIND_MASTER_CORE_0'] = "1"

        print("Load tasks from %s" % args.load_network)
        mod, params, input_name, data_shape, out_shape = get_network(args.load_network, args.model_path, args.batch_size, "NHWC")
        workloads, wkl_weights = auto_scheduler.extract_from_program(mod, target='llvm',
                params=params, ops=(relay.op.nn.dense, relay.op.nn.softmax,
                                    relay.op.nn.conv2d, relay.op.nn.conv2d_transpose,
                                    relay.op.nn.max_pool2d, relay.op.nn.avg_pool2d,
                                    relay.op.nn.global_max_pool2d, relay.op.nn.global_avg_pool2d,
                                    relay.op.nn.conv3d, relay.op.nn.adaptive_avg_pool3d,
                                    relay.op.nn.batch_matmul, relay.op.mean))

    print("Loading the log file...")
    database = LogFileDatabase(args.log_file, args.n_lines)

    if args.remeasure:
        print("Remeasure the log...")
        best_records = list(database.best_by_targetkey.values())
        new_inputs = []
        new_results = []
        for i, (inp, res) in enumerate(best_records):
            dag = auto_scheduler.workload_key_to_dag(inp.task.workload_key)
            s, bufs = dag.apply_steps_from_state(inp.state, auto_scheduler.LayoutRewriteLevel.BOTH_REWRITE)
            costs = measure_schedule(s, bufs, inp.task.target, repeat=3)
            new_res = auto_scheduler.MeasureResult(costs, res.error_no, res.error_msg, res.all_cost, res.timestamp)
            new_inputs.append(inp)
            new_results.append(new_res)
            print("Line: %d / %d\tRemeasured: %.4fms\tRecorded: %.4f ms" %
                    (i + 1, len(best_records), 1e3 * np.mean(costs), 1e3 * np.mean([x.value for x in res.costs])))
        auto_scheduler.write_measure_records(out_file, new_inputs, new_results)
    else:
        database.write_best(out_file)

    print("The best records are written to output file %s" % out_file)


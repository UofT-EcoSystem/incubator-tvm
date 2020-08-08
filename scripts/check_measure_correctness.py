"""Replay the log and check the accuracy of measurement for a log file

Usage:
python3 check_measure_correctness.py log.json
"""

import argparse
import os

import numpy as np

import tvm
from tvm import relay, ansor
from tvm.ansor import workload_key_to_dag, LogReader, LayoutRewriteLevel

from common import measure_schedule, load_network, verify_gpu_code
import tune_op_subgraph
from tune_network import get_network

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=str)
    parser.add_argument("--n-lines", type=int, default=5)
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--print-schedule", action='store_true')
    parser.add_argument("--print-state", action='store_true')

    parser.add_argument("--load-network", type=str)
    parser.add_argument("--network-path", type=str, default=None, help="The path of tflite model")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--layout", type=str, default='NHWC')
    args = parser.parse_args()

    if args.load_network:
        load_network(args.load_network, args.network_path, args.batch_size, args.layout)

    os.environ['TVM_AUTO_CACHE_FLUSH'] = "1"
    os.environ['TVM_BIND_MASTER_CORE_0'] = "1"

    ct = 0
    for i, (inp, res) in enumerate(LogReader(args.log_file)):
        print(60 * "-")
        print("Line %d\twkl_key: %s\tcost: %.2f ms" % (
            i, inp.task.workload_key, 1e3 * ansor.utils.array_mean(res.costs)))

        try:
            dag = workload_key_to_dag(inp.task.workload_key)
        except Exception:
            print("Failed to load key: %s" % inp.task.workload_key)
            continue
        flops1 = dag.flop_ct / ansor.utils.array_mean(res.costs) / 10 ** 9

        if True or flops1 > args.threshold:
            s, bufs = dag.apply_steps_from_state(inp.state, LayoutRewriteLevel.BOTH_REWRITE)
            stmt = tvm.lower(s, bufs, simple_mode=True)
            #verify_gpu_code(stmt)
            if args.print_schedule:
                print(tvm.lower(s, bufs, simple_mode=True))
            if args.print_state:
                print(dag.infer_bound_from_state(inp.state))
                continue
            costs = measure_schedule(s, bufs, inp.task.target, repeat=10)
            flops2 = dag.flop_ct / np.mean(costs) / 10 ** 9

            print("Recorded GFLOPS: %.2f\tRemeasured GFLOPS: %.2f" % (flops1, flops2))

            ct += 1
            if ct >= args.n_lines:
                break


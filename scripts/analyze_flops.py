"""Analyze the GFLOPS of similar tasks. This is used to debug task scheduler"""
import argparse
import logging
import time
import random
import os
import numpy as np

import tvm
from tvm import ansor, relay
from tvm.ansor.task_scheduler import derive_similarity_tag

from common import load_network, to_str_round

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=str)
    parser.add_argument("--network", type=str, required=True)
    parser.add_argument("--network-path", type=str, default=None, help="The path of tflite model")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--layout", type=str, default='NHWC')
    args = parser.parse_args()

    load_network(args.network, args.network_path, args.batch_size, args.layout)

    dag_list = []
    dag_tag_list = []
    gflops_list = []

    dag_group = []
    tag_to_group = {}

    i = 0
    visited = set()
    for inp, res in ansor.LogReader(args.log_file):
        print(20 * "=" + " Line %d " % i + 20 * "=")
        print("Workload key: %s " % inp.task.workload_key)
        if inp.task.workload_key in visited:
            continue
        visited.add(inp.task.workload_key)
        cost = ansor.utils.array_mean(res.costs)
        dag = ansor.workload_key_to_dag(inp.task.workload_key)
        gflops = dag.flop_ct / cost / 1e9
        print("GFLOPS: %.2f\tTime: %.2f ms\tFLOP: %0.f" % 
                (gflops, cost * 1000, dag.flop_ct))
        tag = derive_similarity_tag(dag)
        dag_list.append(dag)
        gflops_list.append(gflops)
        dag_tag_list.append(tag)

        if tag != "":
            if tag not in tag_to_group:
                tag_to_group[tag] = len(tag_to_group)
                dag_group.append([])
            idx = tag_to_group[tag] 
            dag_group[idx].append(i)
        i += 1

    print("\n\n\n")

    #order = np.argsort(-np.array(gflops_list))
    order = list(range(len(gflops_list)))
    for i, index in enumerate(order):
        print(20 * "=" + " Graph %d : %7.2f GFLOPS " % (i, gflops_list[index]) + 20 * "=")
        print(dag_list[index])

    for group_id in range(len(dag_group)):
        print(20 * "=" + " Group %s : %s " % (group_id+1, dag_tag_list[dag_group[group_id][0]]) + 20 * "=")
        print(dag_group[group_id])
        group_gflops = []
        for dag_id in dag_group[group_id]:
            group_gflops.append(gflops_list[dag_id])
        print(to_str_round(group_gflops, 2))


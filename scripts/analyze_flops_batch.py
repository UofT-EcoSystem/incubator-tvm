"""
Analyze the GFLOPS of similar tasks. This is used to debug task scheduler
This script does analysis for a batch of jobs by calling analyze_flops.py
"""

import argparse
import logging
import time
import random
import os
import numpy as np

from common import run_cmd

def file_name_to_network_name(filename):
    items = filename.split('-')
    network_name = "".join(items[:-2])
    return network_name


if __name__ == "__main__":
    for target_name in ['llvm']:
        for network in ['bert']:
            for batch_size in [1]:
                log_file = "%s-B%d-%s.json" % (network, batch_size, target_name)
                best_log_file = log_file.replace(".json", ".best.json")
                output_file = "%s-B%d-%s.flops.ana" % (network, batch_size, target_name)

                if not os.path.exists(best_log_file):
                    run_cmd('python3 distill_log.py %s' % log_file)

                run_cmd('python3 analyze_flops.py %s --network %s --batch-size %d  > %s' % (best_log_file, network, batch_size, output_file))



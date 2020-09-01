"""Test the robustness of crossover with several workloads"""
from common import run_cmd

if __name__ == "__main__":
    for target in ['llvm', 'cuda']:
        run_cmd("rm -rf *.json")

        run_cmd('python3 tune_test.py --wkl matmul-512                     --n-trials 5 --target %s' % (target))
        run_cmd('python3 tune_test.py --wkl double-matmul                  --n-trials 5 --target %s' % (target))
        run_cmd('python3 tune_test.py --wkl nhwc-resnet-50.C0              --n-trials 5 --target %s' % (target))
        run_cmd('python3 tune_test.py --wkl bert-softmax                   --n-trials 5 --target %s' % (target))
        run_cmd('python3 tune_op_subgraph.py --wkl transpose_batch_matmul_softmax --fast-check --n-trials 5 --target %s' % (target))
        run_cmd('python3 tune_op_subgraph.py --wkl C2DWG_NHWC                     --fast-check --n-trials 5 --target %s' % (target))

        run_cmd('python3 tune_test.py --wkl matmul-512                     --tune false --target %s' % (target))
        run_cmd('python3 tune_test.py --wkl double-matmul                  --tune false --target %s' % (target))
        run_cmd('python3 tune_test.py --wkl nhwc-resnet-50.C0              --tune false --target %s' % (target))
        run_cmd('python3 tune_test.py --wkl bert-softmax                   --tune false --target %s' % (target))
        run_cmd('python3 tune_op_subgraph.py --wkl transpose_batch_matmul_softmax --fast-check --tune false --target %s' % (target))
        run_cmd('python3 tune_op_subgraph.py --wkl C2DWG_NHWC                     --fast-check --tune false --target %s' % (target))


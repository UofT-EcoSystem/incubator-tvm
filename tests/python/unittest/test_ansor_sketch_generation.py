import tvm
from tvm import te, ansor

from test_ansor_common import (matmul_ansor_test, conv2d_nchw_bn_relu_ansor_test,
                               max_pool2d_ansor_test, softmax_mn_ansor_test)

def print_sketches(sketches):
    for i, s in enumerate(sketches):
        print("=" * 20 + " %d " % i + "=" * 20)
        print(s)

def generate_sketches(workload_func, args, target):
    workload_key = ansor.make_workload_key_func(workload_func, args)
    dag = ansor.workload_key_to_dag(workload_key)
    task = ansor.SearchTask(dag, workload_key, tvm.target.create(target))
    policy = ansor.SketchSearchPolicy(ansor.RandomModel())
    return policy.generate_sketches(task)

def test_cpu_matmul_sketch():
    sketches = generate_sketches(matmul_ansor_test, (512, 512, 512), 'llvm')
    assert len(sketches) == 4  # 4 multi-level tiling sketches

    sketches = generate_sketches(matmul_ansor_test, (8, 8, 512), 'llvm')
    assert len(sketches) == 6  # 4 multi-level tiling sketches + 2 rfactor sketches

def test_cpu_conv2d_bn_relu_sketch():
    sketches = generate_sketches(conv2d_nchw_bn_relu_ansor_test,
                                 (1, 56, 56, 512, 512, 3, 1, 1), 'llvm')
    assert len(sketches) == 4

def test_cpu_max_pool2d_sketch():
    sketches = generate_sketches(max_pool2d_ansor_test, (1, 56, 56, 512, 1), 'llvm')
    assert len(sketches) == 1

def test_cpu_softmax_sketch():
    sketches = generate_sketches(softmax_mn_ansor_test, (1, 1024), 'llvm')
    assert len(sketches) == 9

def test_gpu_matmul_sketch():
    if not tvm.context("cuda", 0).exist:
        return

    sketches = generate_sketches(matmul_ansor_test, (512, 512, 512), 'cuda')
    assert len(sketches) == 2  # 2 multi-level tiling sketches

def test_gpu_conv2d_bn_relu_sketch():
    if not tvm.context("cuda", 0).exist:
        return

    sketches = generate_sketches(conv2d_nchw_bn_relu_ansor_test,
                                 (1, 56, 56, 512, 512, 3, 1, 1), 'cuda')
    assert len(sketches) == 2  # 2 multi-level tiling sketches

def test_gpu_max_pool2d_sketch():
    if not tvm.context("cuda", 0).exist:
        return

    sketches = generate_sketches(max_pool2d_ansor_test, (1, 56, 56, 512, 0), 'cuda')
    assert len(sketches) == 1

def test_gpu_softmax_sketch():
    if not tvm.context("cuda", 0).exist:
        return

    # todo(lmzheng): support rfactor for cuda
    return
    sketches = generate_sketches(softmax_mn_ansor_test, (1, 1024), 'cuda')
    assert len(sketches) == 9

if __name__ == "__main__":
    test_cpu_matmul_sketch()
    test_cpu_conv2d_bn_relu_sketch()
    test_cpu_max_pool2d_sketch()
    test_cpu_softmax_sketch()
    test_gpu_matmul_sketch()
    test_gpu_conv2d_bn_relu_sketch()
    test_gpu_max_pool2d_sketch()
    test_gpu_softmax_sketch()


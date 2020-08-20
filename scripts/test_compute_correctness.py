"""Test the correctness of compute definition"""

import numpy as np
import torch

import topi
import tvm
from tvm import te, ansor
import topi.testing

from common import random_bsr_matrix, random_csr_matrix, sparse_dense_bsr_compute,\
    sparse_conv2d_csr_compute, conv2d_nchw_bn_relu, conv2d_nhwc_bn_relu

def test_conv2d():
    #args = (1, 224, 224, 3, 64, 7, 2, 3)
    args = (1, 28, 28, 56, 56, 3, 1, 1)

    target = target_host = 'llvm'

    dag_ref = ansor.ComputeDAG(conv2d_nchw_bn_relu(*args))
    s_ref, bufs_ref = dag_ref.apply_steps_from_state(dag_ref.get_init_state())
    func_ref = tvm.build(s_ref, bufs_ref, target='llvm')
    ctx_ref = tvm.cpu()
    np_args_ref = [np.random.uniform(0.1, 1.0, topi.get_const_tuple(x.shape)).astype(x.dtype) for x in bufs_ref]
    tvm_args_ref = [tvm.nd.array(x, ctx=ctx_ref) for x in np_args_ref]

    dag = ansor.ComputeDAG(conv2d_nhwc_bn_relu(*args))
    s, bufs = dag.apply_steps_from_state(dag.get_init_state())
    func = tvm.build(s, bufs, target=target, target_host=target_host)
    ctx = tvm.context(str(target), 0)
    np_args = [np.array(x) for x in np_args_ref]
    np_args[0] = np_args[0].transpose([0, 2, 3, 1])
    np_args[1] = np_args[1].transpose([2, 3, 1, 0])
    np_args[2] = np_args[2].reshape((args[4],))
    np_args[3] = np_args[3].reshape((args[4],))
    np_args[4] = np_args[4].reshape((args[4],))
    np_args[5] = np_args[5].transpose([0, 2, 3, 1])
    tvm_args = [tvm.nd.array(x, ctx=ctx) for x in np_args]
    ctx.sync()

    func_ref(*tvm_args_ref)
    func(*tvm_args)
    ctx.sync()

    np_args_ref[5] = tvm_args_ref[5].asnumpy()
    np_args[5] = tvm_args[5].asnumpy()
    np_args[5] = np_args[5].transpose([0, 3, 1, 2])
    np.testing.assert_allclose(np_args[5], np_args_ref[5], atol=1e-3)

    # additional test against pytorch
    torch_args = [torch.tensor(x) for x in np_args_ref]
    torch_res = torch.nn.functional.conv2d(torch_args[0], torch_args[1],
            stride=args[6], padding=args[7], dilation=1, groups=1)
    torch_res = torch.nn.functional.relu((torch_res + torch_args[2]) * torch_args[4] + torch_args[3])
    torch_res = torch_res.numpy()
    np.testing.assert_allclose(np_args[5], torch_res, atol=1e-3)


@ansor.register_workload_func
def conv2d_transpose_nchw_bias(N, H, W, CI, CO, KH, KW, strides, padding):
    data = te.placeholder((N, CI, H, W), name='data')
    kernel = te.placeholder((CI, CO, KH, KW), name='kernel')
    bias = te.placeholder((CO, 1, 1), name='bias')
    conv = topi.nn.conv2d_transpose_nchw(data, kernel, (strides, strides), padding, out_dtype=data.dtype)
    out = topi.add(conv, bias)
    return [data, kernel, bias, out]

@ansor.register_workload_func
def conv2d_transpose_nhwc_bias(N, H, W, CI, CO, KH, KW, strides, padding):
    data = te.placeholder((N, H, W, CI), name='data')
    kernel = te.placeholder((KH, KW, CI, CO), name='kernel')
    bias = te.placeholder((CO,), name='bias')
    conv = topi.nn.conv2d_transpose_nhwc(data, kernel, (strides, strides), padding, out_dtype=data.dtype)
    out = topi.add(conv, bias)
    return [data, kernel, bias, out]


def test_conv2d_transpose():
    shape_list = [
        (1, 4, 4, 512, 256, 4, 4, 2, 1),
        (1, 8, 8, 256, 128, 4, 4, 2, 1),
        (1, 16, 16, 128, 64, 4, 4, 2, 1),
        (1, 32, 32, 64, 3, 4, 4, 2, 1),
    ]

    for args in shape_list:
        target = target_host = 'llvm'

        dag_ref = ansor.ComputeDAG(conv2d_transpose_nchw_bias(*args))
        s_ref, bufs_ref = dag_ref.apply_steps_from_state(dag_ref.get_init_state())
        func_ref = tvm.build(s_ref, bufs_ref, target='llvm')
        ctx_ref = tvm.cpu()
        np_args_ref = [np.random.randn(*topi.get_const_tuple(x.shape)).astype(x.dtype) for x in bufs_ref]
        tvm_args_ref = [tvm.nd.array(x, ctx=ctx_ref) for x in np_args_ref]

        dag = ansor.ComputeDAG(conv2d_transpose_nhwc_bias(*args))
        s, bufs = dag.apply_steps_from_state(dag.get_init_state())
        func = tvm.build(s, bufs, target=target, target_host=target_host)
        ctx = tvm.context(str(target), 0)
        np_args = [np.array(x) for x in np_args_ref]
        np_args[0] = np_args[0].transpose([0, 2, 3, 1])
        np_args[1] = np_args[1].transpose([2, 3, 0, 1])
        np_args[2] = np_args[2].reshape((args[4],))
        np_args[3] = np_args[3].transpose([0, 2, 3, 1])
        tvm_args = [tvm.nd.array(x, ctx=ctx) for x in np_args]
        ctx.sync()

        func_ref(*tvm_args_ref)
        func(*tvm_args)
        ctx.sync()

        np_args_ref[3] = tvm_args_ref[3].asnumpy()
        np_args[3] = tvm_args[3].asnumpy()
        np_args[3] = np_args[3].transpose([0, 3, 1, 2])
        np.testing.assert_allclose(np_args[3], np_args_ref[3], rtol=1e-3, atol=1e-3)

        # additional test against pytorch
        torch_args = [torch.tensor(x) for x in np_args_ref]
        torch_res = torch.nn.functional.conv_transpose2d(torch_args[0], torch_args[1],
                stride=args[7], padding=args[8], dilation=1, groups=1)
        torch_res = torch_res + torch_args[2]
        torch_res = torch_res.numpy()
        np.testing.assert_allclose(np_args[3], torch_res, rtol=1e-4, atol=1e-4)

def test_sparse_dense_bsr(M, N, K, BS_R, BS_C, density, use_relu):
    X_np = np.random.randn(M, K).astype("float32")
    W_sp_np = random_bsr_matrix(N, K, BS_R, BS_C, density=density, dtype="float32")
    W_np = W_sp_np.todense()
    Y_np = X_np.dot(W_np.T)
    if use_relu:
        Y_np = np.maximum(Y_np, 0.0)

    W_data = te.placeholder(shape=W_sp_np.data.shape, dtype=str(W_sp_np.data.dtype))
    W_indices = te.placeholder(shape=W_sp_np.indices.shape, dtype=str(W_sp_np.indices.dtype))
    W_indptr = te.placeholder(shape=W_sp_np.indptr.shape, dtype=str(W_sp_np.indptr.dtype))
    X = te.placeholder(shape=X_np.shape, dtype=str(X_np.dtype))

    target = 'llvm'
    ctx = tvm.context(target)

    Y = sparse_dense_bsr_compute(X, W_data, W_indices, W_indptr)
    if use_relu:
        Y = topi.nn.relu(Y)
    s = te.create_schedule([Y.op])
    func = tvm.build(s, [X, W_data, W_indices, W_indptr, Y], target)
    Y_tvm = tvm.nd.array(np.zeros(Y_np.shape, dtype=Y_np.dtype), ctx=ctx)
    func(tvm.nd.array(X_np, ctx=ctx),
         tvm.nd.array(W_sp_np.data, ctx=ctx),
         tvm.nd.array(W_sp_np.indices, ctx=ctx),
         tvm.nd.array(W_sp_np.indptr, ctx=ctx),
         Y_tvm)
    tvm.testing.assert_allclose(Y_tvm.asnumpy(), Y_np, atol=1e-4, rtol=1e-4)


def test_sparse_conv2d_csr(N, H, W, CI, CO, KH, KW, strides, padding, dilation, density, use_relu):
    dtype = 'float32'
    X_np = np.random.randn(N, CI, H, W).astype(dtype)
    W_sp_np = random_csr_matrix(CO, CI * KH * KW, density=density, dtype=dtype)
    W_np = np.array(W_sp_np.todense()).reshape((CO, CI, KH, KW))
    Y_np = topi.testing.conv2d_nchw_python(X_np, W_np, strides, padding).astype(dtype)

    if use_relu:
        Y_np = np.maximum(Y_np, 0.0)

    W_data = te.placeholder(shape=W_sp_np.data.shape, dtype=str(W_sp_np.data.dtype))
    W_indices = te.placeholder(shape=W_sp_np.indices.shape, dtype=str(W_sp_np.indices.dtype))
    W_indptr = te.placeholder(shape=W_sp_np.indptr.shape, dtype=str(W_sp_np.indptr.dtype))
    X = te.placeholder(shape=X_np.shape, dtype=str(X_np.dtype))

    target = 'llvm'
    ctx = tvm.context(target)

    Y = sparse_conv2d_csr_compute(X, W_data, W_indices, W_indptr, (KH, KW), strides, padding, dilation)
    if use_relu:
        Y = topi.nn.relu(Y)

    s = te.create_schedule([Y.op])
    func = tvm.build(s, [X, W_data, W_indices, W_indptr, Y], target)
    Y_tvm = tvm.nd.array(np.zeros(Y_np.shape, dtype=Y_np.dtype), ctx=ctx)
    func(tvm.nd.array(X_np, ctx=ctx),
         tvm.nd.array(W_sp_np.data, ctx=ctx),
         tvm.nd.array(W_sp_np.indices, ctx=ctx),
         tvm.nd.array(W_sp_np.indptr, ctx=ctx),
         Y_tvm)

    tvm.testing.assert_allclose(Y_tvm.asnumpy(), Y_np, atol=1e-4, rtol=1e-4)

if __name__ == "__main__":
    test_sparse_dense_bsr(128, 64, 32, 16, 1, 0.15, False)
    test_sparse_dense_bsr(128, 64, 32, 16, 1, 0.15, True)

    test_sparse_conv2d_csr(1, 7, 7, 128, 128, 1, 1, 1, 0, 1, 0.15, False)
    test_sparse_conv2d_csr(1, 7, 7, 128, 128, 3, 3, 1, 1, 1, 0.15, True)

    test_conv2d()
    test_conv2d_transpose()


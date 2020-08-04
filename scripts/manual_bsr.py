"""Manual baseline schedule for bsr matmul and conv2d_1x1"""
import topi
import tvm
from tvm import te
import numpy as np

from topi.util import get_const_int, traverse_inline
from tvm.ansor import measure
from common import measure_schedule, random_bsr_matrix

def random_bsr_matrix(M, N, BS_R, BS_C, density, dtype):
    import itertools
    import scipy.sparse as sp
    np.random.seed(42)
    Y = np.zeros((M, N), dtype=dtype)
    assert M % BS_R == 0
    assert N % BS_C == 0
    nnz = int(density * M * N)
    num_blocks = int(nnz / (BS_R * BS_C)) + 1
    candidate_blocks = np.asarray(list(itertools.product(range(0, M, BS_R), range(0, N, BS_C))))
    assert candidate_blocks.shape[0] == M // BS_R * N // BS_C
    chosen_blocks = candidate_blocks[np.random.choice(candidate_blocks.shape[0], size=num_blocks, replace=False)]
    for i in range(len(chosen_blocks)):
        r, c = chosen_blocks[i]
        Y[r:r + BS_R, c:c + BS_C] = np.random.randn(BS_R, BS_C)
    s = sp.bsr_matrix(Y, blocksize=(BS_R, BS_C))
    assert s.data.shape == (num_blocks, BS_R, BS_C)
    assert s.indices.shape == (num_blocks, )
    assert s.indptr.shape == (M // BS_R + 1, )
    return s

def manual_sparse_dense_bsr(M, N, K, BS_R, BS_C, density, use_relu, target):
    dtype = "float32"
    W_sp_np = random_bsr_matrix(N, K, BS_R, BS_C, density=density, dtype=dtype)

    # register these special buffers for measurement
    prefix = "sparse_dense_bsr_%d_%d_%d_%d_%d_%.2f_" % (M, N, K, BS_R, BS_C, density)
    measure.register_special_buffer(prefix + "W_data", W_sp_np.data)
    measure.register_special_buffer(prefix + "W_indices", W_sp_np.indices)
    measure.register_special_buffer(prefix + "W_indptr", W_sp_np.indptr)

    W_data = te.placeholder(shape=W_sp_np.data.shape, dtype=str(W_sp_np.data.dtype),
                            name=prefix + "W_data")
    W_indices = te.placeholder(shape=W_sp_np.indices.shape, dtype=str(W_sp_np.indices.dtype),
                               name=prefix + "W_indices")
    W_indptr = te.placeholder(shape=W_sp_np.indptr.shape, dtype=str(W_sp_np.indptr.dtype),
                              name=prefix + "W_indptr")
    X = te.placeholder(shape=(M, K), dtype=dtype, name='X')

    with target:
        Y = topi.nn.sparse_dense(X, W_data, W_indices, W_indptr)
        if use_relu:
            Y = topi.nn.relu(Y)
        s = topi.x86.schedule_sparse_dense([Y])

    bufs = X, W_data, W_indices, W_indptr, Y
    print(tvm.lower(s, bufs, simple_mode=True))
    return np.mean(measure_schedule(s, bufs, target))

def manual_conv2d_1x1_bsr(N, H, W, CI, CO, BS_R, BS_C, density, use_relu, target):
    dtype = "float32"
    W_sp_np = random_bsr_matrix(CI, CO, BS_R, BS_C, density=density, dtype=dtype)

    prefix = "conv2d_1x1_bsr_%d_%d_%d_%d_%d_%d_%d_%.2f_" % (N, H, W, CI, CO, BS_R, BS_C, density)
    measure.register_special_buffer(prefix + "W_data", W_sp_np.data)
    measure.register_special_buffer(prefix + "W_indices", W_sp_np.indices)
    measure.register_special_buffer(prefix + "W_indptr", W_sp_np.indptr)

    W_data = te.placeholder(shape=W_sp_np.data.shape, dtype=str(W_sp_np.data.dtype),
                            name=prefix + "W_data")
    W_indices = te.placeholder(shape=W_sp_np.indices.shape, dtype=str(W_sp_np.indices.dtype),
                               name=prefix + "W_indices")
    W_indptr = te.placeholder(shape=W_sp_np.indptr.shape, dtype=str(W_sp_np.indptr.dtype),
                              name=prefix + "W_indptr")
    data = te.placeholder(shape=(N, H, W, CI), dtype=dtype, name='data')

    (NB_plus_1, ) = W_sp_np.indptr.shape
    NB = NB_plus_1 - 1
    oshape = (N, H, W, NB, BS_R)

    def f(n, h, w, nb, r):
        row_start = W_indptr[nb]
        row_end = W_indptr[nb + 1]
        row_elems = row_end - row_start
        elem_idx = te.reduce_axis((0, row_elems), name="elem_idx")
        jj = row_start + elem_idx
        c = te.reduce_axis((0, BS_C), name="c")
        j = W_indices[jj]
        block_ij_val = W_data[jj][r][c]
        x_val = data[n, h, w, BS_C * j + c]
        return te.sum(block_ij_val * x_val, axis=[elem_idx, c])

    Y = te.compute(
        oshape,
        f,
        name="sparse_dense_bsrmv_block",
        tag="sparse_dense_bsrmv_block"
    )
    output = te.compute(
        (N, H, W, NB * BS_R),
        lambda n, h, w, c : Y[n, h, w, c // BS_R, c % BS_R],
        name="sparse_dense_bsrmv",
        tag="sparse_dense_bsrmv"
    )
    if use_relu:
        output = topi.nn.relu(output)
    outs = [output]

    s = te.create_schedule([x.op for x in outs])
    def callback(op):
        simd_width = 8
        if op.tag == "sparse_dense_bsrmv":
            Y_bsrmv = op.input_tensors[0]
            assert Y_bsrmv.op.tag == "sparse_dense_bsrmv_block"
            Y_reshape = op
            (nn, hh, ww, nb, br) = s[Y_bsrmv].op.axis
            BS_R = get_const_int(br.dom.extent)
            (elem_idx, c) = s[Y_bsrmv].op.reduce_axis
            s[Y_bsrmv].reorder(nn, hh, ww, nb, elem_idx, br, c)
            s[Y_bsrmv].vectorize(br)
            (nn, hh, ww, no) = s[Y_reshape].op.axis
            (noo, noi) = s[Y_reshape].split(no, BS_R)

            s[Y_bsrmv].compute_at(s[Y_reshape], noi)
            s[Y_reshape].vectorize(noi)
            if op != s[outs[0]].op:
                axes = s[outs[0]].op.axis
                (y_o, y_i) = s[outs[0]].split(
                    axes[-1], 2 * simd_width)
                s[Y_reshape].compute_at(s[outs[0]], y_o)
                nn, hh, ww, cc = s[outs[0]].op.axis
                s[outs[0]].parallel(s[outs[0]].fuse(nn, hh, ww))
                s[outs[0]].vectorize(y_i)
            else:
                oo = s[Y_reshape].fuse(nn, hh, ww, noo)
                s[Y_reshape].parallel(oo)
    traverse_inline(s, output.op, callback)

    bufs = data, W_data, W_indices, W_indptr, output
    print(tvm.lower(s, bufs, simple_mode=True))
    return np.mean(measure_schedule(s, bufs, target))


if __name__ == "__main__":
    if True:
        M, N, K, BS_R, BS_C, density = 128, 3072, 768, 16, 1, 0.15
        target = tvm.target.create('llvm -mcpu=core-avx2')
        cost = manual_sparse_dense_bsr(M, N, K, BS_R, BS_C, density, use_relu=False, target=target)
        print("Manual sparse_dense cost: %.6f ms" % (np.mean(cost) * 1e3))

    if False:
        N, H, W, CI, CO, BS_R, BS_C, density = 1, 56, 56, 512, 512, 16, 16, 0.15
        target = tvm.target.create('llvm -mcpu=core-avx2')
        cost = manual_conv2d_1x1_bsr(N, H, W, CI, CO, BS_R, BS_C, density, use_relu=False, target=target)
        print("Manual conv2d_1x1 cost: %.6f ms" % (np.mean(cost) * 1e3))

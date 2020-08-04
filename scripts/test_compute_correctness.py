"""Test the correctness of compute definition"""

import numpy as np

import topi
import tvm
from tvm import te

from common import sparse_dense_bsr_compute, random_bsr_matrix

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

if __name__ == "__main__":
    test_sparse_dense_bsr(128, 3072, 768, 16, 1, 0.15, False)
    test_sparse_dense_bsr(128, 3072, 768, 16, 1, 0.15, True)

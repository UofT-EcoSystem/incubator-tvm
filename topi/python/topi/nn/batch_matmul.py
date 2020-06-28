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
"""Binary Neural Network (BNN) Operators"""
# pylint: disable=invalid-name
from tvm import te
from ..util import get_const_tuple

def batch_matmul(x, y, weight_transposed=True):
    """Computes batch matrix multiplication of `x` and `y` when `x` and `y` are
    data in batch.

    Parameters
    ----------
    x : tvm.te.Tensor
        3-D with shape [batch, M, K]

    y : tvm.te.Tensor
        if weight_transposed is True,  this is a 3-D with shape [batch, N, K]
        if weight_transposed is False, this is a 3-D with shape [batch, K, N]

    weight_transposed: bool
        Whether the layout of y tensor is transposed

    Returns
    -------
    output : tvm.te.Tensor
        3-D with shape [batch, M, N]
    """
    assert len(x.shape) == 3 and len(y.shape) == 3, "only support 3-dim batch_matmul"
    x_shape = get_const_tuple(x.shape)
    y_shape = get_const_tuple(y.shape)
    assert x_shape[0] == y_shape[0], "batch dimension doesn't match"
    batch, M, K = x.shape
    k = te.reduce_axis((0, K), name='k')
    if weight_transposed:
        N = y.shape[1]
        assert x_shape[2] == y_shape[2], "shapes of x and y is inconsistant"
        return te.compute((batch, M, N),
                          lambda b, i, j: te.sum(x[b, i, k] * y[b, j, k], axis=k),
                          tag='batch_matmul')
    else:
        N = y.shape[2]
        assert x_shape[2] == y_shape[1], "shapes of x and y is inconsistant"
        return te.compute((batch, M, N),
                          lambda b, i, j: te.sum(x[b, i, k] * y[b, k, j], axis=k),
                          tag='batch_matmul')


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
# pylint: disable=invalid-name, unused-variable, no-else-return, unused-argument, import-outside-toplevel
"""Conv2D schedule for ARM CPU"""
from __future__ import absolute_import as _abs

import tvm
from tvm import te
from tvm import autotvm
from tvm import ansor
import tvm.contrib.nnpack

from ..util import traverse_inline, get_const_tuple
from .. import nn
from ..nn.util import get_const_int, get_pad_tuple
from ..nn.winograd_util import winograd_transform_matrices
from .conv2d_spatial_pack import conv2d_spatial_pack_nchw, \
    conv2d_spatial_pack_nhwc, \
    schedule_conv2d_spatial_pack_nchw, \
    schedule_conv2d_spatial_pack_nhwc
from .cortex_m7.conv2d import direct_simd


@autotvm.register_topi_compute("conv2d_nchw_spatial_pack.arm_cpu")
def conv2d_nchw_spatial_pack(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Compute conv2d with NCHW layout"""
    return conv2d_spatial_pack_nchw(cfg, data, kernel, strides, padding,
                                    dilation, out_dtype, num_tile=2)


@autotvm.register_topi_schedule("conv2d_nchw_spatial_pack.arm_cpu")
def schedule_conv2d_nchw_spatial_pack(cfg, outs):
    """Create schedule for conv2d_nchw"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        # schedule conv2d
        if 'spatial_conv2d_output' in op.tag:
            output = op.output(0)
            conv = op.input_tensors[0]

            data_vec = conv.op.input_tensors[0]
            data_pad = data_vec.op.input_tensors[0]
            s[data_pad].compute_inline()

            kernel_vec = conv.op.input_tensors[1]
            if kernel_vec.op.name == 'kernel_vec':
                kernel = kernel_vec.op.input_tensors[0]
            else:
                kernel = kernel_vec
            if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
                s[kernel].compute_inline()

            schedule_conv2d_spatial_pack_nchw(cfg, s, data_vec, kernel_vec,
                                              conv, output, outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute("conv2d_nhwc_spatial_pack.arm_cpu")
def conv2d_nhwc_spatial_pack(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Compute conv2d with NHWC layout"""
    return conv2d_spatial_pack_nhwc(cfg, data, kernel, strides, padding,
                                    dilation, out_dtype)


@autotvm.register_topi_schedule("conv2d_nhwc_spatial_pack.arm_cpu")
def schedule_conv2d_nhwc_spatial_pack(cfg, outs):
    """Create schedule for conv2d_nhwc"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if 'spatial_conv_output_NHWC' in op.tag:
            schedule_conv2d_spatial_pack_nhwc(cfg, s, op, outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute("conv2d_nchw_winograd.arm_cpu")
def conv2d_nchw_winograd(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Compute conv2d_nchw layout using Winograd with weight transform"""
    tile_size = 4
    return _decl_winograd(cfg, data, kernel, strides, padding, dilation,
                          out_dtype, tile_size)


@autotvm.register_topi_schedule("conv2d_nchw_winograd.arm_cpu")
def schedule_conv2d_nchw_winograd(cfg, outs):
    """Create schedule for conv2d_nchw_winograd"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if 'winograd_conv2d_output' in op.tag:
            output = op.output(0)
            _schedule_winograd(cfg, s, output, outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s


def _decl_winograd(cfg, data, kernel, strides, padding, dilation, out_dtype, tile_size):
    N, CI, IH, IW = get_const_tuple(data.shape)

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    if len(kernel.shape) == 4:
        if dilation_h != 1 or dilation_w != 1:
            kernel = nn.dilate(kernel, (1, 1, dilation_h, dilation_w))
        pre_computed = False
        CO, _, KH, KW = get_const_tuple(kernel.shape)
    else:
        assert (dilation_h, dilation_w) == (1, 1), "Does not support dilation"
        pre_computed = True
        H_CAT, W_CAT, CO, CI, VC = get_const_tuple(kernel.shape)
        CO *= VC
        KH, KW = H_CAT - tile_size + 1, W_CAT - tile_size + 1
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    pt, pl, pb, pr = get_pad_tuple(padding, (KH, KW))

    assert KH == 3 and KW == 3 and HSTR == 1 and WSTR == 1
    data_pad = nn.pad(data, (0, 0, pt, pl), (0, 0, pb, pr), name="data_pad")

    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod

    r = KW
    m = tile_size
    alpha = m + r - 1
    A, B, G = winograd_transform_matrices(m, r, out_dtype)

    K = CO
    C = CI

    H = (IH + pt + pb - 3) // HSTR + 1
    W = (IW + pl + pr - 3) // WSTR + 1
    nH, nW = (H + m-1) // m, (W + m-1) // m
    P = N * nH * nW

    cfg.define_split('tile_p', cfg.axis(P), num_outputs=2, filter=lambda x: x.size[-1] <= 16)
    cfg.define_split('tile_k', cfg.axis(K), num_outputs=2, filter=lambda x: x.size[-1] <= 16)
    VP = cfg['tile_p'].size[-1]
    VK = cfg['tile_k'].size[-1]

    # pack input tile
    input_tile = te.compute((C, idxd(P, VP), alpha, alpha, VP),
                            lambda c, b, eps, nu, bb:
                            data_pad[idxd(b*VP + bb, nH*nW), c,
                                     idxm(idxd(b*VP + bb, nW), nH) * m + eps,
                                     idxm(b*VP + bb, nW) * m + nu],
                            name='d')

    if autotvm.GLOBAL_SCOPE.in_tuning:
        VC = cfg['tile_k'].size[-1]
        kvshape = (KH + tile_size - 1, KW + tile_size - 1, idxd(CO, VC), CI, VC)
        U = tvm.te.placeholder(kvshape, kernel.dtype, name="U")
    else:
        # transform kernel
        if pre_computed:
            U = kernel
        else:
            r_kh = te.reduce_axis((0, KH), 'r_kh')
            r_kw = te.reduce_axis((0, KW), 'r_kw')
            U = te.compute((alpha, alpha, idxd(K, VK), C, VK), lambda eps, nu, k, c, kk:
                           te.sum(kernel[k * VK + kk][c][r_kh][r_kw].astype(out_dtype) *
                                  G[eps][r_kh] * G[nu][r_kw], axis=[r_kh, r_kw]), name='U')

    # transform image
    r_eps = te.reduce_axis((0, alpha), 'r_eps')
    r_nu = te.reduce_axis((0, alpha), 'r_nu')
    V = te.compute((alpha, alpha, idxd(P, VP), C, VP), lambda eps, nu, b, c, bb:
                   te.sum(input_tile[c][b][r_eps][r_nu][bb].astype(out_dtype) *
                          B[r_eps][eps] * B[r_nu][nu], axis=[r_eps, r_nu]), name='V')

    # batch gemm
    c = te.reduce_axis((0, C), name='c')
    M = te.compute((alpha, alpha, K, P), lambda eps, nu, k, b:
                   te.sum(U[eps][nu][idxd(k, VK)][c][idxm(k, VK)] *
                          V[eps][nu][idxd(b, VP)][c][idxm(b, VP)], axis=c), name='M')

    # inverse transform
    r_eps = te.reduce_axis((0, alpha), 'r_eps')
    r_nu = te.reduce_axis((0, alpha), 'r_nu')
    Y = te.compute((K, P, m, m), lambda k, b, vh, vw:
                   te.sum(M[r_eps][r_nu][k][b] * A[r_eps][vh] * A[r_nu][vw],
                          axis=[r_eps, r_nu]), name='Y')

    # unpack output
    output = te.compute((N, K, H, W), lambda n, k, h, w:
                        Y[k][n * nH * nW + idxd(h, m) * nW + idxd(w, m),
                             idxm(h, m), idxm(w, m)],
                        name='output', tag='winograd_conv2d_output')

    # we have to manually assign effective GFLOP for winograd
    cfg.add_flop(2 * N * K * H * W * KH * KW * C)
    return output


def _schedule_winograd(cfg, s, output, last):
    Y = output.op.input_tensors[0]
    M, A = Y.op.input_tensors
    U, V = M.op.input_tensors
    d, B = V.op.input_tensors
    data_pad = d.op.input_tensors[0]

    # padding
    s[data_pad].compute_inline()

    # pack input tiles
    s[d].compute_inline()

    # transform kernel
    if isinstance(U.op, tvm.te.ComputeOp):
        kernel, G = U.op.input_tensors
        s[G].compute_inline()
        eps, nu, k, c, kk, = s[U].op.axis
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # kernel transformation will be pre-computed during compilation, so we skip
            # this part to make tuning records correct
            s[U].pragma(eps, 'debug_skip_region')
        else:
            r_kh, r_kw = s[U].op.reduce_axis
            s[U].reorder(k, c, eps, nu, r_kh, r_kw, kk)
            for axis in [eps, nu, r_kh, r_kw]:
                s[U].unroll(axis)
            s[U].vectorize(kk)
            s[U].parallel(k)

        if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
            s[kernel].compute_inline()

    # transform image
    DD = s.cache_read(d, 'global', [V])
    s[B].compute_inline()
    eps, nu, b, c, bb = s[V].op.axis
    r_eps, r_nu = s[V].op.reduce_axis
    s[V].reorder(b, c, eps, nu, r_eps, r_nu, bb)
    for axis in [eps, nu, r_eps, r_nu]:
        s[V].unroll(axis)
    s[DD].compute_at(s[V], c)
    s[V].vectorize(bb)
    s[V].parallel(b)

    # batch gemm
    eps, nu, k, b = s[M].op.axis
    c = s[M].op.reduce_axis[0]
    cfg.define_split('tile_c', c, num_outputs=2, filter=lambda x: x.size[-1] <= 16)
    co, ci = cfg['tile_c'].apply(s, M, c)
    xo, xi = cfg['tile_p'].apply(s, M, b)
    s[M].reorder(eps, nu, xo, co, k, ci, xi)
    cfg.define_annotate('ann_reduce', [ci], policy='try_unroll')
    cfg.define_annotate('ann_spatial', [k, xi], policy='try_unroll_vec')
    cfg['ann_reduce'].apply(s, M, [ci],
                            axis_lens=[cfg['tile_c'].size[-1]],
                            max_unroll=16,
                            cfg=cfg)
    cfg['ann_spatial'].apply(s, M, [k, xi])

    # inverse transform
    s[A].compute_inline()
    k, b, vh, vw = s[Y].op.axis
    r_eps, r_nu = s[Y].op.reduce_axis
    for axis in [vh, vw, r_eps, r_nu]:
        s[Y].unroll(axis)

    # output
    n, co, h, w = s[last].op.axis
    co, coi = cfg['tile_k'].apply(s, last, co)
    p = s[last].fuse(n, co)
    s[M].compute_at(s[last], p)
    s[last].parallel(p)

    MM = s.cache_read(M, 'global', [Y])
    m = get_const_int(V.shape[0]) + 1 - 3
    ho, wo, hi, wi = s[last].tile(h, w, m, m)
    s[Y].compute_at(s[last], wo)
    s[MM].compute_at(s[last], wo)

    if output != last:
        s[output].compute_inline()


@autotvm.register_topi_compute("conv2d_nchw_winograd_nnpack.arm_cpu")
def conv2d_nchw_winograd_nnpack(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Compute conv2d_nchw using nnpack Winograd implementation"""
    dtype = data.dtype
    if dtype == "float32":
        return _conv2d_arm_cpu_winograd_nnpack(
            cfg, data, kernel, strides, padding, dilation, out_dtype,
            tvm.contrib.nnpack.ConvolutionAlgorithm.WT_8x8)
    elif dtype == "float16":
        return _conv2d_arm_cpu_winograd_nnpack(
            cfg, data, kernel, strides, padding, dilation, out_dtype,
            tvm.contrib.nnpack.ConvolutionAlgorithm.WT_8x8_FP16)
    else:
        raise ValueError("Unsupported data type {} for conv2d winograd nnpack".
                         format(dtype))


@autotvm.register_topi_schedule("conv2d_nchw_winograd_nnpack.arm_cpu")
def schedule_conv2d_nchw_winograd_nnpack(cfg, outs):
    """Create schedule for conv2d_nchw_winograd_nnpack"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if 'winograd_nnpack_conv2d_output' in op.tag:
            output = op.output(0)
            _schedule_winograd_nnpack(cfg, s, output, outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s


def _conv2d_arm_cpu_winograd_nnpack(
        cfg, data, kernel, strides, padding, dilation, out_dtype, convolution_algorithm):
    """ TOPI compute callback. Use winograd NNPACK template """
    N, CI, IH, IW = get_const_tuple(data.shape)

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation
    assert (dilation_h, dilation_w) == (1, 1)
    assert len(kernel.shape) == 4
    CO, _, KH, KW = get_const_tuple(kernel.shape)
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    pt, pl, pb, pr = get_pad_tuple(padding, (KH, KW))

    assert KH == 3 and KW == 3 and pt == 1 and pb == 1 and pl == 1 and pr == 1 and HSTR == 1\
        and WSTR == 1
    H = (IH + pt + pb - 3) // HSTR + 1
    W = (IW + pl + pr - 3) // WSTR + 1

    cfg.define_knob('winograd_nnpack_algorithm', [convolution_algorithm])

    assert N == 1
    with tvm.te.tag_scope("winograd_nnpack_conv2d_weight_transform"):
        transformed_kernel = tvm.contrib.nnpack.convolution_inference_weight_transform(
            kernel, algorithm=cfg['winograd_nnpack_algorithm'].val)
        if autotvm.GLOBAL_SCOPE.in_tuning:
            transformed_kernel = te.compute(transformed_kernel.shape, lambda *args: 0.0)

    with tvm.te.tag_scope("winograd_nnpack_conv2d_output"):
        output = tvm.contrib.nnpack.convolution_inference_without_weight_transform(
            data, transformed_kernel,
            bias=None,
            padding=[pt, pb, pl, pr],
            stride=[HSTR, WSTR],
            algorithm=cfg['winograd_nnpack_algorithm'].val)

    # we have to manually assign effective GFLOP for winograd
    cfg.add_flop(2 * N * CI * H * W * KH * KW * CO)
    return output


def _schedule_winograd_nnpack(cfg, s, output, last):
    # Could have bias.

    (X, TK) = output.op.input_tensors[:2]

    # transform kernel
    assert isinstance(TK.op, (te.tensor.ComputeOp, te.tensor.ExternOp, te.tensor.PlaceholderOp))
    if autotvm.GLOBAL_SCOPE.in_tuning and isinstance(TK.op, te.tensor.ComputeOp):
        # kernel transformation will be pre-computed during compilation, so we skip
        # this part to make tuning records correct
        s[TK].pragma(s[TK].op.axis[0], 'debug_skip_region')


@autotvm.register_topi_compute("conv2d_nchw_winograd_nnpack_without_weight_transform.arm_cpu")
def conv2d_nchw_winograd_nnpack_without_weight_transform(
        cfg, data, transformed_kernel, bias, strides, padding, dilation, out_dtype):
    """Compute conv2d_nchw using NNPack winograd without weight transform"""
    N, CI, IH, IW = get_const_tuple(data.shape)
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation
    assert (dilation_h, dilation_w) == (1, 1)
    assert len(transformed_kernel.shape) == 4
    CO, _, _, _ = get_const_tuple(transformed_kernel.shape)
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    KH, KW = 3, 3
    pt, pl, pb, pr = get_pad_tuple(padding, (KH, KW))

    assert KH == 3 and KW == 3 and pt == 1 and pb == 1 and pl == 1 and pr == 1 and HSTR == 1\
        and WSTR == 1
    H = (IH + pt + pb - 3) // HSTR + 1
    W = (IW + pl + pr - 3) // WSTR + 1

    assert N == 1
    with tvm.te.tag_scope("winograd_nnpack_conv2d_output"):
        output = tvm.contrib.nnpack.convolution_inference_without_weight_transform(
            data=data,
            transformed_kernel=transformed_kernel,
            bias=bias,
            padding=[pt, pb, pl, pr],
            stride=[HSTR, WSTR],
            algorithm=cfg['winograd_nnpack_algorithm'].val)

    # we have to manually assign effective GFLOP for winograd
    cfg.add_flop(2 * N * CI * H * W * KH * KW * CO)
    return output


@autotvm.register_topi_schedule("conv2d_nchw_winograd_nnpack_without_weight_transform.arm_cpu")
def schedule_conv2d_nchw_winograd_nnpack_without_weight_transform(cfg, outs):
    """TOPI schedule callback"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if 'winograd_nnpack_conv2d_output' in op.tag:
            output = op.output(0)
            _schedule_winograd_nnpack(cfg, s, output, outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s

@autotvm.register_topi_compute("conv2d_direct_simd.arm_cpu")
def conv2d_direct_simd(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Compute conv2d with SIMD (v7e-m)."""
    return direct_simd.conv2d_direct_simd_compute(
        cfg, data, kernel, strides, padding, dilation, out_dtype)


@autotvm.register_topi_schedule("conv2d_direct_simd.arm_cpu")
def schedule_conv2d_direct_simd(cfg, outs):
    """Create schedule for conv2d_direct_simd"""
    return direct_simd.conv2d_direct_simd_nhwc_schedule(cfg, outs)

def _conv2d_nhwc_winograd_impl(input, weight, strides, padding, dilation, out_dtype, tile_size, pre_computed=False):
    """Conv2D NHWC Winograd implementation.

    Parameters
    ----------
    input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    filter : tvm.Tensor
        4-D with shape [filter_height, filter_width, in_channel, num_filter]

    strides : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    out_dtype : str, optional
        Specifies the output data type.

    tile_size : int
        The size of the tile to use for the Winograd filter

    pre_computed: bool
        Whether the kernel is precomputed

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    N, H, W, CI = get_const_tuple(input.shape)
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    assert (dilation_h, dilation_w) == (1, 1), "Does not support dilation"
    if not pre_computed:
        KH, KW, CI, CO = get_const_tuple(weight.shape)
    else:
        if ansor.GLOBAL_SCOPE.topi_in_compute_rewrite_mode:
            if len(weight.shape) >= 14:
                # For cpu tile structure SSRSRS
                base = len(weight.shape) - 14
                H_CAT = get_const_int(weight.shape[0 + base] * weight.shape[3 + base] *
                                        weight.shape[7 + base] * weight.shape[11 + base])
                W_CAT = get_const_int(weight.shape[1 + base] * weight.shape[4 + base] *
                                        weight.shape[8 + base] * weight.shape[12 + base])
                CO = get_const_int(weight.shape[2 + base] * weight.shape[5 + base] *
                                     weight.shape[9 + base] * weight.shape[13 + base])
                CI = get_const_int(weight.shape[6 + base] * weight.shape[10 + base])
                assert base % 3 == 0
                for i in range(base // 3):
                    H_CAT *= get_const_int(weight.shape[i * 3])
                    W_CAT *= get_const_int(weight.shape[i * 3 + 1])
                    CO *= get_const_int(weight.shape[i * 3 + 2])
            elif len(weight.shape) == 10:
                # For cpu tile structure SRS
                H_CAT = get_const_int(weight.shape[0] * weight.shape[3] * weight.shape[7])
                W_CAT = get_const_int(weight.shape[1] * weight.shape[4] * weight.shape[8])
                CO = get_const_int(weight.shape[2] * weight.shape[5] * weight.shape[9])
                CI = get_const_int(weight.shape[6])
            elif len(weight.shape) == 7:
                # For cpu tile structure SRS
                H_CAT = get_const_int(weight.shape[0] * weight.shape[4])
                W_CAT = get_const_int(weight.shape[1] * weight.shape[5])
                CO = get_const_int(weight.shape[2] * weight.shape[6])
                CI = get_const_int(weight.shape[3])
            elif len(weight.shape) == 4:
                H_CAT, W_CAT, CO, CI = get_const_tuple(weight.shape)
            else:
                raise ValueError("Unhandlede case for weight shape: " + str(weight))
        else:
            assert len(weight.shape) == 4, len(weight.shape)
            H_CAT, W_CAT, CO, CI = get_const_tuple(weight.shape)
        KH, KW = H_CAT - tile_size + 1, W_CAT - tile_size + 1
    pad_t, pad_l, pad_d, pad_r = get_pad_tuple(padding, weight)
    HPAD = pad_t + pad_d
    WPAD = pad_l + pad_r
    HSTR, WSTR = (strides, strides) if isinstance(strides, int) else strides
    assert HSTR == 1 and WSTR == 1 and KH == 3 and KW == 3

    data_pad = nn.pad(input, (0, pad_t, pad_l, 0), (0, pad_d, pad_r, 0), name="data_pad")

    r = KW
    m = tile_size
    alpha = m + r - 1
    A, B, G = winograd_transform_matrices(m, r, out_dtype)

    H = (H + HPAD - KH) // HSTR + 1
    W = (W + WPAD - KW) // WSTR + 1
    nH, nW = (H + m - 1) // m, (W + m - 1) // m
    P = N * nH * nW
    r_kh = te.reduce_axis((0, KH), name='r_kh')
    r_kw = te.reduce_axis((0, KW), name='r_kw')
    if not pre_computed:
        kernel_pack = te.compute((alpha, alpha, CO, CI), lambda eps, nu, co, ci:
                                  te.sum(weight[r_kh][r_kw][ci][co] *
                                         G[eps][r_kh] * G[nu][r_kw],
                                         axis=[r_kh, r_kw]), name='kernel_pack')
    else:
        kernel_pack = weight

    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod
    # pack input tile
    input_tile = te.compute((alpha, alpha, P, CI), lambda eps, nu, p, ci:
                             data_pad[idxdiv(p, (nH * nW))][idxmod(idxdiv(p, nW), nH) * m + eps]
                                     [idxmod(p, nW) * m + nu][ci], name='input_tile')

    # transform data
    r_a = te.reduce_axis((0, alpha), 'r_a')
    r_b = te.reduce_axis((0, alpha), 'r_b')
    data_pack = te.compute((alpha, alpha, P, CI), lambda eps, nu, p, ci:
                            te.sum(input_tile[r_a][r_b][p][ci] * B[r_a][eps] * B[r_b][nu],
                                    axis=[r_a, r_b]), name='data_pack',
                                    attrs={"ansor_no_split_at_inner": ["eps", "nu", "r_a", "r_b"],
                                           "ansor_last_split_is_one": ["ci", "p"],
                                           "ansor_always_unroll": ["eps", "nu", "r_a", "r_b"],
                                           "ansor_no_cache_write": "True",
                                           })

    # do batch gemm
    ci = te.reduce_axis((0, CI), name='ci')
    bgemm = te.compute((alpha, alpha, P, CO), lambda eps, nu, p, co:
                        te.sum(data_pack[eps][nu][p][ci] *
                               kernel_pack[eps][nu][co][ci],
                               axis=[ci]), name='bgemm',
                               attrs={"layout_free_placeholders": [kernel_pack]})

    # inverse transform
    r_a = te.reduce_axis((0, alpha), 'r_a')
    r_b = te.reduce_axis((0, alpha), 'r_b')
    inverse = te.compute((m, m, P, CO), lambda vh, vw, p, co:
                          te.sum(bgemm[r_a][r_b][p][co] * A[r_a][vh] * A[r_b][vw],
                                  axis=[r_a, r_b]), name='inverse',
                          attrs={"ansor_no_split_at_inner": ["vh", "vw", "r_a", "r_b"],
                                 "ansor_always_unroll": ["vh", "vw", "r_a", "r_b"],
                                 "ansor_last_split_is_one": ["co", "p"],
                                 "ansor_no_cache_write": "True",
                                 })

    # output
    output = te.compute((N, H, W, CO), lambda n, h, w, co:
                         inverse[idxmod(h, m),
                                 idxmod(w, m),
                                 n * nH * nW + idxdiv(h, m) * nW + idxdiv(w, m),
                                 co],
                         name='conv2d_winograd',
                         tag='conv2d_winograd_nhwc',
                         attrs={"ansor_no_split_at_outer": ["n","h","w","co"]})
    return output

def conv2d_nhwc_winograd(input, weight, strides, padding, dilation, out_dtype, pre_computed=False):
    """Conv2D NHWC Winograd implementation.

    Parameters
    ----------
    input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    filter : tvm.Tensor
        4-D with shape [filter_height, filter_width, in_channel, num_filter]

    strides : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    out_dtype : str, optional
        Specifies the output data type.

    pre_computed: bool
        Whether the kernel is precomputed

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    tile_size = 4
    return _conv2d_nhwc_winograd_impl(input, weight, strides, padding, dilation, out_dtype, tile_size, pre_computed)

def conv2d_nhwc_winograd_without_weight_transform(input, weight, strides, padding,
                                                  dilation, out_dtype):
    return conv2d_nhwc_winograd(input, weight, strides, padding,
                                dilation, out_dtype, pre_computed=True)
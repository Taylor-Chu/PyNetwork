import pyopencl as cl
from pyopencl.elementwise import ElementwiseKernel
import pyopencl.array as cl_array
import pyopencl.reduction as cl_reduction

import numpy as np

from PyNetwork.gpu.elementwise import *
from PyNetwork.gpu.c_code import *


class GPUOPERATOR:
    """
    GPU operators with return values still stored on GPU. 
    """

    def __init__(self, context, queue):
        self.context = context
        self.queue = queue
        self.program = cl.Program(self.context, c_code).build()

        self.relu_program = ElementwiseKernel(self.context,
                                              "float *x, float *out",
                                              "out[i] = x[i] > 0 ? x[i] : 0.0",
                                              "relu")

        self.exp_program = ElementwiseKernel(self.context, "float *a_gpu, float *out",
                                             "out[i] = exp(a_gpu[i])",
                                             "exponential")

        self.log_program = ElementwiseKernel(self.context, "float *a_gpu, float *out",
                                      "out[i] = log(a_gpu[i])",
                                      "logarithm")
        self.pow_program = ElementwiseKernel(self.context, "float *a_gpu, float b, float *out",
                                      "out[i] = pow(a_gpu[i],b)",
                                      "power")
        self.abs_program = ElementwiseKernel(self.context, "float *a_gpu, float *out",
                                      "out[i] = fabs(a_gpu[i])",
                                      "abs")

        self.clip_program = ElementwiseKernel(self.context, "float *a_gpu, float *out, float vmin, float vmax",
                                      "out[i] = a_gpu[i] < vmin ? vmin : (a_gpu[i] > vmax ? vmax : a_gpu[i])",
                                      "clip")

    def repeat(self, a, n):
        """a is assumed to be on the device, 
            a is 1-d clarray with size m, return a nd-clarray with size m*n"""
        length = len(a)
        out = cl_array.zeros(self.queue, (length, n), dtype=np.float32)

        self.program.repeat(self.queue, (length * n,), None,
                            out.data, a.data, np.int32(n)).wait()
        return out

    def add(self, A, B):
        """A, B are assumed to be on the device"""
        if A.ndim == 1:
            height, width = B.shape
            if len(A) == height:
                A = self.repeat(A, width)
            elif len(A) == width:
                A = self.repeat(A, height).T

        if B.ndim == 1:
            height, width = A.shape
            if len(B) == height:
                B = self.repeat(B, width)
            elif len(B) == width:
                B = self.repeat(B, height).T

        else:
            height, width = A.shape
            heightB, widthB = B.shape
            assert height == heightB and width == widthB, "Arrays have different shapes."

        out = cl_array.zeros_like(A)
        self.program.ew_add(self.queue, (width, height), None,
                            A.data, B.data, np.int32(width), out.data).wait()
        return out

    def sub(self, A, B):
        """A, B are assumed to be on the device"""
        if A.ndim == 1:
            height, width = B.shape
            if len(A) == height:
                A = self.repeat(A, width)
            elif len(A) == width:
                A = self.repeat(A, height).T

        if B.ndim == 1:
            height, width = A.shape
            if len(B) == height:
                B = self.repeat(B, width)
            elif len(B) == width:
                B = self.repeat(B, height).T

        else:
            height, width = A.shape
            heightB, widthB = B.shape
            assert height == heightB and width == widthB, "Arrays have different shapes."
        out = cl_array.zeros_like(A)

        self.program.ew_sub(self.queue, (width, height), None,
                            A.data, B.data, np.int32(width), out.data).wait()
        return out

    def sub1d(self, A, B):
        """A, B are assumed to be on the device"""
        assert len(A) == len(B), "1-d arrays have different length."
        length = len(A)
        out = cl_array.zeros_like(A)

        self.program.ew_sub1d(self.queue, (length,), None,
                              A.data, B.data, np.int32(length), out.data).wait()
        return out

    def div(self, A, B):
        if A.ndim == 1:
            height, width = B.shape
            if len(A) == height:
                A = self.repeat(A, width)
            elif len(A) == width:
                A = self.repeat(A, height).T

        if B.ndim == 1:
            height, width = A.shape
            if len(B) == height:
                B = self.repeat(B, width)
            elif len(B) == width:
                B = self.repeat(B, height).T

        else:
            height, width = A.shape
            heightB, widthB = B.shape
            assert height == heightB and width == widthB, "Arrays have different shapes."
        out = cl_array.zeros_like(A)

        self.program.ew_div(self.queue, (width, height), None,
                            A.data, B.data, np.int32(width), out.data).wait()
        return out

    def mul(self, A, B):
        if A.ndim == 1:
            height, width = B.shape
            if len(A) == height:
                A = self.repeat(A, width)
            elif len(A) == width:
                A = self.repeat(A, height).T

        if B.ndim == 1:
            height, width = A.shape
            if len(B) == height:
                B = self.repeat(B, width)
            elif len(B) == width:
                B = self.repeat(B, height).T

        else:
            height, width = A.shape
            heightB, widthB = B.shape
            assert height == heightB and width == widthB, "Arrays have different shapes."
        out = cl_array.zeros_like(A)

        self.program.ew_mul(self.queue, (width, height), None,
                            A.data, B.data, np.int32(width), out.data).wait()
        return out

    def greater(self, A, B):
        """A, B are assumed to be on the device"""
        height, width = A.shape
        heightB, widthB = B.shape
        assert height == heightB and width == widthB, "Arrays have different shapes."
        out = cl_array.zeros_like(A)

        self.program.ew_greater(self.queue, (width, height), None,
                                A.data, B.data, np.int32(width), out.data).wait()
        return out

    def equal(self, A, B):
        """A, B are assumed to be on the device"""
        height, width = A.shape
        heightB, widthB = B.shape
        assert height == heightB and width == widthB, "Arrays have different shapes."
        out = cl_array.zeros_like(A)

        self.program.ew_equal(self.queue, (width, height), None,
                              A.data, B.data, np.int32(width), out.data).wait()
        return out

    def matmul_local(self, A, B):
        """A, B are assumed to be on the device"""
        heightA, widthA = A.shape
        heightB, widthB = B.shape
        assert widthA == heightB, "Cannot do matrix multiplication."
        C = np.empty((heightA, widthB), dtype=np.float32)
        out = cl_array.to_device(self.queue, C)

        BLOCK_SIZE = min(np.gcd(np.gcd(widthA, heightA), widthB), 16)
        local_size = (BLOCK_SIZE, BLOCK_SIZE)

        self.program.matrixmultiply2dlocal(self.queue, (heightA, widthB), local_size,
                                           np.int32(heightA), np.int32(widthA), np.int32(heightB), np.int32(widthB),
                                           A.data, B.data, out.data).wait()
        return out

    def matmul(self, A, B):
        """A, B are assumed to be on the device"""
        heightA, widthA = A.shape
        heightB, widthB = B.shape
        assert widthA == heightB, "Cannot do matrix multiplication."
        size = heightA * widthB

        C = np.empty((heightA, widthB), dtype=np.float32)
        out = cl_array.to_device(self.queue, C)

        self.program.matrixmultiply1d(self.queue, (size,), None,
                                      np.int32(heightA), np.int32(widthA), np.int32(heightB), np.int32(widthB), A.data,
                                      B.data, out.data).wait()
        return out.reshape((heightA, widthB))

    def matmul2d(self, A, B):
        """A, B are assumed to be on the device"""
        heightA, widthA = A.shape
        heightB, widthB = B.shape
        assert widthA == heightB, "Cannot do matrix multiplication."

        out = cl_array.zeros(self.queue, (heightA, widthB), dtype=np.float32)

        print(A.shape, B.shape)
        self.program.matrixmultiply2d(self.queue, (heightA, widthB), None,
                                      np.int32(heightA), np.int32(widthA), np.int32(heightB), np.int32(widthB), A.data,
                                      B.data, out.data).wait()
        return out

    def sign(self, A):
        """A is assumed to be on the device"""
        height, width = A.shape
        out = cl_array.zeros_like(A)

        self.program.ew_sign(self.queue, (width, height), None,
                             A.data, np.int32(width), out.data).wait()
        return out

    def transpose(self, x):
        """x is assumed to be on the device"""
        global_size = x.shape
        width, height = x.shape
        BLOCK_SIZE = np.gcd(width, height)
        local_size = (BLOCK_SIZE, BLOCK_SIZE)

        out = np.empty((height, width), dtype=np.float32)
        x_transpose = cl_array.to_device(self.queue, out)

        # x_transpose = cl_array.zeros_like(x)
        a_local = cl.LocalMemory(4 * BLOCK_SIZE * (BLOCK_SIZE + 1))

        self.program.transpose(self.queue, global_size, local_size,
                               x_transpose.data, x.data, np.int32(width), np.int32(height), a_local).wait()

        # self.program.naive_transpose(self.queue, (height*width, ), None,
        #                             x_transpose.data, x.data, np.int32(height), np.int32(width)).wait()

        return x_transpose

    def sum(self, A, axis=None):
        """A is assumed to be on the device"""
        heightA, widthA = A.shape

        if axis == None:
            rk = cl_reduction.ReductionKernel(self.context, np.float32, neutral="0", reduce_expr="a+b", map_expr="a[i]",
                                              arguments="__global const float *a")
            output_sum = rk(A)
            return output_sum

        else:
            if axis == 1:
                r_gpu = cl_array.zeros(self.queue, (A.shape[0],), dtype=np.float32)
                self.program.reduce_axis_1(self.queue, (A.shape[0],), None,
                                           A.data, r_gpu.data, np.int32(A.shape[0]), np.int32(A.shape[1])).wait()
                return r_gpu

            r_gpu = cl_array.zeros(self.queue, (A.shape[1],), dtype=np.float32)
            self.program.reduce(self.queue, (A.shape[1],), None,
                                A.data, r_gpu.data, np.int32(A.shape[1]), np.int32(A.shape[0])).wait()
            return r_gpu

    def mean(self, A, axis=None):
        """A is assumed to be on the device"""
        heightA, widthA = A.shape

        if axis == None:
            norm = np.array([heightA * widthA]).astype(np.float32)
            norm_gpu = cl_array.to_device(self.queue, norm)
            rk = cl_reduction.ReductionKernel(self.context, np.float32, neutral="0", reduce_expr="a+b", map_expr="a[i]",
                                              arguments="__global const float *a")
            output_sum = rk(A)
            output = output_sum / norm_gpu
            return output

        else:
            if axis == 1:
                A = A.T.copy()
                norm = np.ones(heightA).astype(np.float32) * widthA
            else:
                norm = np.ones(widthA).astype(np.float32) * heightA

            norm_gpu = cl_array.to_device(self.queue, norm)
            r = np.empty(A.shape[1]).astype(np.float32)
            r_gpu = cl_array.to_device(self.queue, r)
            self.program.reduce(self.queue, (A.shape[0] * A.shape[1],), None,
                                A.data, r_gpu.data, np.int32(A.shape[1]), np.int32(A.shape[0])).wait()
            output = r_gpu / norm_gpu
            return output

    def std(self, A, axis=None):
        mean_square = self.mean(self.pow(A, 2), axis=axis)
        square_mean = self.pow(self.mean(A, axis=axis), 2)
        return self.pow(mean_square - square_mean, 0.5)

    def exp(self, A):
        """A is assumed to be on the device"""
        out = cl_array.zeros_like(A)

        self.exp_program(A, out)
        return out

    def log(self, A):
        """A is assumed to be on the device"""
        out = cl_array.zeros_like(A)

        self.log_program(A, out)
        return out

    def pow(self, A, b):
        """A is assumed to be on the device"""
        out = cl_array.zeros_like(A)
        self.pow_program(A, b, out)
        return out

    def abs(self, A):
        """A is assumed to be on the device"""
        out = cl_array.zeros_like(A)
        self.abs_program(A, out)
        return out

    def relu(self, A):
        """A is assumed to be on the device"""
        out = cl_array.zeros_like(A)
        self.relu_program(A, out)
        return out

    def tanh(self, A):
        """A is assumed to be on the device"""
        out = cl_array.zeros_like(A)
        programme = ElementwiseKernel(self.context, "float *a_gpu, float *out",
                                      "out[i] = tanh(a_gpu[i])",
                                      "tanh")
        programme(A, out)
        return out

    def sigmoid(self, A):
        """A is assumed to be on the device"""
        out = cl_array.zeros_like(A)
        programme = ElementwiseKernel(self.context,
                                      "float *x, float *out",
                                      "out[i] = SIGMOID(x[i])",
                                      "sigmoid",
                                      preamble='#define SIGMOID(x) x > 0 ? 1.0/(1.0 + exp(-x)) : exp(x) / (exp(x) + 1.0)'
                                      )
        programme(A, out)
        return out

    def swish(self, A):
        """A is assumed to be on the device"""
        out = cl_array.zeros_like(A)
        programme = ElementwiseKernel(self.context,
                                      "float *x, float *out",
                                      "out[i] = SWISH(x[i])",
                                      "swish",
                                      preamble='#define SWISH(x) x > 0 ? x/(1.0 + exp(-x)) : x*exp(x) / (exp(x) + 1.0)'
                                      )
        programme(A, out)
        return out

    def clip(self, A, vmin, vmax):
        out = cl_array.zeros_like(A)

        self.clip_program(A, out, np.float32(vmin), np.float32(vmax))
        return out

    def softmax(self, A):
        """A is assumed to be on the device"""

        input_length = A.shape[-1]
        global_shape = A.shape[:-1]

        out = cl_array.zeros(self.queue, global_shape, dtype=np.float32)

        self.program.max_across_last_axis(self.queue, global_shape, None,
                                          A.data, np.int32(input_length), out.data).wait()

        z = self.sub(A, out)
        numerator = self.exp(z)
        denominator = self.sum(numerator, axis=1)
        return self.div(numerator, denominator)
        # np.sum(numerator, axis=-1, keepdims=True)

        # temp = numerator / denominator
        # print(temp.shape)
        # return temp

        # return numerator / denominator

    def dense_predict(self, z, W, b):
        input_length = np.int32(z.shape[1])
        output_length = np.int32(b.shape[0])
        N = len(z)

        out = cl_array.zeros(self.queue, (N, output_length), np.float32)

        device_global_shape = out.shape
        event = self.program.dense_predict(self.queue, device_global_shape, None,
                                           z.data, W.data, b.data,
                                           input_length, output_length, out.data)
        event.wait()
        return out

    def dense_weight_gradient(self, delta, prev_z):
        output_length = np.int32(delta.shape[1])
        input_length = np.int32(prev_z.shape[1])
        N = np.int32(len(prev_z))

        W_grad = cl_array.zeros(self.queue, (output_length, input_length), np.float32)

        device_global_shape = (output_length, input_length)  # Same shape as the weight matrix
        event = self.program.dense_weight_gradient(self.queue, device_global_shape, None,
                                                   delta.data, prev_z.data,
                                                   input_length, output_length, N,
                                                   W_grad.data)
        event.wait()

        return W_grad

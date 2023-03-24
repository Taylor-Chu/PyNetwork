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

    def add(self, A, B):
        """A, B are assumed to be on the device"""
        height, width = A.shape 
        heightB, widthB = B.shape    
        assert height == heightB and width == widthB, "Arrays have different shapes."
        out = cl_array.zeros_like(A)

        self.program.ew_add(self.queue, (width, height), None, 
                            A.data, B.data, np.int32(width), out.data).wait()
        return out
        
    def sub(self, A, B):
        """A, B are assumed to be on the device"""
        height, width = A.shape 
        heightB, widthB = B.shape    
        assert height == heightB and width == widthB, "Arrays have different shapes."
        out = cl_array.zeros_like(A)

        self.program.ew_sub(self.queue, (width, height), None, 
                            A.data, B.data, np.int32(width), out.data).wait()
        return out

    def div(self, A, B):
        """A, B are assumed to be on the device"""
        height, width = A.shape 
        heightB, widthB = B.shape    
        assert height == heightB and width == widthB, "Arrays have different shapes."
        out = cl_array.zeros_like(A)

        self.program.ew_div(self.queue, (width, height), None, 
                            A.data, B.data, np.int32(width), out.data).wait()
        return out

    def mul(self, A, B):
        """A, B are assumed to be on the device"""
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

    def matmul(self, A, B):
        """A, B are assumed to be on the device"""
        heightA, widthA = A.shape
        heightB, widthB = B.shape
        assert widthA == heightB, "Cannot do matrix multiplication."
        C = np.empty((heightA, widthB), dtype=np.float32)
        out = cl_array.to_device(self.queue, C)

        BLOCK_SIZE = 16
        local_size = (BLOCK_SIZE, BLOCK_SIZE)

        self.program.matrixmultiply2dlocal(self.queue, (heightA, widthB), local_size, 
                    np.int32(heightA), np.int32(widthA), np.int32(heightB), np.int32(widthB), A.data, B.data, out.data).wait()
        return out

    def sign(self, A):
        """A is assumed to be on the device"""
        height, width = A.shape    
        out = cl_array.zeros_like(A)

        self.program.ew_sign(self.queue, (width, height), None, 
                               A.data, np.int32(width), out.data).wait()
        return out      
    
    def transpose(self, A):
        """A is assumed to be on the device"""
        global_size = A.shape
        local_size = None

        width, height = A.shape
        A_transpose = cl_array.zeros_like(A)
        self.program.naive_transpose(self.queue, global_size, local_size,
                                     A_transpose.data, A.data, np.int32(width), np.int32(height)).wait()
        # global_size = A.shape
        # local_size = (16, 16)

        # width, height = A.shape

        # A_transpose = cl_array.zeros_like(A)
        # a_local = cl.LocalMemory(4 * 16 * (16 + 1))
        
        # self.program.transpose(self.queue, global_size, local_size,
        #                        A_transpose.data, A.data, np.int32(width), np.int32(height), a_local).wait()
        return A_transpose

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
                A = self.transpose(A).copy()
            
            r = np.empty(A.shape[1]).astype(np.float32)
            r_gpu = cl_array.to_device(self.queue, r)
            self.program.reduce(self.queue, (A.shape[0]*A.shape[1], ), None, 
                                A.data, r_gpu.data, np.int32(A.shape[1]), np.int32(A.shape[0])).wait()
            return r_gpu
    
    def exp(self, A):
        """A is assumed to be on the device"""
        out = cl_array.zeros_like(A)
        programme = ElementwiseKernel(self.context,"float *a_gpu, float *out",
                                          "out[i] = exp(a_gpu[i])",
                                          "exponential")
        
        programme(A, out)
        return out

    def log(self, A):
        """A is assumed to be on the device"""
        out = cl_array.zeros_like(A)
        programme = ElementwiseKernel(self.context,"float *a_gpu, float *out",
                                          "out[i] = log(a_gpu[i])",
                                          "logarithm")
        programme(A, out)
        return out

    def pow(self, A, b):
        """A is assumed to be on the device"""
        out = cl_array.zeros_like(A)
        programme = ElementwiseKernel(self.context,"float *a_gpu, float b, float *out",
                                          "out[i] = pow(a_gpu[i],b)",
                                          "power")
        programme(A, b, out)
        return out
    
    def sqrt(self, A):
        """A is assumed to be on the device"""
        out = cl_array.zeros_like(A)
        programme = ElementwiseKernel(self.context,"float *a_gpu, float *out",
                                          "out[i] = sqrt(a_gpu[i])",
                                          "sqrt")
        programme(A, out)
        return out

    def abs(self, A):
        """A is assumed to be on the device"""
        out = cl_array.zeros_like(A)
        programme = ElementwiseKernel(self.context,"float *a_gpu, float *out",
                                          "out[i] = fabs(a_gpu[i])",
                                          "abs")
        programme(A, out)
        return out

    def relu(self, A):
        """A is assumed to be on the device"""
        out = cl_array.zeros_like(A)
        programme = ElementwiseKernel(self.context,"float *a_gpu, float *out",
                                          "out[i] = 0.5* (a_gpu[i] + fabs(a_gpu[i]))",
                                          "relu")
        programme(A, out)
        return out
    
    def tanh(self, A):
        """A is assumed to be on the device"""
        out = cl_array.zeros_like(A)
        programme = ElementwiseKernel(self.context,"float *a_gpu, float *out",
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
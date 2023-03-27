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

    def repeat(self, a, n):
        """a is assumed to be on the device, 
            a is 1-d clarray with size m, return a nd-clarray with size m*n"""
        length = len(a)
        C = np.zeros((length, n), dtype=np.float32)
        out = cl_array.to_device(self.queue, C)

        self.program.repeat(self.queue, (length*n,), None, 
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

    def matmul(self, A, B):
        """A, B are assumed to be on the device"""
        heightA, widthA = A.shape
        heightB, widthB = B.shape
        assert widthA == heightB, "Cannot do matrix multiplication."
        C = np.empty((heightA, widthB), dtype=np.float32)
        out = cl_array.to_device(self.queue, C)

        BLOCK_SIZE = np.gcd(np.gcd(widthA, heightA), widthB)
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
    
    def transpose(self, x):
        """x is assumed to be on the device"""
        global_size = x.shape
        width, height = x.shape
        BLOCK_SIZE = np.gcd(width, height)
        local_size = (BLOCK_SIZE, BLOCK_SIZE)

        out = np.empty((height, width), dtype=np.float32)
        x_transpose = cl_array.to_device(self.queue, out)

        #x_transpose = cl_array.zeros_like(x)
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
                A = self.transpose(A).copy()
            
            r = np.empty(A.shape[1]).astype(np.float32)
            r_gpu = cl_array.to_device(self.queue, r)
            self.program.reduce(self.queue, (A.shape[0]*A.shape[1], ), None, 
                                A.data, r_gpu.data, np.int32(A.shape[1]), np.int32(A.shape[0])).wait()
            return r_gpu
    
    def mean(self, A, axis=None):
        """A is assumed to be on the device"""
        heightA, widthA = A.shape
        
        if axis == None:
            norm = np.array([heightA*widthA]).astype(np.float32)
            norm_gpu = cl_array.to_device(self.queue,norm)
            rk = cl_reduction.ReductionKernel(self.context, np.float32, neutral="0", reduce_expr="a+b", map_expr="a[i]",
                            arguments="__global const float *a")
            output_sum = rk(A)
            output = self.div(output_sum,norm_gpu)
            return output     
    
        else:
            if axis == 1:
                A = self.transpose(A).copy()
                norm = np.ones(heightA).astype(np.float32)*widthA
            else:
                norm = np.ones(widthA).astype(np.float32)*heightA
            
            norm_gpu = cl_array.to_device(self.queue,norm)
            r = np.empty(A.shape[1]).astype(np.float32)
            r_gpu = cl_array.to_device(self.queue, r)
            self.program.reduce(self.queue, (A.shape[0]*A.shape[1], ), None, 
                                A.data, r_gpu.data, np.int32(A.shape[1]), np.int32(A.shape[0])).wait()
            output = self.div(r,norm_gpu)
            return r_gpu
	
    def std(self,A,axis=None):
        mean_square = self.mean(self.pow(A,2),axis=axis)
        square_mean = self.pow(self.mean(A,axis=axis),2)
        return self.sqrt(self.sub(mean_square, square_mean))        
	
	
	
	
	
	
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
    
   
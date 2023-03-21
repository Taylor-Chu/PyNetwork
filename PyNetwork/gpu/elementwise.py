import pyopencl as cl
from pyopencl.elementwise import ElementwiseKernel
import pyopencl.array as cl_array

import numpy as np
import time 

def addtion(A, B):
    height, width = A.shape 
    heightB, widthB = B.shape    
    assert height == heightB and width == widthB, "Arrays have different shapes."
    C = np.empty((height, width), dtype=np.float32)

    platform = cl.get_platforms()
    devices = platform[0].get_devices()
    context = cl.Context(devices)
    queue = cl.CommandQueue(context)

    elementwise_addition = """
    __kernel void add(__global float *A, __global float *B, int width, __global float *out){
        int col = get_global_id(0);
        int row = get_global_id(1);

        int index = row * width + col;
        out[index] = A[index] + B[index];
    }
    """
    A_gpu = cl_array.to_device(queue, A)
    B_gpu = cl_array.to_device(queue, B)
    out_gpu = cl_array.to_device(queue, C)
    ew_add_program = cl.Program(context, elementwise_addition).build()
    event = ew_add_program.add(queue, (width, height), None, 
                               A_gpu.data, B_gpu.data, np.int32(width), out_gpu.data)
    event.wait()
    return out_gpu.get()    

def subtraction(A, B): 
    height, width = A.shape 
    heightB, widthB = B.shape    
    assert height == heightB and width == widthB, "Arrays have different shapes."
    C = np.empty((height, width), dtype=np.float32)

    platform = cl.get_platforms()
    devices = platform[0].get_devices()
    context = cl.Context(devices)
    queue = cl.CommandQueue(context)

    elementwise_subtraction = """
    __kernel void sub(__global float *A, __global float *B, int width, __global float *out){
        int col = get_global_id(0);
        int row = get_global_id(1);

        int index = row * width + col;
        out[index] = A[index] - B[index];
    }
    """
    A_gpu = cl_array.to_device(queue, A)
    B_gpu = cl_array.to_device(queue, B)
    out_gpu = cl_array.to_device(queue, C)
    ew_add_program = cl.Program(context, elementwise_subtraction).build()
    event = ew_add_program.sub(queue, (width, height), None, 
                               A_gpu.data, B_gpu.data, np.int32(width), out_gpu.data)
    event.wait()
    return out_gpu.get() 

def division(A, B):
    height, width = A.shape 
    heightB, widthB = B.shape    
    assert height == heightB and width == widthB, "Arrays have different shapes."
    C = np.empty((height, width), dtype=np.float32)

    platform = cl.get_platforms()
    devices = platform[0].get_devices()
    context = cl.Context(devices)
    queue = cl.CommandQueue(context)

    elementwise_division = """
    __kernel void div(__global float *A, __global float *B, int width, __global float *out){
        int col = get_global_id(0);
        int row = get_global_id(1);

        int index = row * width + col;
        out[index] = A[index] / B[index];
    }
    """
    A_gpu = cl_array.to_device(queue, A)
    B_gpu = cl_array.to_device(queue, B)
    out_gpu = cl_array.to_device(queue, C)
    ew_add_program = cl.Program(context, elementwise_division).build()
    event = ew_add_program.div(queue, (width, height), None, 
                               A_gpu.data, B_gpu.data, np.int32(width), out_gpu.data)
    event.wait()
    return out_gpu.get() 

def multiplication(A, B):
    height, width = A.shape 
    heightB, widthB = B.shape    
    assert height == heightB and width == widthB, "Arrays have different shapes."
    C = np.empty((height, width), dtype=np.float32)

    platform = cl.get_platforms()
    devices = platform[0].get_devices()
    context = cl.Context(devices)
    queue = cl.CommandQueue(context)

    elementwise_multiplication = """
    __kernel void mul(__global float *A, __global float *B, int width, __global float *out){
        int col = get_global_id(0);
        int row = get_global_id(1);

        int index = row * width + col;
        out[index] = A[index] * B[index];
    }
    """
    A_gpu = cl_array.to_device(queue, A)
    B_gpu = cl_array.to_device(queue, B)
    out_gpu = cl_array.to_device(queue, C)
    ew_add_program = cl.Program(context, elementwise_multiplication).build()
    event = ew_add_program.mul(queue, (width, height), None, 
                               A_gpu.data, B_gpu.data, np.int32(width), out_gpu.data)
    event.wait()
    return out_gpu.get() 

def greater(A, B):
    height, width = A.shape 
    heightB, widthB = B.shape    
    assert height == heightB and width == widthB, "Arrays have different shapes."
    C = np.empty((height, width), dtype=np.float32)

    platform = cl.get_platforms()
    devices = platform[0].get_devices()
    context = cl.Context(devices)
    queue = cl.CommandQueue(context)

    elementwise_greater = """
    __kernel void greater(__global float *A, __global float *B, int width, __global float *out){
        int col = get_global_id(0);
        int row = get_global_id(1);

        int index = row * width + col;
        out[index] = A[index] > B[index];
    }
    """
    A_gpu = cl_array.to_device(queue, A)
    B_gpu = cl_array.to_device(queue, B)
    out_gpu = cl_array.to_device(queue, C)
    ew_add_program = cl.Program(context, elementwise_greater).build()
    event = ew_add_program.greater(queue, (width, height), None, 
                               A_gpu.data, B_gpu.data, np.int32(width), out_gpu.data)
    event.wait()
    return out_gpu.get()

def equal(A, B):
    height, width = A.shape 
    heightB, widthB = B.shape    
    assert height == heightB and width == widthB, "Arrays have different shapes."
    C = np.empty((height, width), dtype=np.float32)

    platform = cl.get_platforms()
    devices = platform[0].get_devices()
    context = cl.Context(devices)
    queue = cl.CommandQueue(context)

    elementwise_equal = """
    __kernel void equal(__global float *A, __global float *B, int width, __global float *out){
        int col = get_global_id(0);
        int row = get_global_id(1);

        int index = row * width + col;
        out[index] = (A[index] == B[index]);
    }
    """
    A_gpu = cl_array.to_device(queue, A)
    B_gpu = cl_array.to_device(queue, B)
    out_gpu = cl_array.to_device(queue, C)
    ew_add_program = cl.Program(context, elementwise_equal).build()
    event = ew_add_program.equal(queue, (width, height), None, 
                               A_gpu.data, B_gpu.data, np.int32(width), out_gpu.data)
    event.wait()
    return out_gpu.get()



def gpu_exp(a):
    a_gpu = cl_array.to_device(queue,a.astype(np.float32))
    out = cl_array.zeros_like(a_gpu)
    exp_program = ElementwiseKernel(context,"float *a_gpu, float *out",
                                          "out[i] = exp(a_gpu[i])",
                                          "exponential")
    exp_program(a_gpu,out)
    return out.get()

def gpu_log(a):
    a_gpu = cl_array.to_device(queue,a.astype(np.float32))
    out = cl_array.zeros_like(a_gpu)
    log_program = ElementwiseKernel(context,"float *a_gpu, float *out",
                                          "out[i] = log(a_gpu[i])",
                                          "logarithm")
    log_program(a_gpu,out)
    return out.get()

def gpu_pow(a,b):
    a_gpu = cl_array.to_device(queue,a.astype(np.float32))
    out = cl_array.zeros_like(a_gpu)
    pow_program = ElementwiseKernel(context,"float *a_gpu, float b, float *out",
                                          "out[i] = pow(a_gpu[i],b)",
                                          "power")
    pow_program(a_gpu,b,out)
    return out.get()

def gpu_sqrt(a):
    a_gpu = cl_array.to_device(queue,a.astype(np.float32))
    out = cl_array.zeros_like(a_gpu)
    sqrt_program = ElementwiseKernel(context,"float *a_gpu, float *out",
                                          "out[i] = sqrt(a_gpu[i])",
                                          "sqrt")
    sqrt_program(a_gpu,out)
    return out.get()

def gpu_abs(a):
    a_gpu = cl_array.to_device(queue,a.astype(np.float32))
    out = cl_array.zeros_like(a_gpu)
    abs_program = ElementwiseKernel(context,"float *a_gpu, float *out",
                                          "out[i] = fabs(a_gpu[i])",
                                          "abs")
    abs_program(a_gpu,out)
    return out.get()

def gpu_relu(a):
    a_gpu = cl_array.to_device(queue,a.astype(np.float32))
    out = cl_array.zeros_like(a_gpu)
    relu_program = ElementwiseKernel(context,"float *a_gpu, float *out",
                                          "out[i] = 0.5* (a_gpu[i] + fabs(a_gpu[i]))",
                                          "relu")
    relu_program(a_gpu,out)
    return out.get()

def gpu_tanh(a):
    a_gpu = cl_array.to_device(queue,a.astype(np.float32))
    out = cl_array.zeros_like(a_gpu)
    tanh_program = ElementwiseKernel(context,"float *a_gpu, float *out",
                                          "out[i] = tanh(a_gpu[i])",
                                          "tanh")
    tanh_program(a_gpu,out)
    return out.get()
def gpu_sigmoid(a):
    a_gpu = cl_array.to_device(queue,a.astype(np.float32))
    out = cl_array.zeros_like(a_gpu)
    sigmoid_program = ElementwiseKernel(context,
                                    "float *x, float *out",
                                    "out[i] = SIGMOID(x[i])",
                                    "sigmoid",
                                    preamble='#define SIGMOID(x) x > 0 ? 1.0/(1.0 + exp(-x)) : exp(x) / (exp(x) + 1.0)'
                                    )
    sigmoid_program(a_gpu,out)
    return out.get()

def gpu_swish(a):
    a_gpu = cl_array.to_device(queue,a.astype(np.float32))
    out = cl_array.zeros_like(a_gpu)
    swish_program = ElementwiseKernel(context,
                                    "float *x, float *out",
                                    "out[i] = SWISH(x[i])",
                                    "swish",
                                    preamble='#define SWISH(x) x > 0 ? x/(1.0 + exp(-x)) : x*exp(x) / (exp(x) + 1.0)'
                                    )
    swish_program(a_gpu,out)
    return out.get()

if __name__=="__main__":
    m, n= 2**8, 2**9
    A = np.random.rand(m, n).astype(np.float32)
    B = np.random.rand(m, n).astype(np.float32)
    np.testing.assert_almost_equal(addtion(A, B), A+B, decimal=3)
    np.testing.assert_almost_equal(subtraction(A, B), A-B, decimal=3)  
    np.testing.assert_almost_equal(division(A, B), A/B, decimal=2)
    np.testing.assert_almost_equal(multiplication(A, B), A*B, decimal=3)  
    np.testing.assert_almost_equal(greater(A, B), A>B, decimal=3)  
    np.testing.assert_almost_equal(equal(A, B), A==B, decimal=3)  




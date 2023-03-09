import pyopencl as cl
from pyopencl.elementwise import ElementwiseKernel
import pyopencl.array as cl_array

import numpy as np
import time 


def matrix_mul1d(A, B):
    heightA, widthA = A.shape
    heightB, widthB = B.shape
    C = np.empty((heightA, widthB), dtype=np.float32)

    platform = cl.get_platforms()
    devices = platform[0].get_devices()
    context = cl.Context(devices)
    queue = cl.CommandQueue(context)

    size = heightA * widthB

    mul_c_code_1d = """
    __kernel void matrixmultiply1d(int heightA, int widthA, int heightB, int widthB, __global float *A, __global float *B, __global float *out){
        int index = get_global_id(0);
        int Arow = index / widthB;
        int Bcol = index % widthB;
        float sum = 0.0f;
        for (int i = 0; i < widthA; i++){
            sum += A[Arow * widthA + i] * B[i * widthB + Bcol];
        }
        out[index] = sum;
    }
    """
    A_gpu = cl_array.to_device(queue, A.flatten())
    B_gpu = cl_array.to_device(queue, B.flatten())
    out_gpu = cl_array.to_device(queue, C.flatten())
    mul_program = cl.Program(context, mul_c_code_1d).build()
    event = mul_program.matrixmultiply1d(queue, (size, ), None, 
                np.int32(heightA), np.int32(widthA), np.int32(heightB), np.int32(widthB), A_gpu.data, B_gpu.data, out_gpu.data)
    event.wait()
    return out_gpu.get().reshape((heightA, widthB))


def matrix_mul2d(A, B):
    heightA, widthA = A.shape
    heightB, widthB = B.shape
    C = np.empty((heightA, widthB), dtype=np.float32)

    platform = cl.get_platforms()
    devices = platform[0].get_devices()
    context = cl.Context(devices)
    queue = cl.CommandQueue(context)

    mul_c_code_2d = """
    __kernel void matrixmultiply2d(int heightA, int widthA, int heightB, int widthB, __global float *A, __global float *B, __global float *out){
        int row = get_global_id(1);
        int col = get_global_id(0);
    
        float sum = 0.0f;
        for (int i=0; i < widthA; i++){
            sum += A[row * widthA + i] * B[i * widthB + col];
        }

        out[row*widthB + col] = sum;
    }
    """
    A_gpu = cl_array.to_device(queue, A)
    B_gpu = cl_array.to_device(queue, B)
    out_gpu = cl_array.to_device(queue, C)
    mul_program = cl.Program(context, mul_c_code_2d).build()
    event = mul_program.matrixmultiply2d(queue, (heightA, widthB), None, 
                np.int32(heightA), np.int32(widthA), np.int32(heightB), np.int32(widthB), A_gpu.data, B_gpu.data, out_gpu.data)
    event.wait()
    return out_gpu.get()   


def matrix_mul_local(A, B):
    """
    Accessing GPU global memory too much will slow down the speed. Instead, we can speed up 
        by manually caching sub-blocks of the matrices in the GPU's on-chip local memory (shared in CUDA)
    """
    heightA, widthA = A.shape
    heightB, widthB = B.shape
    C = np.empty((heightA, widthB), dtype=np.float32)

    platform = cl.get_platforms()
    devices = platform[0].get_devices()
    context = cl.Context(devices)
    queue = cl.CommandQueue(context)

    BLOCK_SIZE = 16
    local_size = (BLOCK_SIZE, BLOCK_SIZE)

    mul_c_code_2d_local = """
    #define BLOCK_SIZE 16

    __kernel void matrixmultiply2dlocal(int heightA, int widthA, int heightB, int widthB, __global float *A, __global float *B, __global float *out){
        int local_row = get_local_id(1);
        int local_col = get_local_id(0);

        int row = get_global_id(1);
        int col = get_global_id(0);

        __local float Ab[BLOCK_SIZE][BLOCK_SIZE];
        __local float Bb[BLOCK_SIZE][BLOCK_SIZE];

        int num = widthA / BLOCK_SIZE;
        float sum = 0.0f;
        
        for (int n=0; n < num; n++){
            int row_b = n*BLOCK_SIZE + local_row;
            int col_b = n*BLOCK_SIZE + local_col;

            Ab[local_row][local_col] = A[row*widthA + col_b];
            Bb[local_row][local_col] = B[row_b*widthB + col];

            barrier(CLK_LOCAL_MEM_FENCE);  

            for (int i=0; i < BLOCK_SIZE; i++){
                sum += Ab[local_row][i] * Bb[i][local_col];
            }  

            barrier(CLK_LOCAL_MEM_FENCE);    
        }

        out[row*widthB + col] = sum;
    }
    """
    A_gpu = cl_array.to_device(queue, A)
    B_gpu = cl_array.to_device(queue, B)
    out_gpu = cl_array.to_device(queue, C)
    mul_program = cl.Program(context, mul_c_code_2d_local).build()
    event = mul_program.matrixmultiply2dlocal(queue, (heightA, widthB), local_size, 
                np.int32(heightA), np.int32(widthA), np.int32(heightB), np.int32(widthB), A_gpu.data, B_gpu.data, out_gpu.data)
    event.wait()
    return out_gpu.get() 


if __name__=="__main__":
    m, n, p = 2**8, 2**9, 2**10
    A = np.random.rand(m, n).astype(np.float32)
    B = np.random.rand(n, p).astype(np.float32)
    np.testing.assert_almost_equal(matrix_mul1d(A, B), A@B, decimal=3)
    np.testing.assert_almost_equal(matrix_mul2d(A, B), A@B, decimal=3)
    np.testing.assert_almost_equal(matrix_mul_local(A, B), A@B, decimal=3)

    trials = 500

    start = time.time()
    for i in range(trials):
        A = np.random.rand(m, n)
        B = np.random.rand(n, p)
        C = matrix_mul1d(A, B)
    end = time.time()
    print(f"GPU matrix multiplication 1d with global memory: average running time = {(end - start)/trials}")
    
    start = time.time()
    for i in range(trials):
        A = np.random.rand(m, n)
        B = np.random.rand(n, p)
        C = matrix_mul2d(A, B)
    end = time.time()
    print(f"GPU matrix multiplication 2d with global memory: average running time = {(end - start)/trials}")

    start = time.time()
    for i in range(trials):
        A = np.random.rand(m, n)
        B = np.random.rand(n, p)
        C = matrix_mul_local(A, B)
    end = time.time()
    print(f"GPU matrix multiplication 2d with local memory: average running time = {(end - start)/trials}")
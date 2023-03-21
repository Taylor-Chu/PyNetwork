import pyopencl as cl
# from pyopencl.elementwise import ElementwiseKernel
import pyopencl.array as cl_array
import pyopencl.reduction as cl_reduction


import numpy as np

def sum(A, axis=None):
    heightA, widthA = A.shape
    

    platform = cl.get_platforms()
    devices = platform[0].get_devices()
    context = cl.Context(devices)
    queue = cl.CommandQueue(context)
    if axis == None:
        a_gpu = cl_array.to_device(queue, A)
        rk = cl_reduction.ReductionKernel(context, np.float32, neutral="0", reduce_expr="a+b", map_expr="a[i]",
                        arguments="__global const float *a")
        output_sum = rk(a_gpu).get()
        return output_sum
    else:
        c_code = """
        __kernel void reduce(__global float *a,
                __global float *r,
                int length1,
                int length2){
            int gj = get_global_id(0); //along row | col number
            float sum = 0.0f;
            for (int k = 0; k < length2; k++){
                sum += a[gj + length1 * k];
            }
            r[gj] = sum;
            
            
        }
        """
        if axis == 1:
            A = A.T.copy()
        program = cl.Program(context, c_code).build()
        r = np.empty(A.shape[1]).astype(np.float32)
        a_gpu = cl_array.to_device(queue, A)
        r_gpu = cl_array.to_device(queue, r)
        program.reduce(queue, (A.shape[0]*A.shape[1], ), None, a_gpu.data, r_gpu.data, np.int32(A.shape[1]), np.int32(A.shape[0]))
        return r_gpu.get()
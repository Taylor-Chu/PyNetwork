import pyopencl as cl
from pyopencl.elementwise import ElementwiseKernel
import pyopencl.array as cl_array

import numpy as np

from elementwise import *
from matrix_mul import matrix_mul_local
from reduction import sum


class GPU_OPERATIONS():
    def __init__(self, A):
        self.A = A

    def __add__(self, other):
        # elementwise add 
        return addtion(self.A, other.A)

    def __sub__(self, other):
        # elementwise subtraction
        return subtraction(self.A, other.A)

    def __truediv__(self, other):
        # elementwise division
        return division(self.A, other.A)
       
    def __mul__(self, other):
        # elementwise multiplication
        return multiplication(self.A, other.A)
    
    def  __matmul__(self, other):
        # matrix multiplication
        return matrix_mul_local(self.A, other.A)

    def sum(self, axis=None):
        return sum(self.A, axis=axis)


if __name__=="__main__":
    # elementwise operators test 
    m, n= 2**8, 2**9
    A = np.random.rand(m, n).astype(np.float32)
    B = np.random.rand(m, n).astype(np.float32)
    A_class = GPU_OPERATIONS(A)
    B_class = GPU_OPERATIONS(B)
    np.testing.assert_almost_equal(A_class + B_class, A+B, decimal=3)
    np.testing.assert_almost_equal(A_class - B_class, A-B, decimal=3)
    np.testing.assert_almost_equal(A_class * B_class, A*B, decimal=3)
    np.testing.assert_almost_equal(A_class / B_class, A/B, decimal=2)

    # # matrix multiplication test 
    # m, n, p = 2**8, 2**9, 2**10
    # A = np.random.rand(m, n).astype(np.float32)
    # B = np.random.rand(n, p).astype(np.float32)
    # A_class = GPU_OPERATIONS(A)
    # B_class = GPU_OPERATIONS(B)
    # np.testing.assert_almost_equal(A_class@B_class, A@B, decimal=3)

    A = np.random.rand(3, 5).astype(np.float32)
    print(np.sum(A, axis=0))
    A_class = GPU_OPERATIONS(A)
    print(A_class.sum(axis=0))

import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array
import PyNetwork
from PyNetwork.layers.BatchNorm_GPU import BatchNorm_GPU

input_shape = (10,)
output_nodes = 100

layer = PyNetwork.layers.Dense(output_nodes, 'relu')
layer.build(input_shape)

platform = cl.get_platforms()
devices = platform[0].get_devices()
context = cl.Context(devices)
queue = cl.CommandQueue(context)

layer_gpu = PyNetwork.layers.Dense_GPU(context = context, queue = queue, hidden_nodes = output_nodes, activation_function='relu')
layer_gpu.build(input_shape)

n_datapoints = 20
z_in = np.random.rand(n_datapoints, *input_shape).astype(np.float32) - 0.5

a_true, z_out_true = layer.predict(z_in, output_only=False)

z_in_gpu = cl_array.to_device(queue, z_in)
a_true_gpu, z_out_true_gpu = layer_gpu.predict(z_in_gpu, output_only=False)


############################################
# BatchNorm Layer 

layer = PyNetwork.layers.BatchNorm()
layer.build(input_shape)


layer_gpu = BatchNorm_GPU(context = context, queue = queue)
layer_gpu.build(input_shape)

n_datapoints = 20
z_in = np.random.rand(n_datapoints, *input_shape).astype(np.float32) - 0.5

a_true, z_out_true = layer.predict(z_in, output_only=False)

z_in_gpu = cl_array.to_device(queue, z_in)
a_true_gpu, z_out_true_gpu = layer_gpu.predict(z_in_gpu, output_only=False)

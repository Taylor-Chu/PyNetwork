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

# layer_gpu = PyNetwork.layers.Dense_GPU(context = context, queue = queue, hidden_nodes = output_nodes, activation_function='relu')
# layer_gpu.build(input_shape)

# n_datapoints = 20
# z_in = np.random.rand(n_datapoints, *input_shape).astype(np.float32) - 0.5

# a_true, z_out_true = layer.predict(z_in, output_only=False)

# z_in_gpu = cl_array.to_device(queue, z_in)
# a_true_gpu, z_out_true_gpu = layer_gpu.predict(z_in_gpu, output_only=False)


# ############################################
# # BatchNorm Layer 

# layer = PyNetwork.layers.BatchNorm()
# layer.build(input_shape)


# layer_gpu = BatchNorm_GPU(context = context, queue = queue)
# layer_gpu.build(input_shape)

# n_datapoints = 20
# z_in = np.random.rand(n_datapoints, *input_shape).astype(np.float32) - 0.5

# a_true, z_out_true = layer.predict(z_in, output_only=False)

# z_in_gpu = cl_array.to_device(queue, z_in)
# a_true_gpu, z_out_true_gpu = layer_gpu.predict(z_in_gpu, output_only=False)

############################################
# Sequential Model
import numpy as np
import matplotlib.pyplot as plt

# Note: TensorFlow is not needed for PyNetwork to work. It's only used to load the dataset
import tensorflow as tf

import PyNetwork
import pyopencl as cl
import pyopencl.array as cl_array
from PyNetwork.gpu.GPUNN import GPUOPERATOR

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

shape = (28, 28)
x_train = x_train.reshape(-1, *shape) / 255
x_test = x_test.reshape(-1, *shape) / 255

labels = np.eye(10)

y_train = labels[y_train.flatten()]
y_test = labels[y_test.flatten()]

# model = PyNetwork.Sequential()

# model.add( PyNetwork.layers.Input((28, 28)) )
# model.add( PyNetwork.layers.Flatten() )
# model.add( PyNetwork.layers.Dense(100, activation_function='relu', l2=0.01, l1=0.0) )
# model.add( PyNetwork.layers.BatchNorm() )
# model.add( PyNetwork.layers.Dense(10, activation_function='relu', l2=0.0, l1=0.0) )

# optimizer = PyNetwork.optimizers.RMSprop(learning_rate=0.0001)
# model.build(loss_function='cross_entropy', optimizer=optimizer, metrics='accuracy')
# model.train(x_train, y_train, epochs=10, batch_size=128, verbose=True)

platform = cl.get_platforms()
devices = platform[0].get_devices()
context = cl.Context(devices)
queue = cl.CommandQueue(context)
gpuoperator = GPUOPERATOR(context=context, queue=queue)

# m, n, p = 2**8, 2**9, 100
# A = np.random.rand(m, n).astype(np.float32)
# B = np.random.rand(n, p).astype(np.float32)

# A_gpu = cl_array.to_device(queue, A)
# B_gpu = cl_array.to_device(queue, B)
# C = gpuoperator.matmul(A_gpu, B_gpu)
# print(C.shape)
# print(type(C))
# x_train_gpu = cl_array.to_device(queue, x_train.astype(np.float32))
# x_test_gpu = cl_array.to_device(queue, x_test.astype(np.float32))
# y_train_gpu = cl_array.to_device(queue, y_train.astype(np.float32))
# y_test_gpu = cl_array.to_device(queue, y_test.astype(np.float32))

model_gpu = PyNetwork.Sequential_GPU(context=context,queue=queue, gpuoperator=gpuoperator)

model_gpu.add( PyNetwork.layers.Input_GPU((28, 28)) )
model_gpu.add( PyNetwork.layers.Flatten_GPU(context=context, queue=queue, gpuoperator=gpuoperator) )
model_gpu.add( PyNetwork.layers.Dense_GPU(hidden_nodes=100, activation_function='relu', l2=0.01, l1=0.0,context=context, queue=queue, gpuoperator=gpuoperator) )
model_gpu.add( PyNetwork.layers.BatchNorm_GPU(context=context, queue=queue, gpuoperator=gpuoperator) )
model_gpu.add( PyNetwork.layers.Dense_GPU(hidden_nodes=10, activation_function='relu', l2=0.0, l1=0.0, context=context, queue=queue, gpuoperator=gpuoperator) )

optimizer = PyNetwork.optimizers.RMSprop_GPU(context=context,queue=queue, gpuoperator=gpuoperator,learning_rate=0.0005)
model_gpu.build(loss_function='cross_entropy', optimizer=optimizer, metrics='accuracy')

model_gpu.train(x_train, y_train, epochs=10, batch_size=128, verbose=True)

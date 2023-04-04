import numpy as np
import pyopencl.array as cl_array

def get_activation_function_gpu(name, gpuoperator, **kwargs):
    """ Returns the function of the given name

        Parameters
        ----------
        name : str
            The name of the desired function

        Raises
        ------
        Exception
            If `name` has not been implemented
    """

    if name == 'relu':
        def relu(x, grad=False):
            if grad:
                return (x > 0).astype(np.float32)

            x = gpuoperator.relu(x)
            return x
        # END def relu
        return relu
    
    elif name == 'softmax':
        def softmax(x, grad=False):
            if grad:
                softmax_val = gpuoperator.softmax(x)
                return softmax_val*(1 - softmax_val)

            return gpuoperator.softmax(x)
        # END def softmax
        return softmax
    
    elif name == 'linear':
        def linear(x, grad=False):
            if grad:
                return 1
            return x
        return linear
    else:
        raise Exception(f'{name} is not a defined function.')


def get_error_function_gpu(name, operator):
    """ Returns the function of the given name

        Parameters
        ----------
        name : str
            The name of the desired function

        Raises
        ------
        Exception
            If `name` has not been implemented
    """
    if name == 'mse':
        def mse(predictions, targets, grad = False):
            if grad:
                return np.float32(2) * (predictions - targets)
            N = predictions.shape[0]
            return cl_array.sum(operator.pow(predictions-targets, 2)) / np.float32(2*N)
            #return np.sum(((predictions - targets)**2)/2)/N
        return mse
    elif name == 'cross_entropy':
        def cross_entropy(predictions, targets, epsilon=1e-12, grad=False):
            """ Computes cross entropy between targets (encoded as one-hot vectors) and predictions.

                Parameters
                ----------
                    predictions : (N, k) np.array
                    targets     : (N, k) np.array

                Returns
                -------
                    float
                        If grad = False then the cross_entropy score is retuned

                    OR

                    (N, k) np.array
                        If grad = True then the gradient of the output is returned
            """
            predictions = operator.clip(predictions, epsilon, 1. - epsilon)

            if grad:
                return (predictions - targets) / (predictions*(1-predictions))
                #return  operator.div((1 - targets),(1 - predictions))-operator.div(targets,predictions)

            N = predictions.shape[0]
            ce = -operator.sum(operator.mul(targets, operator.log(predictions+1e-9))).get()/N
            return ce
        return cross_entropy
    else:
        raise Exception(f'{name} is not a defined function.')


def get_metric_function_gpu(name):
    """ Returns the metric fucntion of a given name

        Parameters
        ----------
        name : str
            The name of the desired function

        Raises
        ------
        Exception
            If `name` has not been implemented
    """
    if name == 'accuracy':
        def accuracy(predictions, target):
            return np.mean(np.argmax(predictions.get(), axis=-1) == np.argmax(target.get(), axis=-1))
        return accuracy
    else:
        raise Exception(f'{name} is not a defined metric.')

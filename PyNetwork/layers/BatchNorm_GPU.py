import numpy as np
import pyopencl as cl
from pyopencl.elementwise import ElementwiseKernel
import pyopencl.array as cl_array

from PyNetwork.gpu.GPUNN import GPUOPERATOR
from PyNetwork.validation import check_layer
from PyNetwork.layers import Layer
from PyNetwork import get_activation_function


class BatchNormGrads:
    """Returns the gradients (partial derivatives) of this layer.
    Note that d is meant to be partial

    1. dgamma = dS/d(gamma)
    2. dbeta = dS/d(beta)
    3. dz = dS/d(z^{k-1})
        This is the gradient with respect to the input of the batch-norm layer
    4. dz_hat = dS/d(z_hat^{k-1})
        The gradient with respect to the normalised input of the batch-norm layer
    5. dsigma2 = dS/d(sigma^2)
    6. dmu = dS/d(mu)
    """

    @staticmethod
    def dgamma(gpuoperator, new_delta, z_hat):
        """Returns ds/d(gamma)

        Parameters
        ----------
        new_delta : (N, ...) np.array
            Should be delta^{k}
        z_hat : (N, ...) np.array
            Should be the normalised input of this layer

        Returns
        -------
        (N, ...) np.array
        """
        # This can probably be speed up by using einsum, so that an intermediate np.array
        # isn't created
        return gpuoperator.sum(gpuoperator.mul(new_delta, z_hat), axis=0)

    @staticmethod
    def dbeta(gpuoperator, new_delta):
        """Returns ds/d(beta)

        Parameters
        ----------
        new_delta : (N, ...) np.array
            Should be delta^{k}

        Returns
        -------
        (N, ...) np.array
        """

        return gpuoperator.sum(new_delta, axis=0)

    @staticmethod
    def dz_hat(gpuoperator, new_delta, gamma):
        """Returns dS/d(z_hat) - The gradient with respect to the normalised input of
        the batch-norm layer

        Parameters
        ----------
        new_delta : (N, ...) np.array
            Should be delta^{k}
        gamma : (...) np.array

        Return
        ------
        (N, ...) np.array
        """
        return gpuoperator.mul(new_delta, gamma)

    @staticmethod
    def dsigma2(gpuoperator, z, dz_hat_, epsilon, mu=None, sigma=None):
        """Returns dS/d(sigma^2)

        Parameters
        ----------
        z : (N, ...) np.array
            The input of this layer: z^{k-1}
        dz_hat_ : (N, ...) np.array
            The gradient with respect to the normalised input: dS/d(z_hat^{k-1})
        epsilon : float

        mu : (...) np.array, optional
            The mean of the input. If None, then it will be computed
        sigma : (...) np.array, optional
            The std of the input. If None, then it will be computed

        Returns
        -------
        (...) np.array
        """
        if mu is None:
            mu = gpuoperator.mean(z, axis=0)
        if sigma is None:
            sigma = gpuoperator.std(z, axis=0)

        # c = (-0.5 * (sigma ** 2 + epsilon) ** (-3 / 2))
        half = -0.5
        c = gpuoperator.pow(half * (gpuoperator.pow(sigma, 2) + epsilon), -1.5)
        # return c * np.sum(dz_hat_ * (z - mu), axis=0)
        return gpuoperator.mul(
            c, gpuoperator.sum(gpuoperator.mul(gpuoperator.sub(z, mu), dz_hat_), axis=0)
        )

    @staticmethod
    def dmu(gpuoperator, z, dz_hat_, epsilon, mu=None, sigma=None, dsigma2_=None):
        """Returns dS/dmu

        Parameters
        ----------
        z : (N, ...) np.array
            The input of this layer: z^{k-1}
        dz_hat_ : (N, ...) np.array
            The gradient with respect to the normalised input: dS/d(z_hat^{k-1})
        epsilon : float

        mu : (...) np.array, optional
            The mean of the input. If None, then it will be computed
        sigma : (...) np.array, optional
            The std of the input. If None, then it will be computed
        dsigma2_ : (...) np.array, optional
            This should be the gradient ds/d(sigma^2). If it is set to None then it
            will be computed when this function is called
        """

        if mu is None:
            mu = gpuoperator.mean(z, axis=0)
        if sigma is None:
            sigma = gpuoperator.std(z, axis=0)
        if dsigma2_ is None:
            dsigma2_ = BatchNormGrads.dsigma2(
                gpuoperator, z, dz_hat_, epsilon, mu, sigma
            )

        m_one = -1
        m_two = -2
        # return (-1 / np.sqrt(sigma**2 + epsilon))*np.sum(dz_hat_, axis=0) + dsigma2_ * np.mean(-2*(z - mu), axis=0)
        return gpuoperator.add(
            gpuoperator.mul(
                m_one / gpuoperator.pow((gpuoperator.pow(sigma, 2) + epsilon), 0.5),
                gpuoperator.sum(dz_hat_, axis=0),
            ),
            gpuoperator.mul(
                dsigma2_, gpuoperator.mean(m_two * gpuoperator.sub(z, mu)), axis=0
            ),
        )

    @staticmethod
    def dz(gpuoperator, z, new_delta, gamma, epsilon, mu=None, sigma=None):
        """Returns the partial derivative with respect to the input: dS/dZ^{n-1}

        Parameters
        ----------
        gpuoperator : class
            The gpuoperator class inherited by the BatchNorm layer class
        z : (N, ...) np.array
            The input of this layer: z^{n-1}
        new_delta : (N, ...) np.array
            The back-prop gradient: delta^{n}
        gamma : (...) np.array
        epsilon : float
            Arbitrarily small float to prevent division by 0 error

        mu : (...) np.array, optional
            The mean of the input. If None, then it will be computed
        sigma : (...) np.array, optional
            The std of the input. If None, then it will be computed

        Returns
        -------
        (N, ...) np.array
        """

        if mu is None:
            mu = gpuoperator.mean(z, axis=0)
        if sigma is None:
            sigma = gpuoperator.std(z, axis=0)
        m = len(z)
        two = 2
        dz_hat_ = BatchNormGrads.dz_hat(gpuoperator, new_delta, gamma)
        dsigma2_ = BatchNormGrads.dsigma2(gpuoperator, z, dz_hat_, epsilon, mu, sigma)
        dmu_ = BatchNormGrads.dmu(gpuoperator, z, dz_hat_, epsilon, mu, sigma, dsigma2_)

        # return dz_hat_ / np.sqrt(sigma**2 + epsilon) + dsigma2_*2*(z - mu)/m + dmu_/m
        return gpuoperator.add(
            gpuoperator.div(
                dz_hat_,
                gpuoperator.pow(gpuoperator.pow(sigma, 2) + epsilon, 0.5),
            ),
            gpuoperator.add(
                gpuoperator.mul(
                    dsigma2_ * two,
                    gpuoperator.sub(z, mu) / m,
                ),
                dmu_ / m,
            ),
        )


class BatchNorm_GPU(Layer):
    """A BatchNorm layer where the inputs are normalised and then linearly scaled.
    Concretely, given an input z, this layer will return
        gamma * z_hat + beta
    where z_hat is the normliased version of z, and gamma and beta are matrices

    Attributes
    ----------
    activation_function : str
        The name of the activation function

    input_shape : k tuple
        The shape of the input of this layer
    output_shape : k tuple
        The shape of the output of this layer
    gamma : np.array (of dimension k)
        The scaling factor of the elements
    beta : np.array (of dimension k)
        The bias units for the elements

    built : bool
        Has the model been initialised

    Notes
    -----
    This implementation of batch-norm assumes that batch-norm is applied AFTER the activation function. For
    example:
        Correct Use: Fully Connected -> Activation Function -> Batch-Norm (Batch-norm after activation)
        Incorrect Use: Fully Connected -> Batch-Norm -> Activation Function (Batch-norm before activation)
    """

    def __init__(
        self,
        context,
        queue,
    ):
        """Initise Class"""
        self.built = False
        self.epsilon_gpu = 1e-10

        self.activation_function = "linear"

        self.input_shape = None
        self.output_shape = None

        self.gamma = None
        self.beta = None

        self.context = context
        self.queue = queue
        self.gpuoperator = GPUOPERATOR(context, queue)

    def build(self, previous_output_shape):
        """Initialise Attributes `gamma` and `beta`

        Parameters
        ----------
        previous_output_shape : k tuple
            The shape of the input of this layer
        """
        self.input_shape = previous_output_shape
        self.output_shape = previous_output_shape

        self.gamma = np.ones(self.input_shape)
        self.beta = np.zeros(self.input_shape)

        self.gamma_gpu = cl_array.to_device(self.queue, self.gamma)
        self.beta_gpu = cl_array.to_device(self.queue, self.beta)

        self.built = True

    def predict(self, z, output_only=True, **kwargs):
        """Returns the output of this layer

        Parameters
        ----------
        z : (N, ...) np.array
            z is assumed to be a list of all the inputs to be forward propagated. In particular
            it is assumed that the first index of z is the index that inputs is accessed by
        output_only : bool, optional
            If set to true, then this function will return only the prediction of the neural
            network. If set to false then this will return the outputs of the individual
            layers. Unless back propagation is being performed, this should be set to true.

        Returns
        -------
        (N, ...) np.array
            The final output of the layer, post activation

        OR (if `output_only = False`)

        (N, ...) np.array, (N, ...) np.array
            The first np.array will store the output before it is passed through the activation
            function.
            The second np.array will store the output after it has passed through the
            activation function.

        Notes
        -----
        Since the activation function is linear the 2 arrays, when output_only = True, are the same
        array
        """
        check_layer(self)
        mean = self.gpuoperator.mean(z, axis=0)
        # mean = np.mean(z, axis=0)
        std = self.gpuoperator.std(z, axis=0)
        # std = np.std(z, axis=0)

        b = self.gpuoperator.div(
            self.gpuoperator.sub(z, mean),
            self.gpuoperator.pow(
                (self.gpuoperator.pow(std, 2) + self.epsilon_gpu), 0.5
            ),
        )
        a = self.gpuoperator.add(self.gpuoperator.mul(self.gamma_gpu, b), self.beta_gpu)
        # a = self.gamma * ((z - mean) / np.sqrt(std ** 2 + self.epsilon)) + self.beta

        if output_only:
            return a
        return a, a

    def get_delta_backprop_(self, g_prime, new_delta, prev_z):
        """Returns the delta for the previous layer, delta^{k-1}_{m,j}.

        Parameters
        ----------
        g_prime : (N, ...) np.array
            Should be the derivative of the ouput of the previous layer, g'_{k-1}(a^{k-1}_{m,j})
        new_delta : (N, ...) np.array
            The delta for this layer, delta^k_{m, j}
        prev_z : (N, ...) np.array
            The input for this layer, z^{k-1}

        Returns
        -------
        (N, ...) np.array
            Returns delta of the previous layer, delta^{k-1}

        Notes
        -----
        We want to return delta^{k-1} because the `sequential` class does not have access to the
        weights, W. But it does know the values of g'_{k-1} and delta^k, due to forward propagation
        and the backwards nature of the back propagation algorithm.
        """
        check_layer(self)

        # dz_ = BatchNormGrads.dz(prev_z, new_delta, self.gamma, self.epsilon)
        dz_ = BatchNormGrads.dz(prev_z, new_delta, self.gamma_gpu, self.epsilon_gpu)
        # return dz_ * prev_z
        return self.gpuoperator.mul(dz_, prev_z)

    def get_weight_grad_(self, delta, prev_z):
        """Returns the gradients with respect to beta and gamma

        Parameters
        ----------
        delta : (N, ...) np.array
            Should be delta^k
        prev_z : (N, ...) np.array
            The input of this layer: z^{k-1}

        Returns
        -------
        (...) np.array, (...) np.array
            The first np.array is dS/d(beta)
            The second np.array is dS/d(gamma)
        """
        check_layer(self)

        # z_hat = (prev_z - np.mean(prev_z, axis=0)) / np.sqrt(np.std(prev_z, axis=0) ** 2 + self.epsilon)
        z_hat = self.gpuoperator.div(
            self.gpuoperator.sub(prev_z, self.gpuoperator.mean(prev_z, axis=0)),
            self.gpuoperator.sqrt(
                self.gpuoperator.add(
                    self.gpuoperator.pow(self.gpuoperator.std(prev_z, axis=0), 2),
                    self.epsilon_gpu,
                )
            ),
        )
        r_a = self.gpuoperator.sum(delta, axis=0)
        r_b = self.gpuoperator.sum(self.gpuoperator.mul(delta, z_hat), axis=0)
        # return np.sum(delta, axis=0), np.sum(delta * z_hat, axis=0)
        return r_a, r_b

    def update_parameters_(self, beta_updates, gamma_updates):
        """Perform an update to the weights by descending down the gradient

        Parameters
        ----------
        beta_updates : np.array (of dimension k)
            Should be dS/d(beta), as scheduled by the optimizer
        gamma_updates : np.array (of dimension k)
            Should be dS/d(gamma), as scheduled by the optimizer
        """
        check_layer(self)

        # self.beta -= beta_updates
        # self.gamma -= gamma_updates
        self.beta_gpu = self.gpuoperator.sub(self.beta_gpu, beta_updates)
        self.gamma_gpu = self.gpuoperator.sub(self.gamma_gpu, gamma_updates)

    def get_weights(self):
        check_layer(self)
        return self.beta_gpu, self.gamma_gpu

    def summary_(self):
        check_layer(self)
        return f"Batch Norm", f"Output Shape {(None, *self.output_shape)}"

    @property
    def activation_function_(self):
        return get_activation_function(self.activation_function)

    def __str__(self):
        return f"Batch Norm; built = {self.built}"

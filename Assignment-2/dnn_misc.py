"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only classes/functions you need to implement in this template is linear_layer, relu, and dropout
"""

import numpy as np
import dnn_im2col


### Modules ###

    ########################################################################################
    #   The following three modules (class) are what you need to complete  (check TODO)    #
    ########################################################################################

class linear_layer:

    """
        The linear (affine/fully-connected) module.

        It is built up with two arguments:
        - input_D: the dimensionality of the input example/instance of the forward pass
        - output_D: the dimensionality of the output example/instance of the forward pass

        It has two learnable parameters:
        - self.params['W']: the W matrix (numpy array) of shape input_D-by-output_D
        - self.params['b']: the b vector (numpy array) of shape 1-by-output_D

        It will record the partial derivatives of loss w.r.t. self.params['W'] and self.params['b'] in:
        - self.gradient['W']: input_D-by-output_D numpy array
        - self.gradient['b']: 1-by-output_D numpy array
    """

    def __init__(self, input_D, output_D):
        self.params = dict()
        self.params['W'] = np.random.normal(0, 0.1, (input_D, output_D))
        self.params['b'] = np.random.normal(0, 0.1, (1, output_D))

        self.gradient = dict()
        self.gradient['W'] = np.zeros((input_D, output_D))
        self.gradient['b'] = np.zeros((1, output_D))

    def forward(self, X):

        """
            The forward pass of the linear (affine/fully-connected) module.

            Input:
            - X: A N-by-input_D numpy array, where each 'row' is an input example/instance (i.e., X[i], where i = 1,...,N).
                The mini-batch size is N.

            Operation:
            - You are going to generate a N-by-output_D numpy array named forward_output.
            - For each row x of X (say X[i]), perform X[i] self.params['W'] + self.params['b'], and store the output in forward_output[i].
            - Please use np.XX to call a numpy function XX.
            - You are encouraged to use matrix/element-wise operations to avoid using FOR loop.

            Return:
            - forward_output: A N-by-output_D numpy array, where each 'row' is an output example/instance.
        """

        ################################################################################
        # TODO: Implement the linear forward pass. Store the result in forward_output  #
        ################################################################################
        forward_output = X.dot(self.params['W']) + self.params['b']
        return forward_output

    def backward(self, X, grad):

        """
            The backward pass of the linear (affine/fully-connected) module.

            Input:
            - X: A N-by-input_D numpy array, the input to the forward pass.
            - grad: A N-by-output_D numpy array, where each 'row' (say row i) is the partial derivatives of the mini-batch loss
                 w.r.t. forward_output[i].

            Operation:
            - Compute the partial derivatives (gradients) of the mini-batch loss w.r.t. self.params['W'], self.params['b'], and X.
            - You are going to generate a N-by-input_D numpy array named backward_output.
            - Store the partial derivatives (gradients) of the mini-batch loss w.r.t. X in backward_output.
            - Store the partial derivatives (gradients) of the mini-batch loss w.r.t. self.params['W'] in self.gradient['W'].
            - Store the partial derivatives (gradients) of the mini-batch loss w.r.t. self.params['b'] in self.gradient['b'].
            - You are encouraged to use matrix/element-wise operations to avoid using FOR loop.

            Return:
            - backward_output: A N-by-input_D numpy array, where each 'row' (say row i) is the partial derivatives of the mini-batch loss
                 w.r.t. X[i].
        """

        ##########################################################################################################################
        # TODO: Implement the backward pass (i.e., compute the following three terms)                                            #
        # self.gradient['W'] = ? (input_D-by-output_D numpy array, the gradient of the mini-batch loss w.r.t. self.params['W'])  #
        # self.gradient['b'] = ? (1-by-output_D numpy array, the gradient of the mini-batch loss w.r.t. self.params['b'])        #
        # backward_output = ? (N-by-input_D numpy array, the gradient of the mini-batch loss w.r.t. X)                           #
        # only return backward_output, but need to compute self.gradient['W'] and self.gradient['b']                             #
        ##########################################################################################################################
        self.gradient['W'] = X.T.dot(grad)
        self.gradient['b'] = np.sum(grad, axis=0)
        backward_output = grad.dot(self.params['W'].T)
        return backward_output


class relu:

    """
        The relu (rectified linear unit) module.

        It is built up with NO arguments.
        It has no parameters to learn.
        self.mask is an attribute of relu. You can use it to store things (computed in the forward pass) for the use in the backward pass.
    """

    def __init__(self):
        self.mask = None

    def forward(self, X):

        """
            The forward pass of the relu (rectified linear unit) module.

            Input:
            - X: A numpy array of arbitrary shape.

            Operation:
            - You are to generate a numpy array named forward_output of the same shape of X.
            - For each element x of X, perform max{0, x}, and store it in the corresponding element of forward_output.
            - Please use np.XX to call a numpy function XX if necessary.
            - You are encouraged to use matrix/element-wise operations to avoid using FOR loop.
            - You can use self.mask to store what you may need (except X) for the use in the backward pass.

            Return:
            - forward_output: A numpy array of the same shape of X
        """

        ################################################################################
        # TODO: Implement the relu forward pass. Store the result in forward_output    #
        ################################################################################
        self.mask = (X > 0).astype(int)
        forward_output = np.maximum(X, 0)
        return forward_output

    def backward(self, X, grad):

        """
            The backward pass of the relu (rectified linear unit) module.

            Input:
            - X: A numpy array of arbitrary shape, the input to the forward pass.
            - grad: A numpy array of the same shape of X, where each element is the partial derivative of the mini-batch loss
                 w.r.t. the corresponding element in forward_output.

            Operation:
            - You are to generate a numpy array named backward_output of the same shape of X.
            - Compute the partial derivatives (gradients) of the mini-batch loss w.r.t. X, and store it in backward_output.
            - You are encouraged to use matrix/element-wise operations to avoid using FOR loop.
            - You can use self.mask.
            - PLEASE follow the Heaviside step function defined in CSCI567_HW2.pdf

            Return:
            - backward_output: A numpy array of the same shape as X, where each element is the partial derivative of the mini-batch loss
                 w.r.t. the corresponding element in  X.
        """

        ##########################################################################################################################
        # TODO: Implement the backward pass (i.e., compute the following term)                                                   #
        # backward_output = ? (A numpy array of the shape of X, the gradient of the mini-batch loss w.r.t. X)                    #
        # PLEASE follow the Heaviside step function defined in CSCI567_HW2.pdf                                                   #
        ##########################################################################################################################
        backward_output = grad * self.mask
        return backward_output


class dropout:

    """
        The dropout module.

        It is built up with one arguments:
        - r: the dropout rate

        It has no parameters to learn.
        self.mask is an attribute of dropout. You can use it to store things (computed in the forward pass) for the use in the backward pass.
    """

    def __init__(self, r):
        self.r = r
        self.mask = None

    def forward(self, X, is_train):

        """
            The forward pass of the dropout module.

            Input:
            - X: A numpy array of arbitrary shape.
            - is_train: A boolean value. If False, no dropout is performed.

            Operation:
            - Sample uniformly a value p in [0.0, 1.0) for each element of X
            - If p >= self.r, output that element multiplied by (1.0 / (1 - self.r)); otherwise, output 0 for that element
            - Please use np.XX to call a numpy function XX if necessary.
            - You are encouraged to use matrix/element-wise operations to avoid using FOR loop.
            - You can use self.mask to store what you may need (except X) for the use in the backward pass.

            Return:
            - forward_output: A numpy array of the same shape of X (the output of dropout)
        """

        ################################################################################
        #  TODO: We provide the forward pass to you. You only need to understand it.   #
        ################################################################################

        if is_train:
            self.mask = (np.random.uniform(0.0, 1.0, X.shape) >= self.r).astype(float) * (1.0 / (1.0 - self.r))
        else:
            self.mask = np.ones(X.shape)
        forward_output = X * self.mask
        return forward_output

    def backward(self, X, grad):

        """
            The backward pass of the dropout module.

            Input:
            - X: A numpy array of arbitrary shape, the input to the forward pass.
            - grad: A numpy array of the same shape of X, where each element is the partial derivative of the mini-batch loss
                 w.r.t. the corresponding element in forward_output.

            Operation:
            - You are to generate a numpy array named backward_output of the same shape of X.
            - Compute the partial derivatives (gradients) of the mini-batch loss w.r.t. X, and store it in backward_output.
            - You are encouraged to use matrix/element-wise operations to avoid using FOR loop.
            - You should use self.mask. You should NOT re-sample p.
            - Check CSCI567_HW2.pdf for the backward pass of dropout.

            Return:
            - backward_output: A numpy array of the same shape as X, where each element is the partial derivative of the mini-batch loss
                 w.r.t. the corresponding element in X.
        """

        ##########################################################################################################################
        # TODO: Implement the backward pass (i.e., compute the following term)                                                   #
        # backward_output = ? (A numpy array of the shape of X, the gradient of the mini-batch loss w.r.t. X)                    #
        # PLEASE follow the formula shown in the homework pdf                                                                    #
        ##########################################################################################################################
        backward_output = grad * self.mask
        return backward_output



    ###########################################################################
    #          Please DO NOT change the following parts of the script         #
    ###########################################################################


class conv_layer:

    def __init__(self, num_input, num_output, filter_len, stride):
        self.params = dict()
        self.params['W'] = np.random.normal(0, 0.1, (num_output, num_input, filter_len, filter_len))
        self.params['b'] = np.random.normal(0, 0.1, (num_output, 1))

        self.gradient = dict()
        self.gradient['W'] = np.zeros((num_output, num_input, filter_len, filter_len))
        self.gradient['b'] = np.zeros((num_output, 1))

        self.stride = stride
        self.padding = int((filter_len - 1) / 2)
        self.X_col = None

    def forward(self, X):
        n_filters, d_filter, h_filter, w_filter = self.params['W'].shape
        n_x, d_x, h_x, w_x = X.shape
        h_out = int((h_x - h_filter + 2 * self.padding) / self.stride + 1)
        w_out = int((w_x - w_filter + 2 * self.padding) / self.stride + 1)

        self.X_col = dnn_im2col.im2col_indices(X, h_filter, w_filter, self.padding, self.stride)
        W_col = self.params['W'].reshape(n_filters, -1)

        out = np.matmul(W_col, self.X_col) + self.params['b']
        out = out.reshape(n_filters, h_out, w_out, n_x)
        out_forward = out.transpose(3, 0, 1, 2)

        return out_forward

    def backward(self, X, grad):
        n_filters, d_filter, h_filter, w_filter = self.params['W'].shape

        self.gradient['b'] = np.sum(grad, axis=(0, 2, 3)).reshape(n_filters, -1)

        grad_reshaped = grad.transpose(1, 2, 3, 0).reshape(n_filters, -1)
        self.gradient['W'] = np.matmul(grad_reshaped, self.X_col.T).reshape(self.params['W'].shape)

        W_reshape = self.params['W'].reshape(n_filters, -1)
        out = np.matmul(W_reshape.T, grad_reshaped)
        out_backward = dnn_im2col.col2im_indices(out, X.shape, h_filter, w_filter, self.padding, self.stride)

        return out_backward


class max_pool:

    def __init__(self, max_len, stride):
        self.max_len = max_len
        self.stride = stride
        self.padding = 0 # int((max_len - 1) / 2)
        self.argmax_cols = None

    def forward(self, X):
        n_x, d_x, h_x, w_x = X.shape
        h_out = int((h_x - self.max_len + 2 * self.padding) / self.stride + 1)
        w_out = int((w_x - self.max_len + 2 * self.padding) / self.stride + 1)

        max_cols, self.argmax_cols = dnn_im2col.maxpool_im2col_indices(X, self.max_len, self.max_len, self.padding, self.stride)
        out_forward = max_cols.reshape(n_x, d_x, h_out, w_out)

        return out_forward

    def backward(self, X, grad):
        out_backward = dnn_im2col.maxpool_col2im_indices(grad, self.argmax_cols, X.shape, self.max_len, self.max_len, self.padding, self.stride)

        return out_backward


class flatten_layer:

    def __init__(self):
        self.size = None

    def forward(self, X):
        self.size = X.shape
        out_forward = X.reshape(X.shape[0], -1)

        return out_forward

    def backward(self, X, grad):
        out_backward = grad.reshape(self.size)

        return out_backward


### Loss functions ###

class softmax_cross_entropy:
    def __init__(self):
        self.expand_Y = None
        self.calib_logit = None
        self.sum_exp_calib_logit = None
        self.prob = None

    def forward(self, X, Y):
        self.expand_Y = np.zeros(X.shape).reshape(-1)
        self.expand_Y[Y.astype(int).reshape(-1) + np.arange(X.shape[0]) * X.shape[1]] = 1.0
        self.expand_Y = self.expand_Y.reshape(X.shape)

        self.calib_logit = X - np.amax(X, axis = 1, keepdims = True)
        self.sum_exp_calib_logit = np.sum(np.exp(self.calib_logit), axis = 1, keepdims = True)
        self.prob = np.exp(self.calib_logit) / self.sum_exp_calib_logit

        forward_output = - np.sum(np.multiply(self.expand_Y, self.calib_logit - np.log(self.sum_exp_calib_logit))) / X.shape[0]
        return forward_output

    def backward(self, X, Y):
        backward_output = - (self.expand_Y - self.prob) / X.shape[0]
        return backward_output


class sigmoid_cross_entropy:
    def __init__(self):
        self.expand_Y = None
        self.calib_logit = None
        self.sum_exp_calib_logit = None
        self.prob = None

    def forward(self, X, Y):
        self.expand_Y = np.concatenate((Y, 1 - Y), axis = 1)

        X_cat = np.concatenate((X, np.zeros((X.shape[0], 1))), axis = 1)
        self.calib_logit = X_cat - np.amax(X_cat, axis=1, keepdims=True)
        self.sum_exp_calib_logit = np.sum(np.exp(self.calib_logit), axis=1, keepdims=True)
        self.prob = np.exp(self.calib_logit[:, 0].reshape(X.shape[0], -1)) / self.sum_exp_calib_logit

        forward_output = - np.sum(np.multiply(self.expand_Y, self.calib_logit - np.log(self.sum_exp_calib_logit))) / X.shape[0]
        return forward_output

    def backward(self, X, Y):
        backward_output = - (self.expand_Y[:, 0].reshape(X.shape[0], -1) - self.prob) / X.shape[0]
        return backward_output


### Momentum ###

def add_momentum(model):
    momentum = dict()
    for module_name, module in model.items():
        if hasattr(module, 'params'):
            for key, _ in module.params.items():
                momentum[module_name + '_' + key] = np.zeros(module.gradient[key].shape)
    return momentum
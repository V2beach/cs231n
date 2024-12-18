from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        for layer in range(self.num_layers):
          if layer == 0:
            self.params[f"W{layer+1}"] = np.random.normal(loc=0.0, scale=weight_scale, size=(input_dim, hidden_dims[0]))
            # self.params[f"W{layer+1}"] = np.random.randn(input_dim, hidden_dims[0]) * weight_scale
            self.params[f"b{layer+1}"] = np.zeros(hidden_dims[0])
            if self.normalization == "batchnorm" or self.normalization == "layernorm":
              self.params[f"gamma{layer+1}"] = np.ones(hidden_dims[0])
              self.params[f"beta{layer+1}"] = np.zeros(hidden_dims[0])
          elif layer == len(hidden_dims):
            self.params[f"W{layer+1}"] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dims[layer-1], num_classes))
            # self.params[f"W{layer+1}"] = np.random.randn(hidden_dims[layer-1], num_classes) * weight_scale
            self.params[f"b{layer+1}"] = np.zeros(num_classes)
          else:
            self.params[f"W{layer+1}"] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dims[layer-1], hidden_dims[layer]))
            # self.params[f"W{layer+1}"] = np.random.randn(hidden_dims[layer-1], hidden_dims[layer]) * weight_scale
            self.params[f"b{layer+1}"] = np.zeros(hidden_dims[layer])
            if self.normalization == "batchnorm" or self.normalization == "layernorm":
              self.params[f"gamma{layer+1}"] = np.ones(hidden_dims[layer])
              self.params[f"beta{layer+1}"] = np.zeros(hidden_dims[layer])
        
        # for k, v in self.params.items():
        #   print(k)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode

        # for i in range(len(self.bn_params)): # 全是mode: train，沒存參數
        #     print(self.bn_params[i])
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax
        caches = []
        out = X
        for layer in range(self.num_layers - 1): # 2 hidden layers = 3 layers (3 affine layers)
          if self.normalization == "batchnorm":
            out, cache = affine_batchnorm_relu_forward(out, self.params[f'W{layer+1}'], self.params[f'b{layer+1}'], self.params[f"gamma{layer+1}"], self.params[f"beta{layer+1}"], self.bn_params[layer])
            caches.append(cache)
            # out, cache = relu_forward(out)
            # caches.append(cache)
          elif self.normalization == "layernorm":
            out, cache = affine_layernorm_relu_forward(out, self.params[f'W{layer+1}'], self.params[f'b{layer+1}'], self.params[f"gamma{layer+1}"], self.params[f"beta{layer+1}"], self.bn_params[layer])
            caches.append(cache)
          else:
            out, cache = affine_relu_forward(out, self.params[f'W{layer+1}'], self.params[f'b{layer+1}'])
            caches.append(cache)
          if self.use_dropout:
            out, cache = dropout_forward(out, self.dropout_param)
            caches.append(cache)
        scores, cache = affine_forward(out, self.params[f'W{self.num_layers}'], self.params[f'b{self.num_layers}'])
        caches.append(cache)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dx = softmax_loss(scores, y)
        for layer in range(self.num_layers):
          loss += 0.5 * self.reg * (np.sum(self.params[f'W{layer+1}'] ** 2))
  
        dx, dw, db = affine_backward(dx, caches.pop())
        grads[f'W{self.num_layers}'] = dw + self.reg * self.params[f'W{self.num_layers}']
        grads[f'b{self.num_layers}'] = db

        for layer in reversed(range(self.num_layers - 1)):
          if self.use_dropout:
            dx = dropout_backward(dx, caches.pop())
          if self.normalization == "batchnorm":
            dx, dw, db, dgamma, dbeta = affine_batchnorm_relu_backward(dx, caches.pop())
            grads[f"gamma{layer+1}"] = dgamma
            grads[f"beta{layer+1}"] = dbeta
          elif self.normalization == "layernorm":
            dx, dw, db, dgamma, dbeta = affine_layernorm_relu_backward(dx, caches.pop())
            grads[f"gamma{layer+1}"] = dgamma
            grads[f"beta{layer+1}"] = dbeta
          else:
            dx, dw, db = affine_relu_backward(dx, caches.pop())
          grads[f'W{layer+1}'] = dw + self.reg * self.params[f'W{layer+1}']
          grads[f'b{layer+1}'] = db

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

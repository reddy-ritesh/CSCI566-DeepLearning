from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


""" Super Class """


class Optimizer(object):
    """
    This is a template for implementing the classes of optimizers
    """

    def __init__(self, net, lr=1e-4):
        self.net = net  # the model
        self.lr = lr  # learning rate

    """ Make a step and update all parameters """

    def step(self):
        #### FOR RNN / LSTM ####
        if hasattr(self.net, "preprocess") and self.net.preprocess is not None:
            self.update(self.net.preprocess)
        if hasattr(self.net, "rnn") and self.net.rnn is not None:
            self.update(self.net.rnn)
        if hasattr(self.net, "postprocess") and self.net.postprocess is not None:
            self.update(self.net.postprocess)

        #### MLP ####
        if (
            not hasattr(self.net, "preprocess")
            and not hasattr(self.net, "rnn")
            and not hasattr(self.net, "postprocess")
        ):
            for layer in self.net.layers:
                self.update(layer)


""" Classes """


class SGD(Optimizer):
    """Some comments"""

    def __init__(self, net, lr=1e-4):
        self.net = net
        self.lr = lr

    def update(self, layer):
        for n, dv in layer.grads.items():
            layer.params[n] -= self.lr * dv


class SGDM(Optimizer):
    """Some comments"""

    def __init__(self, net, lr=1e-4, momentum=0.0):
        self.net = net
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}  # last update of the velocity

    def update(self, layer):
        #############################################################################
        # TODO: Implement the SGD + Momentum                                        #
        #############################################################################
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################


class RMSProp(Optimizer):
    """Some comments"""

    def __init__(self, net, lr=1e-2, decay=0.99, eps=1e-8):
        self.net = net
        self.lr = lr
        self.decay = decay
        self.eps = eps
        self.cache = {}  # decaying average of past squared gradients

    def update(self, layer):
        #############################################################################
        # TODO: Implement the RMSProp                                               #
        #############################################################################
        for param_name, param in layer.params.items():
            if param_name not in self.cache:
                self.cache[param_name] = np.zeros_like(param)

            # Compute the decaying average of past squared gradients
            self.cache[param_name] = self.decay * self.cache[param_name] + (1 - self.decay) * layer.grads[param_name] ** 2

            # Update the parameter using RMSProp update rule
            layer.params[param_name] -= (self.lr * layer.grads[param_name]) / (np.sqrt(self.cache[param_name] + self.eps))
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################


class Adam(Optimizer):
    """Some comments"""

    def __init__(self, net, lr=1e-3, beta1=0.9, beta2=0.999, t=0, eps=1e-8):
        self.net = net
        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.eps = eps
        self.mt = {}
        self.vt = {}
        self.t = t

    def update(self, layer):
        #############################################################################
        # TODO: Implement the Adam                                                  #
        #############################################################################
        for param_name, param in layer.params.items():
            if param_name not in self.mt:
                self.mt[param_name] = np.zeros_like(param)
                self.vt[param_name] = np.zeros_like(param)

            # Compute first and second moment estimates
            self.mt[param_name] = self.beta1 * self.mt[param_name] + (1 - self.beta1) * layer.grads[param_name]
            self.vt[param_name] = self.beta2 * self.vt[param_name] + (1 - self.beta2) * layer.grads[param_name] ** 2

            # Bias correction terms
            mt_hat = self.mt[param_name] / (1 - self.beta1 ** (self.t + 1))
            vt_hat = self.vt[param_name] / (1 - self.beta2 ** (self.t + 1))

            # Update the parameter using Adam update rule
            layer.params[param_name] -= (self.lr * mt_hat) / (np.sqrt(vt_hat) + self.eps)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

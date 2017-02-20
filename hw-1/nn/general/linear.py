from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nn.module import Module
import numpy as np


class Linear(Module):
    def __init__(self, d_in, d_out, alpha=Module.default_alpha):
        super(Linear, self).__init__(d_in, d_out, alpha)
        self.w = np.random.normal(scale=0.1, size=(d_in, d_out))
        self.b = np.random.normal(scale=0.1, size=d_out)
        self.x = None

    def __update_output__(self, inputs, targets):
        self.x = inputs
        self.output = np.dot(inputs, self.w) + self.b

    def __update_grad_input__(self, grad_output):
        self.grad_input = np.matmul(grad_output, self.w.T)

    def update_parameters(self, grad_output):
        self.w -= self.alpha * np.dot(self.x.T, self.grad_output)
        self.b -= self.alpha * np.sum(grad_output, axis=0)
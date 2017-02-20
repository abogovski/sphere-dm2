from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nn.module import Module
import numpy as np


class Sigmoid(Module):
    def __init__(self, d):
        super(Sigmoid, self).__init__(d, d, None)
        self.s = None

    def __update_output__(self, inputs, targets):
        self.output = self.s = 1 / (1 + np.exp(-inputs))

    def __update_grad_input__(self, grad_output):
        self.grad_input = self.output * (1 - self.output) * grad_output

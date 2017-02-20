from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nn.module import Module
import numpy as np


class Tanh(Module):
    def __init__(self, d):
        super(Tanh, self).__init__(d, d, None)

    def __update_output__(self, inputs, targets):
        self.output = np.tanh(inputs)

    def __update_grad_input__(self, grad_output):
        self.grad_input = 1 / np.square(np.cosh(grad_output))

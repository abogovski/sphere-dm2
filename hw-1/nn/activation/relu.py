from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nn.module import Module
import numpy as np


class ReLU(Module):
    def __init__(self, d):
        super(ReLU, self).__init__(d, d, None)

    def __update_output__(self, inputs, targets):
        self.output = np.maximum(inputs, 0)

    def __update_grad_input__(self, grad_output):
        self.grad_input = (self.output > 0) * grad_output

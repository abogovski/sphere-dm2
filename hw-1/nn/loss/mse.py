from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nn.module import Module
import numpy as np
from numpy.linalg import norm


class MSE(Module):
    def __init__(self, d_in):
        super(MSE, self).__init__(d_in, 1, None)

    def __update_output__(self, inputs, targets):
        self.diff = inputs - targets
        self.output = np.square(np.norm(self.diff, axis=1)) / float(self.d_in)

    def __update_grad_input__(self, grad_output):
        self.grad_input = 2 * self.diff / float(self.d_in)

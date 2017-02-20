from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nn.module import Module
from nn.activation.softmax import compute_softmax
import numpy as np


class CrossEntropy(Module):
    def __init__(self, d):
        super(CrossEntropy, self).__init__(d, 1, None)
        self.targets = None
        self.s = None

    def __update_output__(self, inputs, targets):
        self.s = compute_softmax(inputs)
        self.targets = targets
        self.output = np.zeros(shape=(inputs.shape[0],1))
        for i in xrange(inputs.shape[0]):
            self.output[i, 0] = - np.dot(targets[i, :], np.log(self.s[i, :]))

    def __update_grad_input__(self, grad_output):
        self.grad_input = self.s - self.targets
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nn.module import Module
import numpy as np


def compute_softmax(inputs):
    output = np.zeros_like(inputs)
    e = np.exp(inputs)
    for i in xrange(inputs.shape[0]):
        output[i, :] = e[i, :] / np.sum(e[i, :])
    return output


class LogSoftMax(Module):
    def __init__(self, d):
        super(LogSoftMax, self).__init__(d, d, None)

    def __update_output__(self, inputs, targets):
        self.output = np.log(compute_softmax(inputs))

    def __update_grad_input__(self, grad_output):
        self.grad_input = np.zeros_like(grad_output)
        s = np.exp(self.output)
        for i in xrange(grad_output.shape[0]):
            self.grad_input[i, :] = grad_output[i, :] - np.dot(s[i, :], grad_output[i, :])


class SoftMax(Module):
    def __init__(self, d):
        super(SoftMax, self).__init__(d, d, None)

    def __update_output__(self, inputs, targets):
        self.output = compute_softmax(inputs)

    def __update_grad_input__(self, grad_output):
        self.grad_input = np.zeros_like(grad_output)
        for i in xrange(grad_output.shape[0]):
            s = self.output[i, :]
            self.grad_input[i, :] = s * (grad_output[i, :] - np.dot(s, grad_output[i, :]))

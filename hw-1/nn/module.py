from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.linalg import norm


class Module(object):
    default_alpha = 0.1

    def __init__(self, d_in, d_out, alpha):
        self.output = None
        self.grad_input = None
        self.rate = alpha
        self.d_in = d_in
        self.d_out = d_out

    @staticmethod
    def get_batch(var, dim=None, batch_len=None):
        if var is not None:
            var = np.atleast_2d(np.array(var))
            if len(var.shape) > 2:
                raise Exception('multidimensional batches are not supported')
            if dim is not None and var.shape[1] != dim:
                raise Exception('last dimension conflicts d_in or d_out')
            if batch_len is not None and var.shape[0] != batch_len:
                raise Exception('incompatible batch length')
        return var

    def forward(self, inputs, targets=None):
        inputs = self.get_batch(inputs, self.d_in)
        targets = self.get_batch(targets, self.d_in, inputs.shape[0])
        self.__update_output__(inputs, targets)
        return self.output

    def backward(self, grad_output):
        grad_output = self.get_batch(grad_output, self.d_out)
        self.__update_grad_input__(grad_output)
        self.__update_params__(grad_output)
        return self.grad_input

    def __update_output__(self, inputs, targets):
        raise NotImplementedError('implement forward pass')

    def __update_grad_input__(self, grad_output):
        raise NotImplementedError('implement grad_input calculation')

    def __update_params__(self, *args):
        pass


def grad_check(module, inputs, eps, rtol, targets=None):
    def grad_chk(x, t):
        x = np.repeat(np.atleast_2d(x), module.d_in, axis=0)
        if t is not None:
            t = np.repeat(np.atleast_2d(t), module.d_in, axis=0)
        out1 = module.forward(x + eps * np.identity(module.d_in), t)
        out0 = module.forward(x, t)
        module.__update_grad_input__(module.get_batch(np.identity(module.d_out), module.d_out))
        module_grad = module.grad_input
        computed_grad = (out1 - out0).T / eps
        return np.allclose(module_grad, computed_grad, rtol=rtol, atol=1.)

    inputs = module.get_batch(inputs)
    targets = module.get_batch(targets, module.d_in, inputs.shape[0])
    for i in xrange(inputs.shape[0]):
        t = None if targets is None else targets[i]
        if not grad_chk(inputs[i], t):
            return False
    return True

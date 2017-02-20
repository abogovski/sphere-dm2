from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nn.module import Module


class Sequential(Module):
    # TODO: refactor dims check system
    def __init__(self, d_in, d_out):
        super(Sequential, self).__init__(d_in, d_out, None)
        self.layers = []
        self.inner_dims_good = True

    def add(self, module, idx=-1):
        self.layers.insert(idx, module)
        self.inner_dims_good = False

    def remove(self, idx):
        self.inner_dims_good = False
        return self.layers.pop(idx)

    def check_dimensions(self, raise_on_mismatch=False):
        def chk_boundaries():
            if self.d_in is None or self.d_out is None:
                return False
            if not self.layers and self.d_in != self.d_out:
                return False
            return self.d_in == self.layers[0] and self.d_out[-1] == self.layers[-1]

        def chk_inner():
            if not self.inner_dims_good:
                self.inner_dims_good = True
                for i in xrange(1, len(self.layers)):
                    if self.layers[i-1].d_out != self.layers[i].d_in:
                        self.inner_dims_good = False
                        break
            return self.inner_dims_good

        valid = chk_boundaries() and chk_inner()
        if raise_on_mismatch and not valid:
            raise AttributeError('Sequential layer contains incompatible dimensions.')
        return valid

    def __update_output__(self, inputs, targets):
        self.check_dimensions(raise_on_mismatch=True)
        if self.layers:
            for m in self.layers[:-1:]:
                inputs = m.forward(inputs)
            inputs = self.layers[-1].forward(inputs, targets)
        self.output = inputs

    def update_grad_input(self, grad_output):
        self.check_dimensions(raise_on_mismatch=True)
        for m in self.layers[::-1]:
            grad_output = m.backward(grad_output)
        self.grad_input = grad_output

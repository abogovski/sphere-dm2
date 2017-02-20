from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import nn
from mnist import MNIST

def grad_check_all(inputs, eps, tolerance):
    print('gradient check', end='...')
    assert(nn.grad_check(nn.Sigmoid(inputs.shape[1]), inputs, eps, tolerance))
    assert(nn.grad_check(nn.LogSoftMax(inputs.shape[1]), inputs, eps, tolerance))
    assert(nn.grad_check(nn.SoftMax(inputs.shape[1]), inputs, eps, tolerance))
    assert(nn.grad_check(nn.ReLU(inputs.shape[1]), inputs, eps, tolerance))
    assert(nn.grad_check(nn.Tanh(inputs.shape[1]), inputs, eps, tolerance))
    assert(nn.grad_check(nn.Linear(inputs.shape[1], max(1, inputs.shape[1] - 3)), inputs, eps, tolerance))
    assert(nn.grad_check(nn.CrossEntropy(inputs.shape[1]), inputs, eps, tolerance, np.random.rand(*inputs.shape)))
    print('[OK]')


def main():
    # do gradient check
    batch = np.random.rand(20, 10)
    grad_check_all(batch, 1e-8, 1e-6)

    # load data
    mndata = MNIST("./../../../datasets/mnist")  # using python-mnist 3.0 package
    inputs,targets = mndata.load_training()

    # init
    dim = len(inputs[0])
    model = nn.Sequential(dim, 1)
    model.add(nn.Linear(dim, dim))
    model.add(nn.LogSoftMax(dim))
    model.add(nn.CrossEntropy(dim))

    print(len(inputs), len(targets))

    N = len(inputs)
    for i in xrange(N):
        model.forward(np.atleast_2d(inputs[i]).T, targets[i])
        model.backward([1])
    loss = model.forward()
    print("Loss: ", loss)


if __name__ == '__main__':
    main()

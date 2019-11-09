# -*- coding: utf-8 -*-
# @Time    : 2019/11/9 21:11
# @Author  : tys
# @Email   : yangsongtang@gmail.com
# @File    : linear_regression_01.py
# @Software: PyCharm

from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random


num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

batch_size = 10


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)


def linreg(X, w, b):
    return nd.dot(X, w) + b


def squard_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))
w.attach_grad()
b.attach_grad()

lr = 0.03
num_epochs = 3
net = linreg
loss = squard_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)

        l.backward()
        sgd([w, b], lr, batch_size)

    train_l = loss(net(features, w, b), labels)
    print('eopch %d, loss %f' % (epoch + 1, train_l.mean().asnumpy()))

print(true_w, w)
print(true_b, b)
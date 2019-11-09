# -*- coding: utf-8 -*-
# @Time    : 2019/11/9 21:12
# @Author  : tys
# @Email   : yangsongtang@gmail.com
# @File    : linear_regression_02.py
# @Software: PyCharm


from mxnet import nd, autograd
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet import init
from mxnet.gluon import loss as gloss
from mxnet import gluon

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

batch_size = 10
data_set = gdata.ArrayDataset(features, labels)
data_iter = gdata.DataLoader(data_set, batch_size, shuffle=True)

net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(init.Normal(sigma=0.01))
loss = gloss.L2Loss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y).mean()
        l.backward()
        trainer.step(1)
    print('eopch %d, loss %f' % (epoch + 1, l.mean().asnumpy()))


dense = net[0]
print(true_w, dense.weight.data())
print(true_b, dense.bias.data())

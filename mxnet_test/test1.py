from mxnet import nd, autograd
from mxnet.gluon import nn


# x = nd.arange(4).reshape((4, 1))
# print(x)
# x.attach_grad()
#
# with autograd.record():
#     y = 2 * nd.dot(x.T, x)
#
# y.backward()
# print(y)
# assert(x.grad - 4 * x).norm().asscalar() == 0
# print(x.grad)

# x = nd.arange(4).reshape((4, 1))
# x = nd.zeros((4, 1))
# x[0, 0] = 2
# x[3, 0] = 1
# print(x)
# y = (nd.dot(x.T, x))
# print(y)
# x.attach_grad()
#
# with autograd.record():
#     y = nd.dot(x.T, x)
#
# y.backward()
# print(x.grad)

x = nd.array([[1, 2], [3, 4]])

x.attach_grad()

with autograd.record():
    y = x * 2
    z = y * x

z.backward()

print('x.grad:', x.grad)

'''
对控制流进行求导
'''


def f(a):
    b = a * 2
    while nd.norm(b).asscalar() < 1000:
        b = b * 2
    if nd.sum(b).asscalar() > 0:
        c = b
    else:
        c = 100 * b
    return c


a = nd.random_normal(shape=3)
a.attach_grad()
with autograd.record():
    c = f(a)

c.backward()

print('c grad:', a.grad)




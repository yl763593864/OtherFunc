from mxnet import nd
from distutils.core import setup
import autopy3 as autopy

def hello_the_world():
    autopy.alert.alert('Hello, World')

# hello_the_world()

autopy.mouse.smooth_move(50, 50)


# X = nd.array([[1, 2, 3],
#               [4, 5, 6]])
#
# print(X.exp())
# print(X.exp().sum(axis=1))
# print((X.exp()).sum(axis=1, keepdims=True))
# print(X.exp() / ((X.exp()).sum(axis=1, keepdims=True)))

# y_hat = nd.array([[0.1, 0.3, 0.6], [0.2, 0.3, 0.5]])
# y = nd.array([0, 0])
# print(nd.pick(y_hat, y).log())
# x = nd.array([2.71828, 100])
# print(x.log())

# b = nd.array([0.7, 0.1, 0.2])
# a = nd.array([0.7, 0.2, 0.1])
# print(-a.log())
# print((a * -a.log()).sum(axis=0))
# kl = (a * (b/a).log()).sum(axis=0)
# print('kl:', kl)
# h_kl = (b * a.log() * -1).sum(axis=0)
# print('h_kl:', h_kl)
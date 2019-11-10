from mxnet import nd

# X = nd.array([[1, 2, 3],
#               [4, 5, 6]])
#
# print(X.exp())
# print(X.exp().sum(axis=1))
# print((X.exp()).sum(axis=1, keepdims=True))
# print(X.exp() / ((X.exp()).sum(axis=1, keepdims=True)))

y_hat = nd.array([[0.1, 0.3, 0.6], [0.2, 0.3, 0.5]])
y = nd.array([0, 0])
print(nd.pick(y_hat, y).log())
x = nd.array([2.71828, 100])
print(x.log())
# -*- coding: utf-8 -*-
# @Time    : 2019/11/4 20:56
# @Author  : tys
# @Email   : yangsongtang@gmail.com
# @File    : convolution.py
# @Software: PyCharm

import numpy as np
import scipy.special

class NeuralNetwork:

    def __init__(self, lr):
        self.f1 = np.array([[-1.27730064422524, -0.454172162085001, 0.358057715839741],
                      [1.13825014322686, -2.39770457627552, -1.66424031011426],
                      [-0.794134077996211, 0.898583299648462, 0.675446204740727]])
        self.f2 = np.array([[-1.27413497738694, 2.33817332831929, 2.30149047293595],
                      [0.649226815570952, -0.339421187754501, -2.05399200337461],
                      [-1.02192946600546, -1.20445535554298, -1.89956835390658]])
        self.f3 = np.array([[-1.86916063635104, 2.04412149106056, -1.28973337219918],
                       [-1.71011366582004, -2.09086913499937, -2.94551124620789],
                       [0.200874598839371, -1.32285494527113, 0.206768687932447]])
        self.f_bias = np.array([-3.36286301879535, -3.17635443160469, -1.73919464757302])

        self.o1_p1 = np.array([[-0.275747118525667, 0.124448118346582],
                                [-0.961374727597676, 0.717521785409812]])
        self.o1_p2 = np.array([[-3.68037877389129, -0.594173531237077],
                               [0.280479010075263, -0.782201884365295]])
        self.o1_p3 = np.array([[-1.47545213886256, -2.00959584191174],
                              [-1.08543311765283, -0.188094433907142]])

        self.o2_p1 = np.array([[0.0102268520154343, 0.660504529543313],
                              [-1.59129639133946, 2.18914483432551]])
        self.o2_p2 = np.array([[1.7281397324783, 0.00348085822933052],
                              [-0.250073860611539, 1.89754512739309]])
        self.o2_p3 = np.array([[0.238156398148505, 1.58869076601118],
                              [2.24643375666837, -0.0926071900914775]])

        self.o3_p1 = np.array([[-1.32191108970296, -0.217675096323542],
                              [3.5266850324431, 0.0609193290771542]])
        self.o3_p2 = np.array([[0.612727450303932, 0.217952343864534],
                              [-2.13027321488121, -1.67829858835797]])
        self.o3_p3 = np.array([[1.23567557077078, -0.486471873086058],
                              [-0.144296581719309, -1.23461550134234]])
        self.o_bias = np.array([2.05974733875712, -2.74625195414688, -1.81779135388])

        self.lr = lr


        self.activation_func = lambda x : scipy.special.expit(x)

    def train(self, inputs, targets, times=1):
        zf1 = self.get_zf(inputs, self.f1, self.f_bias[0])
        zf2 = self.get_zf(inputs, self.f2, self.f_bias[1])
        zf3 = self.get_zf(inputs, self.f3, self.f_bias[2])

        af1 = self.activation_func(zf1)
        af2 = self.activation_func(zf2)
        af3 = self.activation_func(zf3)

        ap1 = self.get_ap(af1)
        ap2 = self.get_ap(af2)
        ap3 = self.get_ap(af3)

        zo1 = self.get_out(self.o1_p1, ap1, self.o1_p2, ap2, self.o1_p3, ap3, self.o_bias[0])
        zo2 = self.get_out(self.o2_p1, ap1, self.o2_p2, ap2, self.o2_p3, ap3, self.o_bias[1])
        zo3 = self.get_out(self.o3_p1, ap1, self.o3_p2, ap2, self.o3_p3, ap3, self.o_bias[2])

        ao1 = self.activation_func(zo1)
        ao2 = self.activation_func(zo2)
        ao3 = self.activation_func(zo3)

        total_error = (np.power(targets[0] - ao1, 2) + np.power(targets[1] - ao2, 2) + np.power(targets[2] - ao3, 2))/2
        print(total_error)
        az1 = (ao1 - targets[0]) * ao1 * (1 - ao1)
        az2 = (ao2 - targets[1]) * ao2 * (1 - ao2)
        az3 = (ao3 - targets[2]) * ao3 * (1 - ao3)

        b1 = self.get_f(az1, az2, az3, self.o1_p1, self.o2_p1, self.o3_p1, ap1, af1, inputs, self.f1)
        b2 = self.get_f(az1, az2, az3, self.o1_p2, self.o2_p2, self.o3_p2, ap2, af2, inputs, self.f2)
        b3 = self.get_f(az1, az2, az3, self.o1_p3, self.o2_p3, self.o3_p3, ap3, af3, inputs, self.f3)
        self.get_p(az1, ap1, self.o1_p1)
        self.get_p(az1, ap2, self.o1_p2)
        self.get_p(az1, ap3, self.o1_p3)
        self.get_p(az2, ap1, self.o2_p1)
        self.get_p(az2, ap2, self.o2_p2)
        self.get_p(az2, ap3, self.o2_p3)
        self.get_p(az3, ap1, self.o3_p1)
        self.get_p(az3, ap2, self.o3_p2)
        self.get_p(az3, ap3, self.o3_p3)

        self.o_bias += np.array([az1, az2, az3]) * self.lr * -1
        self.f_bias += np.array([b1, b2, b3]) * self.lr * -1



    def get_zf(self, inputs, f1, bias):
        zf = np.zeros((4, 4))
        for x in np.arange(np.shape(zf)[0]):
            for y in np.arange(np.shape(zf)[1]):
                temp = 0
                for x1 in np.arange(np.shape(f1)[0]):
                    for y1 in np.arange((np.shape(f1)[1])):
                        temp += f1[x1][y1] * inputs[x + x1][y + y1]
                zf[x][y] = temp + bias
        return zf

    def get_ap(self, af1):
        ap = np.zeros((2, 2))
        for i in np.arange(np.shape(ap)[0]):
            for j in np.arange(np.shape(ap)[1]):
                ap[i][j] = max(af1[2*i][2*j],
                               af1[2*i][2*j + 1],
                               af1[2*i + 1][2*j],
                               af1[2*i + 1][2*j + 1])

        return  ap

    def get_out(self, o1p, a1p, o2p, a2p, o3p, a3p, bias):
        zo = 0
        for i in np.arange(np.shape(a1p)[0]):
            for j in np.arange(np.shape(a1p)[1]):
                zo += o1p[i][j] * a1p[i][j] + o2p[i][j] * a2p[i][j] + o3p[i][j] * a3p[i][j]
        return zo + bias

    def get_f(self, az1, az2, az3, o1_p1, o2_p1, o3_p1, ap1, af1, inputs, f1):
        ff = np.zeros((np.shape(af1)))
        f = np.zeros((3, 3))
        for i in np.arange(np.shape(af1)[0]):
            for j in np.arange(np.shape(af1)[1]):
                ff[i][j] = (az1 * o1_p1[i//2][j//2] + az2 * o2_p1[i//2][j//2] + az3 * o3_p1[i//2][j//2]) * af1[i][j] * (1 - af1[i][j])
                if af1[i][j] >= ap1[i//2, j//2]:
                    ff[i][j] *= 1
                else:
                    ff[i][j] *= 0

        for i in np.arange(np.shape(f)[0]):
            for j in np.arange(np.shape(f)[1]):
               for x in np.arange(4):
                   for y in np.arange(4):
                       f[i][j] += ff[x][y] * inputs[i + x][j + y]

        f1 += f * self.lr * -1
        bias = 0
        for i in np.arange(4):
            for j in np.arange(4):
                bias += ff[i][j]

        return bias

    def get_p(self, az1, ap1, o1_p1):
        pp = np.zeros((2, 2))
        pp = ap1 * az1
        self.o1_p1 += pp * self.lr * -1



if __name__ == '__main__':
        inputs_data = np.array([[0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 1, 0, 0]])
        in1 = np.array([[0,	0,	0,	1,	0,	0],
                    [0,	0,	1,	1,	0,	0],
                    [0,	0,	0,	1,	0,	0],
                    [0,	0,	0,	1,	0,	0],
                    [0,	0,	0,	1,	0,	0],
                    [0,	0,	0,	1,	0,	0]])

        targets_data = np.array([1, 0, 0])
        learning_rate = 0.2

        nn = NeuralNetwork(learning_rate)
        for i in np.arange(100):
            nn.train(inputs_data, targets_data)

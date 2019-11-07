import numpy as np
import scipy.special


# class NeuralNetwork:
#
#     def __init__(self, lr=0.2):
#         self.f1 = np.random.normal(0.0, 1.0, (5, 5))
#         self.f2 = np.random.normal(0.0, 1.0, (5, 5))
#         self.f3 = np.random.normal(0.0, 1.0, (5, 5))
#         self.f_bias = np.random.normal(0.0, 1.0, (3, 1))
#
#         self.o1_p1 = np.random.normal(0.0, 1.0, (2, 2))
#         self.o1_p2 = np.random.normal(0.0, 1.0, (2, 2))
#         self.o1_p3 = np.random.normal(0.0, 1.0, (2, 2))
#
#         self.o2_p1 = np.random.normal(0.0, 1.0, (2, 2))
#         self.o2_p2 = np.random.normal(0.0, 1.0, (2, 2))
#         self.o2_p3 = np.random.normal(0.0, 1.0, (2, 2))
#
#         self.o3_p1 = np.random.normal(0.0, 1.0, (2, 2))
#         self.o3_p2 = np.random.normal(0.0, 1.0, (2, 2))
#         self.o3_p3 = np.random.normal(0.0, 1.0, (2, 2))
#
#         self.o4_p1 = np.random.normal(0.0, 1.0, (2, 2))
#         self.o4_p2 = np.random.normal(0.0, 1.0, (2, 2))
#         self.o4_p3 = np.random.normal(0.0, 1.0, (2, 2))
#
#         self.o_bias = np.random.normal(0.0, 1.0, (4, 1))
#
#         self.lr = lr
#
#
#         self.activation_func = lambda x : scipy.special.expit(x)
#
#     def train(self, inputs, targets, times=1):
#         zf1 = self.get_zf(inputs, self.f1, self.f_bias[0])
#         zf2 = self.get_zf(inputs, self.f2, self.f_bias[1])
#         zf3 = self.get_zf(inputs, self.f3, self.f_bias[2])
#
#         af1 = self.activation_func(zf1)
#         af2 = self.activation_func(zf2)
#         af3 = self.activation_func(zf3)
#
#         ap1 = self.get_ap(af1)
#         ap2 = self.get_ap(af2)
#         ap3 = self.get_ap(af3)
#
#         zo1 = self.get_out(self.o1_p1, ap1, self.o1_p2, ap2, self.o1_p3, ap3, self.o_bias[0])
#         zo2 = self.get_out(self.o2_p1, ap1, self.o2_p2, ap2, self.o2_p3, ap3, self.o_bias[1])
#         zo3 = self.get_out(self.o3_p1, ap1, self.o3_p2, ap2, self.o3_p3, ap3, self.o_bias[2])
#         zo4 = self.get_out(self.o4_p1, ap1, self.o4_p2, ap2, self.o4_p3, ap3, self.o_bias[3])
#
#         ao1 = self.activation_func(zo1)
#         ao2 = self.activation_func(zo2)
#         ao3 = self.activation_func(zo3)
#         ao4 = self.activation_func(zo4)
#
#         total_error = (np.power(targets[0] - ao1, 2) + np.power(targets[1] - ao2, 2) + np.power(targets[2] - ao3, 2) + np.power(targets[3] - ao4, 2))/2
#         # print(total_error)
#         az1 = (ao1 - targets[0]) * ao1 * (1 - ao1)
#         az2 = (ao2 - targets[1]) * ao2 * (1 - ao2)
#         az3 = (ao3 - targets[2]) * ao3 * (1 - ao3)
#         az4 = (ao4 - targets[3]) * ao4 * (1 - ao4)
#
#         b1 = self.get_f(az1, az2, az3, az4, self.o1_p1, self.o2_p1, self.o3_p1, self.o4_p1, ap1, af1, inputs, self.f1)
#         b2 = self.get_f(az1, az2, az3, az4,  self.o1_p2, self.o2_p2, self.o3_p2, self.o4_p2, ap2, af2, inputs, self.f2)
#         b3 = self.get_f(az1, az2, az3, az4, self.o1_p3, self.o2_p3, self.o3_p3, ap3, self.o4_p3, af3, inputs, self.f3)
#
#         self.get_p(az1, ap1, self.o1_p1)
#         self.get_p(az1, ap2, self.o1_p2)
#         self.get_p(az1, ap3, self.o1_p3)
#         self.get_p(az2, ap1, self.o2_p1)
#         self.get_p(az2, ap2, self.o2_p2)
#         self.get_p(az2, ap3, self.o2_p3)
#         self.get_p(az3, ap1, self.o3_p1)
#         self.get_p(az3, ap2, self.o3_p2)
#         self.get_p(az3, ap3, self.o3_p3)
#         self.get_p(az4, ap1, self.o4_p1)
#         self.get_p(az4, ap2, self.o4_p2)
#         self.get_p(az4, ap3, self.o4_p3)
#
#         self.o_bias += np.array([az1, az2, az3, az4]) * self.lr * -1
#         self.f_bias += np.array([[b1], [b2], [b3]]) * self.lr * -1
#
#         return total_error
#
#     def test(self, inputs, targets):
#         zf1 = self.get_zf(inputs, self.f1, self.f_bias[0])
#         zf2 = self.get_zf(inputs, self.f2, self.f_bias[1])
#         zf3 = self.get_zf(inputs, self.f3, self.f_bias[2])
#
#         af1 = self.activation_func(zf1)
#         af2 = self.activation_func(zf2)
#         af3 = self.activation_func(zf3)
#
#         ap1 = self.get_ap(af1)
#         ap2 = self.get_ap(af2)
#         ap3 = self.get_ap(af3)
#
#         zo1 = self.get_out(self.o1_p1, ap1, self.o1_p2, ap2, self.o1_p3, ap3, self.o_bias[0])
#         zo2 = self.get_out(self.o2_p1, ap1, self.o2_p2, ap2, self.o2_p3, ap3, self.o_bias[1])
#         zo3 = self.get_out(self.o3_p1, ap1, self.o3_p2, ap2, self.o3_p3, ap3, self.o_bias[2])
#         zo4 = self.get_out(self.o4_p1, ap1, self.o4_p2, ap2, self.o4_p3, ap3, self.o_bias[3])
#
#         ao1 = self.activation_func(zo1)
#         ao2 = self.activation_func(zo2)
#         ao3 = self.activation_func(zo3)
#         ao4 = self.activation_func(zo4)
#         total_error = (np.power(targets[0] - ao1, 2) + np.power(targets[1] - ao2, 2) + np.power(targets[2] - ao3, 2) + np.power(targets[3] - ao4, 2))/2
#
#         print('total error:', total_error)
#         print('target:', targets[0], targets[1], targets[2], targets[3])
#         print("test data:", ao1, ao2, ao3, ao4)
#
#         if total_error >= 0.1:
#             return False
#         else:
#             return True
#         # max_num = max(ao1, ao2, ao3, ao4)
#         # if ao1 == max_num:
#         #     return 0
#         # elif ao2 == max_num:
#         #     return 1
#         # elif ao3 == max_num:
#         #     return 2
#         # elif ao4 == max_num:
#         #     return 3
#
#
#
#
#     def get_zf(self, inputs, f1, bias):
#         zf = np.zeros((4, 4))
#         for x in np.arange(np.shape(zf)[0]):
#             for y in np.arange(np.shape(zf)[1]):
#                 temp = 0
#                 for x1 in np.arange(np.shape(f1)[0]):
#                     for y1 in np.arange((np.shape(f1)[1])):
#                         temp += f1[x1][y1] * inputs[x + x1][y + y1]
#                 zf[x][y] = temp + bias
#         return zf
#
#     def get_ap(self, af1):
#         ap = np.zeros((2, 2))
#         for i in np.arange(np.shape(ap)[0]):
#             for j in np.arange(np.shape(ap)[1]):
#                 ap[i][j] = max(af1[2*i][2*j],
#                                af1[2*i][2*j + 1],
#                                af1[2*i + 1][2*j],
#                                af1[2*i + 1][2*j + 1])
#
#         return ap
#
#     def get_out(self, o1p, a1p, o2p, a2p, o3p, a3p, bias):
#         zo = 0
#         for i in np.arange(np.shape(a1p)[0]):
#             for j in np.arange(np.shape(a1p)[1]):
#                 zo += o1p[i][j] * a1p[i][j] + o2p[i][j] * a2p[i][j] + o3p[i][j] * a3p[i][j]
#         return zo + bias
#
#     def get_f(self, az1, az2, az3, az4, o1_p1, o2_p1, o3_p1, o4_p1, ap1, af1, inputs, f1):
#         ff = np.zeros((np.shape(af1)))
#         f = np.zeros((5, 5))
#         for i in np.arange(np.shape(af1)[0]):
#             for j in np.arange(np.shape(af1)[1]):
#                 ff[i][j] = (az1 * o1_p1[i//2][j//2] + az2 * o2_p1[i//2][j//2] + az3 * o3_p1[i//2][j//2] + az4 * o4_p1[i//2][j//2]) * af1[i][j] * (1 - af1[i][j])
#                 if af1[i][j] >= ap1[i//2, j//2]:
#                     ff[i][j] *= 1
#                 else:
#                     ff[i][j] *= 0
#
#         for i in np.arange(np.shape(f)[0]):
#             for j in np.arange(np.shape(f)[1]):
#                for x in np.arange(4):
#                    for y in np.arange(4):
#                        f[i][j] += ff[x][y] * inputs[i + x][j + y]
#
#         f1 += f * self.lr * -1
#         bias = 0
#         for i in np.arange(4):
#             for j in np.arange(4):
#                 bias += ff[i][j]
#
#         return bias
#
#     def get_p(self, az1, ap1, o1_p1):
#         pp = np.zeros((2, 2))
#         pp = ap1 * az1
#         self.o1_p1 += pp * self.lr * -1
#
#     def write_data(self):
#         np.save(r'Data/f1.npy', self.f1)
#         np.save(r'Data/f2.npy', self.f2)
#         np.save(r'Data/f3.npy', self.f3)
#         np.save(r'Data/f_bias.npy', self.f_bias)
#
#         np.save(r'Data/o1_p1.npy', self.o1_p1)
#         np.save(r'Data/o1_p2.npy', self.o1_p2)
#         np.save(r'Data/o1_p3.npy', self.o1_p3)
#
#         np.save(r'Data/o2_p1.npy', self.o2_p1)
#         np.save(r'Data/o2_p2.npy', self.o2_p2)
#         np.save(r'Data/o2_p3.npy', self.o2_p3)
#
#         np.save(r'Data/o3_p1.npy', self.o3_p1)
#         np.save(r'Data/o3_p2.npy', self.o3_p2)
#         np.save(r'Data/o3_p3.npy', self.o3_p3)
#
#         np.save(r'Data/o4_p1.npy', self.o4_p1)
#         np.save(r'Data/o4_p2.npy', self.o4_p2)
#         np.save(r'Data/o4_p3.npy', self.o4_p3)
#
#         np.save(r'Data/o_bias.npy', self.o_bias)
#
#     def read_data(self):
#         self.f1 = np.load(r'Data/f1.npy')
#         self.f2 = np.load(r'Data/f2.npy')
#         self.f3 = np.load(r'Data/f3.npy')
#         self.f_bias = np.load(r'Data/f_bias.npy')
#
#         self.o1_p1 = np.load(r'Data/o1_p1.npy')
#         self.o1_p2 = np.load(r'Data/o1_p2.npy')
#         self.o1_p3 = np.load(r'Data/o1_p3.npy')
#
#         self.o2_p1 = np.load(r'Data/o2_p1.npy')
#         self.o2_p2 = np.load(r'Data/o2_p2.npy')
#         self.o2_p3 = np.load(r'Data/o2_p3.npy')
#
#         self.o3_p1 = np.load(r'Data/o3_p1.npy')
#         self.o3_p2 = np.load(r'Data/o3_p2.npy')
#         self.o3_p3 = np.load(r'Data/o3_p3.npy')
#
#         self.o4_p1 = np.load(r'Data/o4_p1.npy')
#         self.o4_p2 = np.load(r'Data/o4_p2.npy')
#         self.o4_p3 = np.load(r'Data/o4_p3.npy')
#         self.o_bias = np.load(r'Data/o_bias.npy')


class NeuralNetwork:

    def __init__(self, lr):
        self.f1 = np.random.normal(0, 1, (3, 3))
        self.f2 = np.random.normal(0, 1, (3, 3))
        self.f3 = np.random.normal(0, 1, (3, 3))
        self.f_bias = np.random.normal(0, 1, (3, 1))

        self.o1_p1 = np.random.normal(0, 1, (2, 2))
        self.o1_p2 = np.random.normal(0, 1, (2, 2))
        self.o1_p3 = np.random.normal(0, 1, (2, 2))

        self.o2_p1 = np.random.normal(0, 1, (2, 2))
        self.o2_p2 = np.random.normal(0, 1, (2, 2))
        self.o2_p3 = np.random.normal(0, 1, (2, 2))

        self.o3_p1 = np.random.normal(0, 1, (2, 2))
        self.o3_p2 = np.random.normal(0, 1, (2, 2))
        self.o3_p3 = np.random.normal(0, 1, (2, 2))

        self.o4_p1 = np.random.normal(0, 1, (2, 2))
        self.o4_p2 = np.random.normal(0, 1, (2, 2))
        self.o4_p3 = np.random.normal(0, 1, (2, 2))

        self.o_bias = np.random.normal(0, 1, (4, 1))

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
        zo4 = self.get_out(self.o4_p1, ap1, self.o4_p2, ap2, self.o4_p3, ap3, self.o_bias[3])

        ao1 = self.activation_func(zo1)
        ao2 = self.activation_func(zo2)
        ao3 = self.activation_func(zo3)
        ao4 = self.activation_func(zo4)

        total_error = (np.power(targets[0] - ao1, 2) + np.power(targets[1] - ao2, 2) + np.power(targets[2] - ao3, 2) + np.power(target_data[3] - ao4, 2))/2
        # print(total_error)
        az1 = (ao1 - targets[0]) * ao1 * (1 - ao1)
        az2 = (ao2 - targets[1]) * ao2 * (1 - ao2)
        az3 = (ao3 - targets[2]) * ao3 * (1 - ao3)
        az4 = (ao4 - targets[3]) * ao4 * (1 - ao4)

        b1 = self.get_f(az1, az2, az3, az4, self.o1_p1, self.o2_p1, self.o3_p1, self.o4_p1, ap1, af1, inputs, self.f1)
        b2 = self.get_f(az1, az2, az3, az4, self.o1_p2, self.o2_p2, self.o3_p2, self.o4_p2, ap2, af2, inputs, self.f2)
        b3 = self.get_f(az1, az2, az3, az4, self.o1_p3, self.o2_p3, self.o3_p3, self.o4_p3, ap3, af3, inputs, self.f3)
        self.get_p(az1, ap1, self.o1_p1)
        self.get_p(az1, ap2, self.o1_p2)
        self.get_p(az1, ap3, self.o1_p3)
        self.get_p(az2, ap1, self.o2_p1)
        self.get_p(az2, ap2, self.o2_p2)
        self.get_p(az2, ap3, self.o2_p3)
        self.get_p(az3, ap1, self.o3_p1)
        self.get_p(az3, ap2, self.o3_p2)
        self.get_p(az3, ap3, self.o3_p3)
        self.get_p(az4, ap1, self.o4_p1)
        self.get_p(az4, ap2, self.o4_p2)
        self.get_p(az4, ap3, self.o4_p3)

        self.o_bias += np.array([az1, az2, az3, az4]) * self.lr * -1
        self.f_bias += np.array([[b1, b2, b3]]).T * self.lr * -1

        return total_error


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

        return ap

    def get_out(self, o1p, a1p, o2p, a2p, o3p, a3p, bias):
        zo = 0
        for i in np.arange(np.shape(a1p)[0]):
            for j in np.arange(np.shape(a1p)[1]):
                zo += o1p[i][j] * a1p[i][j] + o2p[i][j] * a2p[i][j] + o3p[i][j] * a3p[i][j]
        return zo + bias

    def get_f(self, az1, az2, az3, az4, o1_p1, o2_p1, o3_p1,o4_p1, ap1, af1, inputs, f1):
        ff = np.zeros((np.shape(af1)))
        f = np.zeros((3, 3))
        for i in np.arange(np.shape(af1)[0]):
            for j in np.arange(np.shape(af1)[1]):
                ff[i][j] = (az1 * o1_p1[i//2][j//2] + az2 * o2_p1[i//2][j//2] + az3 * o3_p1[i//2][j//2] + az4 * o4_p1[i//2][j//2]) * af1[i][j] * (1 - af1[i][j])
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
        o1_p1 += pp * self.lr * -1


    def write_data(self):
        np.save(r'Data/f1.npy', self.f1)
        np.save(r'Data/f2.npy', self.f2)
        np.save(r'Data/f3.npy', self.f3)
        np.save(r'Data/f_bias.npy', self.f_bias)

        np.save(r'Data/o1_p1.npy', self.o1_p1)
        np.save(r'Data/o1_p2.npy', self.o1_p2)
        np.save(r'Data/o1_p3.npy', self.o1_p3)

        np.save(r'Data/o2_p1.npy', self.o2_p1)
        np.save(r'Data/o2_p2.npy', self.o2_p2)
        np.save(r'Data/o2_p3.npy', self.o2_p3)

        np.save(r'Data/o3_p1.npy', self.o3_p1)
        np.save(r'Data/o3_p2.npy', self.o3_p2)
        np.save(r'Data/o3_p3.npy', self.o3_p3)

        np.save(r'Data/o4_p1.npy', self.o4_p1)
        np.save(r'Data/o4_p2.npy', self.o4_p2)
        np.save(r'Data/o4_p3.npy', self.o4_p3)

        np.save(r'Data/o_bias.npy', self.o_bias)

    def read_data(self):
        self.f1 = np.load(r'Data/f1.npy')
        self.f2 = np.load(r'Data/f2.npy')
        self.f3 = np.load(r'Data/f3.npy')
        self.f_bias = np.load(r'Data/f_bias.npy')

        self.o1_p1 = np.load(r'Data/o1_p1.npy')
        self.o1_p2 = np.load(r'Data/o1_p2.npy')
        self.o1_p3 = np.load(r'Data/o1_p3.npy')

        self.o2_p1 = np.load(r'Data/o2_p1.npy')
        self.o2_p2 = np.load(r'Data/o2_p2.npy')
        self.o2_p3 = np.load(r'Data/o2_p3.npy')

        self.o3_p1 = np.load(r'Data/o3_p1.npy')
        self.o3_p2 = np.load(r'Data/o3_p2.npy')
        self.o3_p3 = np.load(r'Data/o3_p3.npy')

        self.o4_p1 = np.load(r'Data/o4_p1.npy')
        self.o4_p2 = np.load(r'Data/o4_p2.npy')
        self.o4_p3 = np.load(r'Data/o4_p3.npy')
        self.o_bias = np.load(r'Data/o_bias.npy')

    def test_data(self, inputs, targets=None):
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
        zo4 = self.get_out(self.o4_p1, ap1, self.o4_p2, ap2, self.o4_p3, ap3, self.o_bias[3])

        ao1 = self.activation_func(zo1)
        ao2 = self.activation_func(zo2)
        ao3 = self.activation_func(zo3)
        ao4 = self.activation_func(zo4)

        if targets is not None:
            total_error = (np.power(targets[0] - ao1, 2) + np.power(targets[1] - ao2, 2) + np.power(targets[2] - ao3,
                                                                                                2) + np.power(
            target_data[3] - ao4, 2)) / 2
        print(total_error)
        a = [ao1, ao2, ao3, ao4]
        max_index = a.index(max(a))
        t_list = [targets[0], targets[1], targets[2], targets[3]]
        target_index = t_list.index(max(t_list))
        print(a, 'index', max_index, 'target index:', target_index)
        if max_index == target_index:
            return True
        else:
            return False




def generate_game():
    game_array = np.zeros((6, 6))

    zero_pos = [np.random.randint(0, 6), np.random.randint(0, 6)]
    target_pos = [np.random.randint(0, 6), np.random.randint(0, 6)]
    while True:
        if zero_pos == target_pos:
            zero_pos = [np.random.randint(0, 6), np.random.randint(0, 6)]
        else:
            break
    game_array[zero_pos[0], zero_pos[1]] = 0.8
    game_array[target_pos[0], target_pos[1]] = 1

    target_array = np.zeros((4, 1))
    pos = [target_pos[0] - zero_pos[0], target_pos[1] - zero_pos[1]]
    # print('pos', pos, target_pos, zero_pos)
    # print(game_array)
    if pos[0] >= 0:
        target_array[0] = 0
        target_array[1] = abs(pos[0])
    else:
        target_array[0] = abs(pos[0])
        target_array[1] = 0

    if pos[1] >= 0:
        target_array[2] = 0
        target_array[3] = abs(pos[1])
    else:
        target_array[2] = abs(pos[1])
        target_array[3] = 0

    max_num = max(target_array)
    # print('max num', max_num, target_array)
    for i in np.arange(4):
        if target_array[i][0] <= 0:
            target_array[i][0] = 0
        else:
            if target_array[i] == max_num:
                target_array[i] = 1
            else:
                target_array[i] /= 6
    print('target array', target_array)
    return game_array, target_array


if __name__ == '__main__':
    learning_rate = 0.1

    nn = NeuralNetwork(learning_rate)
    nn.read_data()
    true_num = 0
    for x in range(10000):
        inputs_data, target_data = generate_game()
        # for y in range(100):
        #     num = nn.train(inputs_data, target_data)
        #     if num <= 0.005:
        #         nn.write_data()
        #         break
        #     print('x', x, 'y', y, num)

        if nn.test_data(inputs_data, target_data):
            true_num += 1

        print('true num:', true_num, true_num/(x+1))



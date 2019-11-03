# -*- coding: utf-8 -*-
# @Time    : 2019/11/1 22:11
# @Author  : tys
# @Email   : yangsongtang@gmail.com
# @File    : BP_test.py
# @Software: PyCharm


import numpy as np
import scipy.spatial


class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        """
        init the neural network
        """

        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes
        self.lr = learning_rate

        self.w_input_hidden = (np.random.rand(self.h_nodes, self.i_nodes) - 0.5)
        self.w_hidden_output = (np.random.rand(self.o_nodes, self.h_nodes) - 0.5)
        self.w_input_hidden = np.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.w_hidden_output = np.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))
        self.tt = [[0.49038557898997, 0.442372911956926, 0.654393041569443],
                    [0.348475676796501, -0.536877487857221, -1.38856820257739],
                    [0.0725879008695083, 1.00782536916829, 1.24648311661583],
                    [0.837472826850604, 1.07196001297575, 0.0572877158406771],
                    [-0.0706798311519743, -0.732814485632708, -0.183237472237546],
                    [-3.6169369170322, 0.822959617857012, -0.74305066513479],
                    [-0.53557819719488, -0.453282364154155, -0.460930664925325],
                    [-0.0228584789393108, -0.0138979392949318, 0.331118557255208],
                    [-1.71745249082217, -0.0274233258563056, 0.449470835925128],
                    [-1.45563751579807, -0.426670298661898, -1.29645372413246],
                    [-0.555799932254451, 1.87560275441379, 1.56850561324256],
                    [0.852476539980059, -2.30528048189891, -0.470667153317658]]
        self.w_input_hidden = np.transpose(self.tt)
        self.b_input_hidden = np.array([-0.185002356132065, 0.525676844318642, -1.16862269778991], ndmin=2).T
        self.w_hidden_output = [[0.3880031194962, 0.803384989025837, 0.0292864334994403],
                                [0.0254467679708455, -0.790397993881956, 1.55313793058729]]
        self.b_hidden_output = np.array([-1.43803971240614, -1.37933790823328], ndmin=2).T

        self.activation_func = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list, times=1):
        """
        train the neural network
        """
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        for i in np.arange(times):
            hidden_inputs = np.dot(self.w_input_hidden, inputs)
            hidden_outputs = self.activation_func(hidden_inputs)

            final_inputs = np.dot(self.w_hidden_output, hidden_outputs)
            final_outputs = self.activation_func(final_inputs)

            output_errors = targets - final_outputs
            hidden_errors = np.dot(self.w_hidden_output.T, output_errors)

            self.w_hidden_output += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
            self.w_input_hidden += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
            print('times', i, 'output error:', np.transpose(output_errors), 'final output', np.transpose(final_outputs))

    def bp_train(self, inputs_list, targets_list, times=1):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        for i in np.arange(times):
            hidden_inputs = np.dot(self.w_input_hidden, inputs) + self.b_input_hidden
            hidden_outputs = self.activation_func(hidden_inputs)

            final_inputs = np.dot(self.w_hidden_output, hidden_outputs) + self.b_hidden_output
            final_outputs = self.activation_func(final_inputs)

            print("final outputs", final_outputs)
            output_errors = (np.power(targets[0] - final_outputs[0], 2) + np.power(targets[1] - final_outputs[1], 2))/2
            print("output_errors:", output_errors)
            # print('z2i:', hidden_inputs)
            # print('a2i:', hidden_outputs)
            # # az2 = self.activation_func(hidden_inputs) * (1.0 - self.activation_func(hidden_inputs))
            az2 = hidden_outputs * (1.0 - hidden_outputs)
            # print('az2:', az2)
            #
            # print('final inputs:', final_inputs)
            # print('final outputs:', final_outputs)
            # # az3 = self.activation_func(final_inputs) * (1.0 - self.activation_func(final_inputs))
            az3 = final_outputs * (1.0 - final_outputs)
            # print('az3:', az3)
            caa3 = final_outputs - targets
            # print('caa3:', caa3)
            a3 = caa3 * az3
            # print('a3:', a3)
            a2 = np.dot(np.transpose(self.w_hidden_output), a3) * az2
            # print('a2:', a2)

            self.b_hidden_output += self.lr * -1 * a3
            self.b_input_hidden += self.lr * -1 * a2

            self.w_hidden_output += self.lr * -1 * np.dot(a3, hidden_outputs.T)
            self.w_input_hidden += self.lr * -1 * np.dot(a2, inputs.T)


if __name__ == '__main__':
    i_nodes = 12
    h_nodes = 3
    o_nodes = 2
    lr = 0.2
    in_list = [1, 1, 1,
               1, 0, 1,
               1, 0, 1,
               1, 1, 1]
    out_list = [1, 0]

    n = NeuralNetwork(i_nodes, h_nodes, o_nodes, lr)
    n.bp_train(in_list, out_list, 50)

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
            hidden_inputs = np.dot(self.w_input_hidden, inputs)
            hidden_outputs = self.activation_func(hidden_inputs)

            final_inputs = np.dot(self.w_hidden_output, hidden_outputs)
            final_outputs = self.activation_func(final_inputs)

            print("final outputs", final_outputs)
            output_errors = (np.power(targets[0] - final_outputs[0], 2) + np.power(targets[1] - final_outputs[1], 2))/2
            print("output_errors:", output_errors)
            d_final_inputs = (targets - final_outputs) * final_inputs * (1.0 - final_inputs)

            self.w_hidden_output += self.lr * np.dot(d_final_inputs, np.transpose(hidden_outputs))
            d_hidden_inputs = np.dot(np.transpose(self.w_hidden_output), d_final_inputs) * hidden_inputs * (1.0 - hidden_inputs)
            self.w_input_hidden += self.lr * np.dot(d_hidden_inputs, np.transpose(inputs))

            print()



if __name__ == '__main__':
    i_nodes = 12
    h_nodes = 3
    o_nodes = 2
    lr = 0.3
    in_list = [0.9, 0.9, 0.9,
               0.9, 0.1, 0.9,
               0.9, 0.1, 0.9,
               0.9, 0.9, 0.9]
    out_list = [0.99, 0.01]

    n = NeuralNetwork(i_nodes, h_nodes, o_nodes, lr)
    n.bp_train(in_list, out_list, 1)
# -*- coding: utf-8 -*-
# @Time    : 2019/10/26 13:46
# @Author  : tys
# @Email   : yangsongtang@gmail.com
# @File    : neural_network.py
# @Software: PyCharm

import numpy as np
import scipy.special


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

    def query(self, inputs_list):
        """
        query the neural network
        """
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.w_input_hidden, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)

        final_inputs = np.dot(self.w_hidden_output, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)

        return final_outputs


if __name__ == '__main__':
    i_nodes = 3
    h_nodes = 10000
    o_nodes = 3
    lr = 0.001
    in_list = [0.9, 0.5, -0.5]
    out_list = [0.99, 0.3, 0.99]

    n = NeuralNetwork(i_nodes, h_nodes, o_nodes, lr)
    n.train(in_list, out_list, 10000)


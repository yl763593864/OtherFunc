# -*- coding: utf-8 -*-
# @Time    : 2019/10/27 20:50
# @Author  : tys
# @Email   : yangsongtang@gmail.com
# @File    : hand_writing.py
# @Software: PyCharm


import numpy as np
import scipy.spatial
import time


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
            # print('times', i, 'output error:', np.transpose(output_errors), 'final output', np.transpose(final_outputs))


    def query(self, inputs_list):
        """
        query the neural network
        """
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.w_input_hidden, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)

        final_inputs = np.dot(self.w_hidden_output, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)

        value_list = []
        for j in np.arange(self.o_nodes):
            value_list.append(final_outputs[j])
        return value_list.index(max(value_list))


if __name__ == '__main__':
    start_time = time.process_time()

    i_nodes = 784
    h_nodes = 100
    o_nodes = 10
    lr = 0.3

    n = NeuralNetwork(i_nodes, h_nodes, o_nodes, lr)

    train_data_list = []
    # 文件太大不能上传到git上, 放在OneDrive里面的
    with open('train.txt') as fs:
        train_data_list = fs.readlines()

    for record in train_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(o_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets, 1)

    test_data_list = []
    with open('test.txt') as fs:
        test_data_list = fs.readlines()

    correct_num = 0
    for record in test_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        output_value = n.query(inputs)
        print('target', all_values[0], 'output', output_value)
        if str(all_values[0]) == str(output_value):
            correct_num += 1
    print(correct_num/len(test_data_list))
    print('time used', time.process_time() - start_time)







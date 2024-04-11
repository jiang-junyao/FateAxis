#!/usr/bin/env python3
"""
CNN1D CLF
"""

import os
import torch
import itertools
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class Limited(nn.Module):
    """
    Defining a CNN model treating input as 1D data
    with given hyperparameters
    Layer set number limited to max == 3
    """
    def __init__(self, id, param, n_class = 2):
        # Initialization
        super().__init__()
        self.id = id
        self.model_type = 'CNN_1D_Limited'
        self.num_layers = param['num_layers']
        self.loss_func = nn.CrossEntropyLoss()
        # Layer set 1
        self.pool = nn.MaxPool1d(param['maxpool_kernel_size'])
        self.conv = nn.Conv1d(
            1,
            param['conv_kernel_num'],
            param['conv_kernel_size']
        )

        # Layer set 2
        self.pool1 = nn.MaxPool1d(param['maxpool_kernel_size'])
        self.conv1 = nn.Conv1d(
            param['conv_kernel_num'],
            param['conv_kernel_num'],
            param['conv_kernel_size']
        )

        # Layer set 3
        self.pool2 = nn.MaxPool1d(param['maxpool_kernel_size'])
        self.conv2 = nn.Conv1d(
            param['conv_kernel_num'],
            param['conv_kernel_num'],
            param['conv_kernel_size']
        )

        ### Trying to avoid Lazy module since it's under development ###
        ### But so far it's working just fine, so still using lazy module ###
        # flattenLength = int(featureNum / pow(maxpool_kernel_size, num_layers))
        # self.dense = nn.Linear(flattenLength, densed_size)
        ### -------------------------------------------------------- ###

        self.dense = nn.LazyLinear(param['densed_size'])
        self.decision = nn.Linear(param['densed_size'], n_class)
        self.optimizer = optim.SGD(self.parameters(), param['learning_rate'])

    # Overwrite the forward function in nn.Module
    def forward(self, input):

        input = self.pool(F.relu(self.conv(input)))
        if self.num_layers > 1:
            input = self.pool1(F.relu(self.conv1(input)))
        if self.num_layers > 2:
            input = self.pool2(F.relu(self.conv2(input)))
 
        input = torch.flatten(input, start_dim = 1)
        input = F.relu(self.dense(input))
        input = F.softmax(self.decision(input), dim = 1)
        return input


class Unlimited(nn.Module):
    """
    Defining a CNN model treating input as 1D data
    with given hyperparameters
    """
    def __init__(self, id, param, n_class = 2):
        # Initialization
        super().__init__()
        self.id = id
        self.model_type = 'CNN_1D_Unlimited'
        self.num_layers = param['num_layers']
        self.loss_func = nn.CrossEntropyLoss()
        self.conv = nn.Conv1d(
            1,
            param['conv_kernel_num'],
            param['conv_kernel_size']
        )
        self.convMore = nn.Conv1d(
            param['conv_kernel_num'],
            param['conv_kernel_num'],
            param['conv_kernel_size']
        )
        self.pool = nn.MaxPool1d(param['maxpool_kernel_size'])

        ### Trying to avoid Lazy module since it's under development ###
        ### But so far it's working just fine, so still using lazy module ###
        # flattenLength = int(featureNum / pow(maxpool_kernel_size, num_layers))
        # self.dense = nn.Linear(flattenLength, densed_size)
        ### -------------------------------------------------------- ###

        self.dense = nn.LazyLinear(param['densed_size'])
        self.decision = nn.Linear(param['densed_size'], n_class)
        self.optimizer = optim.SGD(self.parameters(), param['learning_rate'])

    # Overwrite the forward function in nn.Module
    def forward(self, input):
        input = self.pool(F.relu(self.conv(input)))
        for i in range(self.num_layers - 1):
            input = self.pool(F.relu(self.convMore(input)))
        input = torch.flatten(input, start_dim = 1)
        input = F.relu(self.dense(input))
        input = F.softmax(self.decision(input), dim = 1)
        return input




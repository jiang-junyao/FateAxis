
import torch
import itertools
import torch.nn as nn
from warnings import warn
import torch.optim as optim
import torch.nn.functional as F
import math

class Limited(nn.Module):
    """
    Defining a CNN model treating input as 2D data
    with given hyperparameters
    then using 2 1D convolution kernels to generate layers
    Layer set number limited to max == 3
    """
    def __init__(self, id, param, n_class = 2):
        super().__init__()

        # Initialization
        self.id = id
        self.model_type = 'CNN_Hybrid_Limited'
        self.matrixSize = [5,40]
        self.num_layers = param['num_layers']
        self.loss_func = nn.CrossEntropyLoss()

        # Layer set 1
        self.poolVer = nn.MaxPool2d((1, param['maxpool_kernel_size']))
        self.convVer = nn.Conv2d(
            1,
            param['conv_kernel_num'],
            (1, self.matrixSize[1])
        )
        self.poolHor = nn.MaxPool2d((param['maxpool_kernel_size'], 1))
        self.convHor = nn.Conv2d(
            1,
            param['conv_kernel_num'],
            (self.matrixSize[0], 1)
        )
        self.poolVer1 = nn.MaxPool2d((1, param['maxpool_kernel_size']))
        self.convVer1 = nn.Conv2d(
            param['conv_kernel_num'],
            param['conv_kernel_num'],
            (int(
            self.matrixSize[1]/pow(param['maxpool_kernel_size'],self.num_layers)
            ), 1)
        )
        self.poolHor1 = nn.MaxPool2d((param['maxpool_kernel_size'], 1))
        self.convHor1 = nn.Conv2d(
            param['conv_kernel_num'],
            param['conv_kernel_num'],
            (1, int(
            self.matrixSize[0]/pow(param['maxpool_kernel_size'],self.num_layers)
            ))
        )

        # Layer set 3
        self.poolVer2 = nn.MaxPool2d((1, param['maxpool_kernel_size']))
        self.convVer2 = nn.Conv2d(
            param['conv_kernel_num'],
            param['conv_kernel_num'],
            (int(
            self.matrixSize[1]/pow(param['maxpool_kernel_size'],self.num_layers)
            ), 1)
        )
        self.poolHor2 = nn.MaxPool2d((param['maxpool_kernel_size'], 1))
        self.convHor2 = nn.Conv2d(
            param['conv_kernel_num'],
            param['conv_kernel_num'],
            (1, int(
            self.matrixSize[0]/pow(param['maxpool_kernel_size'],self.num_layers)
            ))
        )


        ### Same problem as 1D model ###
        # flattenLength = int(featureNum / pow(maxpool_kernel_size, num_layers))
        # self.dense = nn.Linear(flattenLength, densed_size)
        self.dense = nn.LazyLinear(param['densed_size'])
        self.decision = nn.Linear(param['densed_size'], n_class)
        self.optimizer = optim.SGD(self.parameters(), param['learning_rate'])

    # Overwrite the forward function in nn.Module
    def forward(self, input):
        #self.__check_input_matrix_size(input.shape[1]*input.shape[0])
        input = self.reshape(input)
        self.matrixSize = [5,40]
        temp0 = self.poolVer(F.relu(self.convHor(input)))
        temp1 = self.poolHor(F.relu(self.convVer(input)))
        if self.num_layers > 1:
            temp0 = self.poolVer1(F.relu(self.convHor1(temp0)))
            temp1 = self.poolHor1(F.relu(self.convVer1(temp1)))
        if self.num_layers > 2:
            temp0 = self.poolVer2(F.relu(self.convHor2(temp0)))
            temp1 = self.poolHor2(F.relu(self.convVer2(temp1)))

        temp0 = torch.flatten(temp0, start_dim = 1)
        temp1 = torch.flatten(temp1, start_dim = 1)
        input = torch.cat((temp0, temp1), dim = 1)
        input = F.relu(self.dense(input))
        input = F.softmax(self.decision(input),dim = 1)
        return input

    # transform input(1D) into a 2D matrix
    def reshape(self, input):
        return torch.reshape(input, (input.shape[0], 1,
                                    13, 154))
    
    def __check_input_matrix_size(self, grp_amount):
        matrix_dim = int(math.sqrt(grp_amount))
        square_size = [matrix_dim, matrix_dim]
        self.matrixSize = square_size





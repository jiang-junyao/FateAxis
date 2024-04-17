#!/usr/bin/env python3
"""
This file contains classes to build LSTM based classifiers

ToDo:
Implement _earlystopping

author: jy, nkmtmsys
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F




class LSTM(nn.Module):
    """
    Recurrent neural network (many-to-one)
    """
    def __init__(self,
                 device,
                 id,
                 input_size,
                 num_layers,
                 hidden_size,
                 dropout,
                 learning_rate,
                 n_class = 2,
                 bias = 'True',
                 bidirectional = 'False',
                 proj_size = 0,
                ):
        super(LSTM, self).__init__()
        self.device=device
        self.id = id
        self.model_type = 'LSTM'
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p = dropout)
        self.lstm = nn.LSTM(
            input_size,
            self.hidden_size,
            self.num_layers,
            batch_first = True,
            bias = bias,
            bidirectional = bidirectional,
            proj_size = proj_size
        )
        self.fc = nn.Linear(self.hidden_size, n_class)
        self.optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, input):
        input = self.dropout(input)
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        h0 = h0.to(self.device)
        c0 = c0.to(self.device)
        # Forward propagate LSTM
        out, _ = self.lstm(input, (h0, c0))
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = F.softmax(self.fc(out[:, -1, :]), dim = 1)
        return out


    def set_hidden_device(self,device='gpu'):
        if device =='cpu':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
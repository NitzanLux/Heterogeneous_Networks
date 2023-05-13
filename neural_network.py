import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List,Iterable
import matplotlib.pyplot as plt
from cd_data_squre import CDataLoader
from sklearn import metrics
from tqdm import tqdm
from copy import deepcopy
from torch import Tensor
import torch.nn.functional as F
from utils import slurm_job
class SingleNeuron(nn.Module):
    def __init__(self, in_features: int, w_size: int, lr:float ):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.lr = lr
        self.weights=torch.normal(mean=torch.zeros((w_size,1)), std=torch.ones((w_size,1)))# self.w=np.random.normal(0,1,size=w_size)
        self.lr_result = lr

    def forward(self, input: Tensor) -> Tensor:
        output = self.linear(input)
        # with torch.no_grad():
        #     self.lr_result = F.relu6(input@self.weights+self.lr).mean()
        return output


class CustomLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, lr: [List[float],None]=None):
        super().__init__()
        assert lr is None or len(lr)==out_features,"features supposed to be as the number of lr"
        if lr is None:
            lr=[1e-5]*(out_features//2) +[1e-1]*(out_features-out_features//2)
        self.neuron_in_layer = nn.ModuleList(
            [SingleNeuron(in_features, out_features, _lr) for i, _lr in zip(range(out_features), lr)])
    def forward(self, input):
        out = torch.stack([torch.squeeze(i(input)) for i in self.neuron_in_layer], dim=1)
        return torch.relu(out)

class CustomNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes:List[int], output_size, lr_arr:[List[List[float]],float, None],homogeneous_lr=False,  entropy_dependent_lr=True):
        super().__init__()
        self.layers= nn.ModuleList()
        last_layer=input_size

        if lr_arr is not None:
            pass
            # assert (homogeneous_lr and isinstance(lr_arr, float)) or (all([len(lr) == h for lr,h in zip(lr_arr, hidden_sizes + [output_size])])), "number of lr should be congurent"
            # assert False
        for i in hidden_sizes:
            # self.layers.append(CustomLayer(last_layer, i,np.random.random((i,))))
            self.layers.append(CustomLayer(last_layer, i))
            last_layer=i
        self.lr = lr_arr if isinstance(lr_arr,float) else 0.
        self.lr_arr=None if isinstance(lr_arr,float) else lr_arr
        # self.layers.append(CustomLayer(last_layer, output_size,np.random.random((output_size,))))#todo fix
        self.layers.append(CustomLayer(last_layer, output_size))#todo fix
        self.softmax = nn.Softmax(dim=0)
        self.homogeneous_lr = homogeneous_lr
        self.entropy_dependent_lr = entropy_dependent_lr

    def forward(self, input):
        out = input
        for l in self.layers:
            out=l(out)
        return self.softmax(out)

    def generate_lr_params(self):
        data = []
        for l in self.layers:
            for m in l.neuron_in_layer:
                data.append(
                    {'params': m.parameters(), 'lr': m.lr_result if self.entropy_dependent_lr else m.lr})
        return data

    def get_optimizer(self):
        if self.homogeneous_lr:
            return optim.SGD(self.parameters(), lr=self.lr)
        return optim.SGD(self.generate_lr_params(), lr=self.lr)

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.apply(_init_weights)

    def pred(self,x):
        out=self(x)
        out_one_hot= torch.zeros_like(out)
        out_one_hot[torch.argmax(out_one_hot)]=1
        return out_one_hot
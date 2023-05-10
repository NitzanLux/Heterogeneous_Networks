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
class SingleNeuron(nn.Linear):
    def __init__(self, in_features: int, w_size: int, lr:float ):
        super().__init__(in_features, 1)
        self.lr = lr
        self.weights=torch.normal(mean=torch.zeros((w_size,1)), std=torch.ones((w_size,1)))# self.w=np.random.normal(0,1,size=w_size)
        self.lr_result = lr

    def forward(self, input: Tensor) -> Tensor:
        output = super().forward(input)
        # with torch.no_grad():
        #     self.lr_result = F.relu6(input@self.weights+self.lr).mean()
        return output


class CustomLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, lr: List[float]):
        super().__init__()
        assert len(lr)==out_features,"features supposed to be as the number of lr"
        self.neuron_in_layer = nn.ModuleList(
            [SingleNeuron(in_features, out_features, _lr) for i, _lr in zip(range(out_features), lr)])
    def forward(self, input):
        out = torch.squeeze(torch.stack([i(input) for i in self.neuron_in_layer], dim=1))
        return torch.relu(out)

class CustomNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes:Iterable[int], output_size, homogenuos_lr=False, entropy_dependent_lr=True):
        super().__init__()
        self.layers= nn.ModuleList()
        last_layer=input_size
        for i in hidden_sizes:
            self.layers.append(CustomLayer(last_layer, i,np.random.random((i,))))
            last_layer=i
        self.layers.append(CustomLayer(last_layer, output_size,np.random.random((output_size,))))#todo fix
        self.softmax = nn.Softmax(dim=0)
        self.homogeneous_lr = homogenuos_lr
        self.entropy_dependent_lr = entropy_dependent_lr

    def forward(self, input):
        out = input
        for l in self.layers:
            out=l(out)
        return self.softmax(out)

    def generate_lr_params(self):
        data = []
        for l in [self.layer1, self.layer2]:
            for m in l.neuron_in_layer:
                data.append(
                    {'params': m.classifier.parameters(), 'lr': m.lr_result if self.entropy_dependent_lr else m.lr})
        return data

    def get_optimizer(self):
        if self.homogeneous_lr:
            return optim.SGD(self.parameters(), lr=1e-2)
        return optim.SGD(self.generate_lr_params(), lr=1e-2)

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
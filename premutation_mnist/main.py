import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from cd_data_squre import CDataLoader
from sklearn import metrics
from tqdm import tqdm
from copy import deepcopy
from torch import Tensor
import torch.nn.functional as F
from utils import slurm_job
from neural_network import *

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainset = PermutedMNIST(train=True, permute_seed=42)
    testset = PermutedMNIST(train=False, permute_seed=42)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    # Display a sample from the dataset
    display_sample(trainset, 0)

    epochs = 10
    for epoch in range(epochs):
        train_loss = train(model, trainloader, device, optimizer, criterion)
        test_loss, accuracy = evaluate(model, testloader, device, criterion)
        print(f"Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()

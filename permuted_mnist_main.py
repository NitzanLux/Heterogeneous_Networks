import os

import numpy.random
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
import pickle
from neural_network import *
from premutation_mnist.permuted_mnist_dataset import PermutedMNIST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def custom_loss_function(pred, target):
    target = target.to(torch.float32)
    loss = nn.MSELoss()(pred, target) + nn.L1Loss()(pred,
                                                    target)  # Replace with the loss function suitable for your problem
    return loss


def train(model, data_loader, n_steps: [None, int] = None, n_epochs: [None, int] = None):
    assert (not n_steps and n_epochs) or (
            not n_epochs and n_steps), "n_steps and n_epochs should be choose(independently"

    convergence_arr_train = []
    model.train()
    counter = 0
    while True:
        for d_input, target in (data_loader):
            d_input = d_input.to(device)
            target = target.to(device)
            pred = model(d_input)
            optimizer = model.get_optimizer()
            target_ohv = F.one_hot(target, 10)
            optimizer.zero_grad()
            loss = custom_loss_function(pred, target_ohv)
            loss.backward()
            optimizer.step()
            if n_steps is not None:
                counter += 1
                if counter > n_steps:
                    return
        if n_epochs is not None:
            counter += 1
            if counter > n_epochs:
                return


def evaluation_score(model, x, y):
    x = x.to(device)
    y = y.to(device)
    y= F.one_hot(y, 10)
    return 1-torch.sum(torch.abs(model(x) - y)) / (2 * y.shape[0])


def test(model, data_loader, n_samples=4):
    accuracies = []
    model.eval()
    for i, xy in zip(range(n_samples), data_loader):
        x, y = xy
        accuracies.append(evaluation_score(model, x, y))
    return torch.mean(torch.stack(accuracies)).cpu().detach().numpy()


def train_residual(model, dataloaders: List[PermutedMNIST], batch_size: int, n_steps: [None, int] = None,
                   n_epochs: [None, int] = None, num_workers: int = 1, n_f_epochs: [None, int] = None,
                   n_f_steps: [None, int] = None):
    assert (not n_steps and n_epochs) or (
            not n_epochs and n_steps), "n_steps and n_epochs should be choose(independently)"
    print('start residual training...')
    for i, d in tqdm(zip([50] + [20] * (len(dataloaders) - 1), dataloaders)):
        train(model, d.get_dataloader(batch_size=batch_size, num_workers=num_workers, shuffle=True),
              n_epochs=n_epochs if i > 0 else n_f_epochs,
              n_steps=n_steps if i > 0 else n_f_steps)
        yield model


def evaluate_performance(model, dataloaders: List[PermutedMNIST], batch_size: int, n_samples: int = 5):
    accuracies = []
    for d in dataloaders:
        accuracies.append(test(model, d.get_dataloader(batch_size=batch_size), n_samples=n_samples))
    print(accuracies)
    return accuracies


def run_permuted_mnist_task(model, n_task: int, batch_size: int, n_steps: [None, int] = None,
                            n_epochs: [None, int] = None, num_workers: int = 1, n_f_epochs: [None, int] = None,
                            n_f_steps: [None, int] = None):
    if n_f_steps is None and n_f_epochs is None:
        n_f_steps = n_steps
        n_f_epochs = n_f_epochs
    dataloaders = [PermutedMNIST(train=True, permute=True if i > 0 else False) for i in range(n_task)]
    dataloaders_test = [PermutedMNIST(train=False, permute=i.permutation) for i in dataloaders]
    tr = train_residual(model, dataloaders, batch_size=batch_size, n_steps=n_steps, n_epochs=n_epochs,
                        num_workers=num_workers, n_f_epochs=n_f_epochs, n_f_steps=n_f_steps)
    output_matrix = np.zeros((n_task, n_task))
    print("start evaluation")
    for i, m in tqdm(enumerate(tr)):
        output_matrix[i, :i+1] = evaluate_performance(m, dataloaders_test[:i+1], 10000, 5)
    return output_matrix

def save_matrix_and_params(seed_number:int,entropy_dependent_lr=False,homogeneous_lr=True,tag='', n_task: int=10, batch_size: int=15, n_steps: [None, int] = None,
                            n_epochs: [None, int] = None, num_workers: int = 1, model_hidden_sizes=(24 * 24, 10 * 10, 5 * 5),n_f_epochs: [None, int] = None,
                            n_f_steps: [None, int] = None):
    os.makedirs('data',exist_ok=True)
    os.makedirs(os.path.join('data',tag),exist_ok=True)
    dir_name=f'd_{len(os.listdir(os.path.join("data",tag)))}_{np.random.randint(0,10000)}'
    dest_path = os.path.join('data',tag,dir_name)
    os.makedirs(dest_path)

    np.random.seed(seed_number)
    f_m = CustomNetwork(28 * 28,model_hidden_sizes , 10)
    f_m.init_weights()
    f_m.entropy_dependent_lr = entropy_dependent_lr
    f_m.homogeneous_lr = homogeneous_lr
    f_m.to(device)
    p = run_permuted_mnist_task(f_m, n_task, batch_size, n_steps=n_steps,n_epochs=n_epochs, n_f_steps=n_f_steps,n_f_epochs=n_f_epochs,num_workers=num_workers)
    with open(os.path.join(dest_path,'performance_mat.p'), 'wb') as f:
        pickle.dump(p, f)
    data_dict = dict(seed_number=seed_number,entropy_dependent_lr=entropy_dependent_lr,homogeneous_lr=homogeneous_lr,tag=tag,n_task=n_task,
                     batch_size=batch_size,n_steps=n_steps,n_epochs=n_epochs,num_workers=num_workers,model_hidden_sizes=model_hidden_sizes,n_f_steps=n_f_steps,n_f_epochs=n_f_epochs)
    with open(os.path.join(dest_path,f'config_dict.pickle'), 'wb') as f:
        pickle.dump(data_dict, f)

    return p
import random
if __name__ == '__main__':
    for i in range(10):
        seed_number=random.randint(0,100000)

        args = dict(seed_number=seed_number,n_task=10,tag="test_basic_network",n_epochs=20,n_f_epochs=50,entropy_dependent_lr=False,homogeneous_lr=True,model_hidden_sizes=[20*20,10*10,10*10,5*5])
        s = slurm_job.SlurmJobFactory('cluster_logs')
        s.send_job_for_function(f'{i}_first_validation','permuted_mnist_main','save_matrix_and_params',args,run_on_GPU=i<5)
        print(i)
    # print(p)

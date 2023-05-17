import os

import numpy.random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Any, Dict
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
import wandb

WANDB_API_KEY = "2725e59f8f4484605300fdf4da4c270ff0fe44a3"

WANDB_PROJECT_NAME = "Heterogeneous-learning-rate"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def custom_loss_function(pred, target):
    target = target.to(torch.float32)
    loss = nn.CrossEntropyLoss()(pred, target)  # + nn.L1Loss()(pred,target)
    return loss


def init_wandb(config, model):
    wandb.login()
    wandb.init(project=(WANDB_PROJECT_NAME), config=config, entity='nilu', allow_val_change=True,
               id=config['dir_name'])
    # config = wandb.config
    # load_and_train(config)
    wandb.watch(model, log_freq=100)


def train(model, data_loader, test_dataloader, n_steps: [None, int] = None, n_epochs: [None, int] = None,
          train_index=0):
    assert (not n_steps and n_epochs) or (
            not n_epochs and n_steps), "n_steps and n_epochs should be choose(independently"

    # convergence_arr_train = []
    model.train()
    counter = 0
    optimizer = model.get_optimizer()
    test_iterator = iter(test_dataloader)
    while True:
        for d_input, target in (data_loader):
            d_input = d_input.to(device)
            target = target.to(device)
            pred = model(d_input)

            target_ohv = F.one_hot(target, 10)
            optimizer.zero_grad()
            loss = custom_loss_function(pred, target_ohv)
            loss.backward()
            optimizer.step()
            model.eval()
            wandb.log({'training_loss_%d' % train_index: loss.cpu().detach().numpy(),
                       'training_accuracy_%d' % train_index: evaluation_score(pred, target)})
            if counter % 20 == 0:
                try:
                    d_input, target = test_iterator.__next__()
                except StopIteration as e:
                    test_iterator = iter(test_dataloader)
                    d_input, target = test_iterator.__next__()

                d_input = d_input.to(device)
                target = target.to(device)
                wandb.log({'testing_accuracy_%d' % train_index: evaluation_score(model(d_input), target)}, commit=False)
            model.train()

            if n_steps is not None:
                counter += 1
                if counter > n_steps:
                    return
        if n_epochs is not None:
            counter += 1
            if counter > n_epochs:
                return


def evaluation_score(pred_y, y):
    pred_y = torch.argmax(pred_y, 1)
    return torch.sum(pred_y == y) / y.shape[0]


def test(model, data_loader, n_samples=4):
    accuracies = []
    model.eval()
    for i, xy in zip(range(n_samples), data_loader):
        x, y = xy
        accuracies.append(evaluation_score(model(x.to(device)), y.to(device)))
    return torch.mean(torch.stack(accuracies)).cpu().detach().numpy()


def simple_train_and_evaluate(model, dataloader_train, data_loader_test, n_steps):
    train(model, dataloader_train, data_loader_test, n_steps=n_steps)
    test(model, data_loader_test)


def train_residual(model, path, dataloaders: List[PermutedMNIST], batch_size: int, n_steps: [None, int] = None,
                   n_epochs: [None, int] = None, num_workers: int = 1, n_f_epochs: [None, int] = None,
                   n_f_steps: [None, int] = None, checkpoint=0):
    assert (not n_steps and n_epochs) or (
            not n_epochs and n_steps), "n_steps and n_epochs should be choose(independently)"
    print('start residual training...')
    counter = 0
    for i, d in tqdm(zip([50] + [20] * (len(dataloaders) - 1), dataloaders[checkpoint:])):
        train(model, d.get_dataloader(batch_size=batch_size, num_workers=num_workers, shuffle=True),
              PermutedMNIST(train=False, permute=d.permutation).get_dataloader(shuffle=True, batch_size=500)
              , n_epochs=n_epochs if i > 0 else n_f_epochs,
              n_steps=n_steps if i > 0 else n_f_steps, train_index=counter)
        counter += 1
        model.save(path)

        yield model


def evaluate_performance(model, dataloaders: List[PermutedMNIST], batch_size: int, n_samples: int = 5):
    accuracies = []
    for d in dataloaders:
        accuracies.append(test(model, d.get_dataloader(batch_size=batch_size), n_samples=n_samples))
        wandb.log({'accuracies_residual': accuracies[-1]}, commit=False)
    print(accuracies)
    return accuracies


def run_permuted_mnist_task(model, path, dataloaders, n_task: int, batch_size: int, n_steps: [None, int] = None,
                            n_epochs: [None, int] = None, num_workers: int = 1, n_f_epochs: [None, int] = None,
                            n_f_steps: [None, int] = None, checkpoint=0):
    if n_f_steps is None and n_f_epochs is None:
        n_f_steps = n_steps
        n_f_epochs = n_f_epochs
    dataloaders_test = [PermutedMNIST(train=False, permute=i.permutation) for i in dataloaders]
    tr = train_residual(model, path, dataloaders, batch_size, n_steps=n_steps, n_epochs=n_epochs,
                        num_workers=num_workers, n_f_epochs=n_f_epochs, n_f_steps=n_f_steps, checkpoint=checkpoint)
    if os.path.exists(os.path.join(path, 'performance_mat.p')):
        with open(os.path.join(path, 'performance_mat.p'), 'rb') as f:
            output_matrix = pickle.load(f)
    else:
        output_matrix = np.zeros((n_task, n_task))
    print("start evaluation")
    for i, m in tqdm(enumerate(tr)):
        i = i + checkpoint
        output_matrix[i, :i + 1] = evaluate_performance(m, dataloaders_test[:i + 1], 10000, 5)
        with open(os.path.join(path, 'checkpoint_step.pickle'), 'wb') as f:
            pickle.dump(i, f)
        with open(os.path.join(path, 'performance_mat.p'), 'wb') as f:
            pickle.dump(output_matrix, f)
    return output_matrix


def build_model(model_hidden_sizes, homogeneous_lr, entropy_dependent_lr, lr, input_size=28 * 28, number_of_classes=10):
    f_m = CustomNetwork(input_size, model_hidden_sizes, number_of_classes, lr_arr=lr)
    f_m.init_weights()
    f_m.entropy_dependent_lr = entropy_dependent_lr
    f_m.homogeneous_lr = homogeneous_lr
    f_m.to(device)
    return f_m


def save_matrix_and_params(seed_number: int, torch_seed_number:int,entropy_dependent_lr=False, homogeneous_lr=True, tag='', n_task: int = 10,
                           batch_size: int = 15, n_steps: [None, int] = None,
                           n_epochs: [None, int] = None, num_workers: int = 1,
                           model_hidden_sizes=(24 * 24, 10 * 10, 5 * 5), n_f_epochs: [None, int] = None,
                           n_f_steps: [None, int] = None, lr=None, checkpoint=0,input_size=28 * 28, number_of_classes=10,**kwargs):
    os.makedirs(os.path.join('data', 'mnist_task_data'), exist_ok=True)
    os.makedirs(os.path.join('data', tag), exist_ok=True)
    dir_name = f'd_{len(os.listdir(os.path.join("data", tag)))}_{np.random.randint(0, 10000)}'
    dest_path = os.path.join('data', tag, dir_name)
    os.makedirs(dest_path, exist_ok=True)
    data_dict = locals()
    data_dict['dest_path'] = dest_path
    np.random.seed(seed_number)
    torch.manual_seed(torch_seed_number)
    f_m = build_model(model_hidden_sizes, homogeneous_lr, entropy_dependent_lr, lr=data_dict['lr'],input_size=28 * 28, number_of_classes=10)
    if os.path.exists(os.path.join(dest_path, 'weights.pt')):
        f_m.load_state_dict(torch.load(os.path.join(dest_path, 'weights.pt')))
    init_wandb(data_dict, f_m)
    if os.path.exists(os.path.join(dest_path, 'permutation_array.pickle')):
        with open(os.path.join(dest_path, 'permutation_array.pickle'), 'rb') as f:
            permutations = pickle.load(f)
        dataloaders = [PermutedMNIST(train=True, permute=True if i > 0 else i) for i in range(permutations)]
    else:
        dataloaders = [PermutedMNIST(train=True, permute=True if i > 0 else False) for i in range(n_task)]
        permutations = [i.permutation for i in dataloaders]
        with open(os.path.join(dest_path, f'permutation_array.pickle'), 'wb') as f:
            pickle.dump(permutations, f)
    if not os.path.exists(os.path.join(dest_path, f'config_dict.pickle')):
        with open(os.path.join(dest_path, f'config_dict.pickle'), 'wb') as f:
            pickle.dump(data_dict, f)

    p = run_permuted_mnist_task(f_m, dest_path, dataloaders, n_task, batch_size,
                                n_steps=n_steps,
                                n_epochs=n_epochs, n_f_steps=n_f_steps,
                                n_f_epochs=n_f_epochs, num_workers=num_workers, checkpoint=checkpoint)
    return p


import random
import platform

if __name__ == '__main__':
    get_args = lambda: dict(seed_number=42,torch_seed_number=28, n_task=6, tag="swip_multiple_parameters", n_epochs=10,
                            n_f_epochs=30,
                            entropy_dependent_lr=False, homogeneous_lr=True, lr=None,
                            model_hidden_sizes=[20 * 20, 10 * 10, 10 * 5],input_size=28 * 28, number_of_classes=10)
    if platform.system() == 'Windows':
        # save_matrix_and_params(**get_args())
        args = get_args()
        args['dir_name'] = args['tag'] + '_' + str(args['seed_number'])
        avarages = np.exp(-np.arange(15))
        ratios = np.arange(5,55,5)/100.
        ab = np.random.choice(avarages, 2, replace=False)
        a,b=np.min(ab) ,np.max(ab)
        r = np.random.choice(ratios, 1, replace=False)
        lr_arr=[]
        args['homogeneous_lr'] = True
        for i in args['model_hidden_sizes'] + [args['number_of_classes']]:
            lr_arr.append(([b] * int(i * r) + [a] * (i - int(i * r))))
        # args['lr'] = lr_arr
        args['lr'] = 1e-3
        m = build_model(get_args()['model_hidden_sizes'], False, False, lr = args['lr'])
        init_wandb(args, m)

        simple_train_and_evaluate(m, PermutedMNIST(train=True).get_dataloader(100),
                                  PermutedMNIST(train=False).get_dataloader(1000), 1000)
    else:
        s = slurm_job.SlurmJobFactory('cluster_logs')
        avarages = np.exp(-np.arange(15))
        ratios = np.arange(5,55,5)/100.
        for j in range(1):
            ab = np.random.choice(avarages, 2, replace=False)
            a,b=np.min(ab) ,np.max(ab)
            r = np.random.choice(ratios, 1, replace=False)
            total_lr = r*b+(1.-r)*a
            args = get_args()

            args['a']=a
            args['b']=b
            args['r']=r
            args['total_lr']=total_lr

            print(total_lr)
            #homogeneous
            args['homogeneous_lr'] = False
            lr_arr=[]
            for i in args['model_hidden_sizes']+[args['number_of_classes']]:
                lr_arr.append(([b]*int(i*r)+[a]*(i-int(i*r))))
            args['lr']=lr_arr
            s.send_job_for_function(f'{j}_first_validation_hetro', 'permuted_mnist_main', 'save_matrix_and_params',
                                    args, run_on_GPU=j > 3)

            #control
            args['homogeneous_lr'] = True
            args['lr'] = total_lr

            s.send_job_for_function(f'{j}_first_validation_homogenous', 'permuted_mnist_main', 'save_matrix_and_params',
                                    args, run_on_GPU=j > 3)
            print(j)
        # print(p)

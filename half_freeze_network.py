import copy

from neural_network import *
output_layer=2
hidden_size=[8]
lr_arr=[([0.,1e-3]*(i//2))for i in hidden_size]
# lr_arr=[([0.]*(i//2)+[0]*(i//2+ int(i%2==1)))for i in hidden_size]
lr_arr.append([0.]*output_layer)
lr_arr_negative = [([1e-3,0]*(i//2))for i in hidden_size]
lr_arr_negative.append([0.]*output_layer)




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
import wandb
import wandb
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WANDB_API_KEY = "2725e59f8f4484605300fdf4da4c270ff0fe44a3"

WANDB_PROJECT_NAME = "Heterogeneous-Half-Freeze"
def init_wandb(config, model):

    wandb.login()
    wandb.init(project=(WANDB_PROJECT_NAME), config=config, entity='nilu', allow_val_change=True)
    # config = wandb.config
    # load_and_train(config)
    wandb.watch(model, log_freq=10)
def custom_loss_function(pred, target):
    loss = nn.CrossEntropyLoss()(pred, target)  # Replace with the loss function suitable for your problem
    return loss


def calculate_lr_scale(relevance):
    # Implement your function that maps input relevance to learning rate scale factors
    # This is a simple example; you should replace it with a more appropriate function
    return 1 / (1 + torch.exp(-relevance))
def calculte_accuracy(y_pred,y):
    if len(y_pred.shape)>2:
        y_pred = np.argmax(y_pred, axis=1)
    if len(y.shape)>2:
        y = np.argmax(y, axis=1)

    # Then we count how many of these are equal and divide by the number of total predictions
    accuracy = np.mean(y_pred == y)

    return accuracy

def train(model, data_loader, test_data,tag, train_to_thresh=False):
    convergence_arr_train = []
    convergence_arr_test = []

    model.train()
    test_data_iter = test_data.__iter__()
    # optimizer = optim.SGD(model.generate_lr_params(), lr=1e-2)

    counter = 0
    for d_input, target in tqdm(data_loader):
        d_input = d_input.to(device)
        target = target.to(device)
        pred = model(d_input)
        optimizer = model.get_optimizer()
        optimizer.zero_grad()
        loss = custom_loss_function(pred, torch.nn.functional.one_hot(target.long()).float())
        # print('train: ',loss.item())
        convergence_arr_train.append(loss.item())
        loss.backward()
        optimizer.step()
        target = target.cpu().detach().numpy()
        prd_m =torch.argmax(pred,dim=1)
        prd_m = prd_m.cpu().detach().numpy()
        accuracy = calculte_accuracy(prd_m,target)
        wandb.log({'accuracies_t_train_'+tag:accuracy,'loss_t_train_'+tag:loss,'step':counter}, commit=True)

        x, y = test_data_iter.__next__()
        model = model.to(device)
        x = x.to(device)
        y = y.to(device)

        model.eval()
        prd_m = (model(x))
        with torch.no_grad():
            test_loss = custom_loss_function(prd_m, torch.nn.functional.one_hot(y.long()).float()).item()
        # print('test:  ',train_loss)
        convergence_arr_test.append(test_loss)

        model.train()
        y = y.cpu().detach().numpy()
        prd_m =torch.argmax(prd_m,dim=1)
        prd_m = prd_m.cpu().detach().numpy()
        accuracy = calculte_accuracy(prd_m,y)
        wandb.log({'accuracies_t_test_'+tag:accuracy,'loss_t_test_'+tag:test_loss,'step':counter}, commit=True)
        counter += 1
        if train_to_thresh:
            if accuracy == 1:
                break
    return convergence_arr_train, convergence_arr_test, counter





# model = CustomNetwork(input_size, hidden_size, output_size)


# Create your data_loader with input, target, and relevance values
def test(model,tag='', first_task=True):
    cd = CDataLoader(100000, mask_d=0.4, n_batch=1)
    if not first_task:
        cd.switch()
    model.eval()
    x, y = cd.__iter__().__next__()
    x = x.to(device)
    y = y.to(device)
    pred = (model(x))
    print('task 1:' if first_task else 'task 2:')

    pred= torch.argmax(pred, dim=1)
    pred = pred.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    accuracy=calculte_accuracy(pred,y)
    wandb.log({f'accuracies_train_task_{1 if first_task else 2}' + tag: accuracy}, commit=True
              )
    print('accuracy: ', accuracy, '%')
    return accuracy

def evaluate_model(model, model_id, train_to_thresh, mask_d, disperssion, length, batch_size, index, data_dict,input_size,hidden_size,output_size,tag):
    if train_to_thresh:
        length = 1000000000000000
    model.init_weights()
    conf_dict =copy.deepcopy(locals())
    del conf_dict['model']
    # del conf_dict['data_dict']
    init_wandb(data_dict,model)
    train_data = CDataLoader(batch_size, mask_d=mask_d, n_batch=length, disperssion=disperssion)
    test_data = CDataLoader(1000, mask_d=mask_d, n_batch=length, disperssion=disperssion)
    _, _, steps_1 = train(model, train_data, test_data,"1_phase", train_to_thresh)
    print('step 1:  ', steps_1)
    if not model.homogeneous_lr: model.update_lr(lr_arr)
    accuracy_1 = test(model,'')
    train_data.switch()
    test_data.switch()
    if not model.homogeneous_lr:model.update_lr(lr_arr_negative)
    _, _, steps_2 = train(model, train_data, test_data,"2_phase", train_to_thresh)
    print('step 2:  ', steps_2)

    accuracy_2 = test(model, '',False)
    print("--------------------first task")
    accuracy_forget = test(model,'_forget')

    data_dict['steps_1'].append(steps_1)
    data_dict['steps_2'].append(steps_2)
    # data_dict['auc_1'].append(auc_1)
    data_dict['accuracy_1'].append(accuracy_1)
    # data_dict['auc_2'].append(auc_2)
    data_dict['accuracy_2'].append(accuracy_2)
    # data_dict['auc_forget'].append(auc_forget)
    data_dict['accuracy_forget'].append(accuracy_forget)
    data_dict['id'].append(model_id)
    data_dict['index'].append(index)
    data_dict['hidden_size'].append(hidden_size)
    data_dict['batch_size'].append(batch_size)
    data_dict['tag'].append(tag)
    wandb.finish()
    return data_dict

def evaluate(data_dict,input_size,hidden_size,output_size, length=20000, batch_size=50, train_to_thresh=False, mask_d=0.4, disperssion=0.4, index=0,tag=''):
    print('************************************************')
    print('model homogeneuos')
    print('************************************************')
    model = CustomNetwork(input_size, hidden_size, output_size, homogeneous_lr=True,lr_arr=1e-3)
    model = model.to(device)


    model_b = deepcopy(model)
    model_c = deepcopy(model)

    data_dict = evaluate_model(model, 'original', train_to_thresh, mask_d, disperssion, length, batch_size, index, data_dict,input_size,hidden_size,output_size,tag)

    print('\n************************************************')
    print('model heterogeneous')
    print('************************************************')
    model_b.homogenuos_lr = False
    model_b.entropy_dependent_lr = False

    data_dict = evaluate_model(model_b, 'heterogeneous_constant', train_to_thresh, mask_d, disperssion, length, batch_size, index, data_dict,input_size,hidden_size,output_size,tag)

    # print('************************************************')
    # print('model heterogeneous custom_response with bounded relu')
    # print('************************************************')
    # model_c.homogenuos_lr = False
    # model_c.entropy_dependent_lr = True
    #
    # data_dict = evaluate_model(model_c, 'heterogeneous_dynamic_weights_relu6', train_to_thresh, mask_d, disperssion, length, batch_size, index, data_dict,input_size,hidden_size,output_size,tag)

    return data_dict


# cd = CDataLoader(500, mask_d=0.5, disperssion=10, n_batch=20000, normal_sampling=False)
# cd.plot_data(*cd.generate_first_rule_data(),500)
# cd.plot_data(*cd.generate_second_rule_data(),500)
# cd.plot_data(n=500)
# cd.plot_data()
tag='custome immidiate response'
data_dict = dict(steps_1=[], steps_2=[], auc_1=[], auc_2=[], auc_forget=[], index=[], id=[],
                     accuracy_1=[], accuracy_2=[], accuracy_forget=[],hidden_size=[],batch_size=[],tag=[])
# def evaluate_on_cluster_debug():

    # evaluate(data_dict,2,hidden_size,output_layer, batch_size=100, train_to_thresh=False, length=20000,mask_d=0.5, disperssion=10,index=0,tag=tag)
def evaluate_on_cluster(data_folder,simulation_id,number_of_sims):
    data_dict = dict(steps_1=[], steps_2=[], auc_1=[], auc_2=[], auc_forget=[], index=[], id=[],
                     accuracy_1=[], accuracy_2=[], accuracy_forget=[],hidden_size=[],batch_size=[],tag=[])
    for i in range(number_of_sims):
        evaluate(data_dict,2,hidden_size,output_layer, batch_size=10, train_to_thresh=True, mask_d=0.5, disperssion=10,index=i,tag=tag)
    try:
        import cPickle as pickle
    except ImportError:  # Python 3.x
        import pickle
    cur_path=os.path.join('data',data_folder)
    os.makedirs(cur_path,exist_ok=True)
    with open(os.path.join(cur_path,f'data_{tag}_{simulation_id}.p'), 'wb') as fp:
        pickle.dump(data_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
if __name__ == '__main__':
    s = slurm_job.SlurmJobFactory('cluster_logs')
    # s = slurm_job.SlurmJobFactory('cluster_logs')
    j = np.random.randint(0,1e+9)
    s.send_job_for_function(f'{j}_debug_freeze', 'half_freeze_network', 'evaluate_on_cluster',
                            [], run_on_GPU=True)
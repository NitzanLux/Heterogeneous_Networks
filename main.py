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

def custom_loss_function(pred, target):
    loss = nn.MSELoss()(pred, target)  # Replace with the loss function suitable for your problem
    return loss


def calculate_lr_scale(relevance):
    # Implement your function that maps input relevance to learning rate scale factors
    # This is a simple example; you should replace it with a more appropriate function
    return 1 / (1 + torch.exp(-relevance))


def train(model, data_loader, test_data, train_to_thresh=False):
    convergence_arr_train = []
    convergence_arr_test = []

    model.train()
    test_data_iter = test_data.__iter__()
    # optimizer = optim.SGD(model.generate_lr_params(), lr=1e-2)

    counter = 0
    for d_input, target in tqdm(data_loader):

        pred = model(d_input)
        optimizer = model.get_optimizer()
        optimizer.zero_grad()
        loss = custom_loss_function(pred, target)
        # print('train: ',loss.item())
        convergence_arr_train.append(loss.item())
        loss.backward()
        optimizer.step()
        x, y = test_data_iter.__next__()
        model.eval()
        prd_m = (model(x))
        test_loss = custom_loss_function(prd_m, y).item()
        # print('test:  ',train_loss)
        convergence_arr_test.append(test_loss)
        counter += 1
        if train_to_thresh:
            y = y.detach().numpy()
            prd_m = prd_m.detach().numpy()

            fpr, tpr, thresholds = metrics.roc_curve(y.astype(bool), prd_m, pos_label=1)
            # auc = metrics.auc(fpr, tpr)
            t = thresholds[np.argmax(tpr - fpr)]
            accuracy = np.linalg.norm((prd_m < t) - y, 1) / y.size
            if accuracy == 1:
                break
        model.train()
    return convergence_arr_train, convergence_arr_test, counter





# model = CustomNetwork(input_size, hidden_size, output_size)


# Create your data_loader with input, target, and relevance values
def test(model, first_task=True):
    cd = CDataLoader(100000, mask_d=0.4, n_batch=1)
    if not first_task:
        cd.switch()
    model.eval()
    x, y = cd.__iter__().__next__()
    pred = (model(x))
    print('task 1:' if first_task else 'task 2:')
    # print('test:  ', custom_loss_function(pred, y).item())
    # b_pred=(pred>0.5)
    pred = pred.detach().numpy()
    y = y.detach().numpy()
    # print('test succsess:  ', np.sum((pred-y)))
    fpr, tpr, thresholds = metrics.roc_curve(y.astype(bool), pred, pos_label=1)
    t = thresholds[np.argmax(tpr - fpr)]
    auc = metrics.auc(fpr, tpr)
    print('\nauc: ', auc)
    accuracy =  np.linalg.norm((pred < t) - y, 1) / y.size
    print('accuracy: ', accuracy, '%')
    return accuracy, auc



def evaluate_model(model, model_id, train_to_thresh, mask_d, disperssion, length, batch_size, index, data_dict,input_size,hidden_size,output_size,tag):
    if train_to_thresh:
        length = 1000000000000000
    train_data = CDataLoader(batch_size, mask_d=mask_d, n_batch=length, disperssion=disperssion)
    test_data = CDataLoader(1000, mask_d=mask_d, n_batch=length, disperssion=disperssion)
    _, _, steps_1 = train(model, train_data, test_data, train_to_thresh)
    print('step 1:  ', steps_1)

    accuracy_1, auc_1 = test(model)
    train_data.switch()
    test_data.switch()
    _, _, steps_2 = train(model, train_data, test_data, train_to_thresh)
    print('step 2:  ', steps_2)

    accuracy_2, auc_2 = test(model, False)
    print("--------------------first task")
    accuracy_forget, auc_forget = test(model)

    data_dict['steps_1'].append(steps_1)
    data_dict['steps_2'].append(steps_2)
    data_dict['auc_1'].append(auc_1)
    data_dict['accuracy_1'].append(accuracy_1)
    data_dict['auc_2'].append(auc_2)
    data_dict['accuracy_2'].append(accuracy_2)
    data_dict['auc_forget'].append(auc_forget)
    data_dict['accuracy_forget'].append(accuracy_forget)
    data_dict['id'].append(model_id)
    data_dict['index'].append(index)
    data_dict['hidden_size'].append(hidden_size)
    data_dict['batch_size'].append(batch_size)
    data_dict['tag'].append(tag)

    return data_dict

def evaluate(data_dict,input_size,hidden_size,output_size, length=20000, batch_size=50, train_to_thresh=False, mask_d=0.4, disperssion=0.4, index=0,tag=''):
    print('************************************************')
    print('model homogeneuos')
    print('************************************************')
    model = CustomNetwork(input_size, hidden_size, output_size, homogenuos_lr=True)
    model.init_weights()
    model_b = deepcopy(model)
    model_c = deepcopy(model)

    data_dict = evaluate_model(model, 'original', train_to_thresh, mask_d, disperssion, length, batch_size, index, data_dict,input_size,hidden_size,output_size,tag)

    print('\n************************************************')
    print('model heterogeneous')
    print('************************************************')
    model_b.homogenuos_lr = False
    model_b.entropy_dependent_lr = False

    data_dict = evaluate_model(model_b, 'heterogeneous_constant', train_to_thresh, mask_d, disperssion, length, batch_size, index, data_dict,input_size,hidden_size,output_size,tag)

    print('************************************************')
    print('model heterogeneous custom_response with bounded relu')
    print('************************************************')
    model_c.homogenuos_lr = False
    model_c.entropy_dependent_lr = True

    data_dict = evaluate_model(model_c, 'heterogeneous_dynamic_weights_relu6', train_to_thresh, mask_d, disperssion, length, batch_size, index, data_dict,input_size,hidden_size,output_size,tag)

    return data_dict

input_size = 2
hidden_size = 8
output_size = 1


cd = CDataLoader(500, mask_d=0.5, disperssion=10, n_batch=20000, normal_sampling=False)
cd.plot_data(*cd.generate_first_rule_data(),500)
cd.plot_data(*cd.generate_second_rule_data(),500)
cd.plot_data(n=500)
cd.plot_data()
tag='custome immidiate response'

def evaluate_on_cluster(data_folder,simulation_id,number_of_sims):
    data_dict = dict(steps_1=[], steps_2=[], auc_1=[], auc_2=[], auc_forget=[], index=[], id=[],
                     accuracy_1=[], accuracy_2=[], accuracy_forget=[],hidden_size=[],batch_size=[],tag=[])
    for i in range(number_of_sims):
        evaluate(data_dict,input_size,hidden_size,output_size, batch_size=10, train_to_thresh=True, mask_d=0.5, disperssion=10,index=i,tag=tag)
    try:
        import cPickle as pickle
    except ImportError:  # Python 3.x
        import pickle
    cur_path=os.path.join('data',data_folder)
    os.makedirs(cur_path,exist_ok=True)
    with open(os.path.join(cur_path,f'data_{tag}_{simulation_id}.p'), 'wb') as fp:
        pickle.dump(data_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
# if __name__ == '__main__':
#     simulation_id=np.random.randint(0,10000)
#     for i in range(100):
#         cur_id = simulation_id+i
#         slurm_job.SlurmJobFactory('cluster_logs').send_job_for_function('heterogeneous_ann_%d'%cur_id,'main','evaluate_on_cluster',[f'data_{simulation_id}',cur_id,1])

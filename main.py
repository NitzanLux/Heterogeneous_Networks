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


class SingleNeuron(nn.Linear):
    def __init__(self, in_features: int, w_size: int, lr: None | float = None):
        super().__init__(in_features, 1)
        if lr:
            self.lr = lr
        else:
            self.lr = np.random.random(1) * 0.4
        # self.w=np.random.normal(0,1,size=w_size)
        self.lr_result = lr

    def forward(self, input: Tensor) -> Tensor:
        output = super().forward(input)
        with torch.no_grad():
            self.lr_result = self.lr  # * output.mean()
        return output


class CustomLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, w_size: int, lr: List[float] | None = None):
        super().__init__()
        if not lr:
            # lr = np.power(np.random.normal(loc=1e-2, scale=5, size=(out_features,)), 2)
            lr = np.random.exponential(scale=1e-2, size=(out_features,))
            # lr = -np.ones((out_features,))
            # lr = np.exp(-np.arange(out_features))
            # lr[lr<0.5]=np.exp(lr[lr<0.5])
            # lr[lr0.5]=np.exp(lr[lr<0.5])
        self.neuron_in_layer = nn.ModuleList(
            [SingleNeuron(in_features, w_size, _lr) for i, _lr in zip(range(out_features), lr)])

    def forward(self, input):
        return torch.squeeze(torch.stack([i(input) for i in self.neuron_in_layer], dim=1))


class CustomNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, homogenuos_lr=False, entropy_dependent_lr=True):
        super().__init__()
        self.layer1 = CustomLayer(input_size, hidden_size, w_size=input_size)
        self.layer2 = CustomLayer(hidden_size, output_size, w_size=input_size)
        self.sigmoid = nn.Sigmoid()
        self.homogeneous_lr = homogenuos_lr
        self.entropy_dependent_lr = entropy_dependent_lr

    def forward(self, input):
        x = torch.relu(self.layer1(input))
        x = self.layer2(x)
        return self.sigmoid(x)

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
        return optim.SGD(self.generate_lr_params(), lr=0)

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.apply(_init_weights)


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
        optimizer = model.get_optimizer()
        optimizer.zero_grad()
        pred = model(d_input)
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
            auc = metrics.auc(fpr, tpr)
            if auc > 0.99999:
                break
        model.train()
    return convergence_arr_train, convergence_arr_test, counter


input_size = 2
hidden_size = 10
output_size = 1


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
    print('accuracy: ', np.linalg.norm((pred < t) - y, 1) / y.size, '%')
    return np.linalg.norm((pred < t) - y, 1) / y.size, auc


def evaluate(data_dict, length=20000, batch_size=100, train_to_thresh=False, mask_d=0.4, disperssion=0.4, index=0):
    print('************************************************')
    print('model homogeneuos')
    print('************************************************')
    model = CustomNetwork(input_size, hidden_size, output_size, homogenuos_lr=True)
    model.init_weights()
    model_b = deepcopy(model)
    model_c = deepcopy(model)
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
    data_dict['id'].append('original')
    data_dict['index'].append(index)

    print('\n************************************************')

    print('model heterogeneous')
    print('************************************************')
    model_b.homogenuos_lr = False
    model_b.entropy_dependent_lr = False

    # model.init_weights()

    train_data = CDataLoader(batch_size, mask_d=mask_d, n_batch=length, disperssion=disperssion)
    test_data = CDataLoader(1000, mask_d=mask_d, n_batch=length, disperssion=disperssion)
    _, _, steps_1 = train(model_b, train_data, test_data, train_to_thresh)
    print('step 1:  ', steps_1)

    accuracy_1, auc_1 = test(model_b)
    train_data.switch()
    test_data.switch()
    _, _, steps_2 = train(model_b, train_data, test_data, train_to_thresh)
    print('step 2:  ', steps_2)

    accuracy_2, auc_2 = test(model_b, False)
    print("--------------------first task")
    accuracy_forget, auc_forget = test(model_c)

    data_dict['steps_1'].append(steps_1)
    data_dict['steps_2'].append(steps_2)
    data_dict['auc_1'].append(auc_1)
    data_dict['accuracy_1'].append(accuracy_1)
    data_dict['auc_2'].append(auc_2)
    data_dict['accuracy_2'].append(accuracy_2)
    data_dict['auc_forget'].append(auc_forget)
    data_dict['accuracy_forget'].append(accuracy_forget)
    data_dict['id'].append('heterogeneous_constant')
    data_dict['index'].append(index)
    print('************************************************')
    print('model heterogeneous custom_response')
    print('************************************************')
    model_c.homogenuos_lr = False
    model_c.entropy_dependent_lr = True
    # model.init_weights()

    train_data = CDataLoader(batch_size, mask_d=mask_d, n_batch=length, disperssion=disperssion)
    test_data = CDataLoader(1000, mask_d=mask_d, n_batch=length, disperssion=disperssion)
    _, _, steps_1 = train(model_c, train_data, test_data, train_to_thresh)
    print('step 1:  ', steps_1)

    accuracy_1, auc_1 = test(model_c)
    train_data.switch()
    test_data.switch()
    _, _, steps_2 = train(model_c, train_data, test_data, train_to_thresh)
    print('step 2:  ', steps_2)

    accuracy_2, auc_2 = test(model_c, False)
    print("--------------------first task")
    accuracy_forget, auc_forget= test(model_c)

    data_dict['steps_1'].append(steps_1)
    data_dict['steps_2'].append(steps_2)
    data_dict['auc_1'].append(auc_1)
    data_dict['accuracy_1'].append(accuracy_1)
    data_dict['auc_2'].append(auc_2)
    data_dict['accuracy_2'].append(accuracy_2)
    data_dict['auc_forget'].append(auc_forget)
    data_dict['accuracy_forget'].append(accuracy_forget)
    data_dict['id'].append('heterogeneous_dynamic_weights')
    data_dict['index'].append(index)


cd = CDataLoader(500, mask_d=0.5, disperssion=10, n_batch=2000, normal_sampling=False)
# cd.plot_data(*cd.generate_first_rule_data(),500)
# cd.plot_data(*cd.generate_second_rule_data(),500)
# cd.plot_data(n=500)
# cd.plot_data()
data_dict = dict(steps_1=[], steps_2=[], auc_1=[], auc_2=[], auc_forget=[], index=[], id=[], condition=[],
                 accuracy_1=[], accuracy_2=[], accuracy_forget=[])

for i in range(1000):
    evaluate(data_dict, 500, 10, train_to_thresh=True, mask_d=0.5, disperssion=10,index=i)
try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle

with open(f'data_{np.random.randint(0,100000)}.p', 'wb') as fp:
    pickle.dump(data_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


class CDataLoader:
    def __init__(self, batch_size=10, mask_d=0.05,boundry_sep_r=0.01, disperssion=10, n_batch=50, label_margin_ratio=0.2,
                 normal_sampling=False):
        self.batch_size = batch_size
        self.is_rule_1 = True
        self.normal_sampling = normal_sampling
        self.mask_d = mask_d
        self.dispersion = disperssion
        self.n_batch = n_batch
        self.label_margin_ratio = label_margin_ratio

    def generate_data(self, n=None):
        x = np.random.random((2, self.batch_size if n is None else n)) * 2 - 1
        # if self.normal_sampling: #todo implement
        #     count = int(self.batch_size/4)
        #     vec = np.array([1,1])*(2/np.sqrt(2.))/2
        #     x1= np.random.multivariate_normal(mean=vec*(1+self.label_margin_ratio),cov=np.eye(2)/(2/np.sqrt(2.))/2,size=(count))
        #     x2= np.random.multivariate_normal(mean=vec*(1-self.label_margin_ratio),cov=np.eye(2)/(2/np.sqrt(2.))/2,size=(count))
        #     vec[0]*=-1
        #     x3= np.random.multivariate_normal(mean=vec*(1-self.label_margin_ratio),cov=np.eye(2)/(2/np.sqrt(2.))/2,size=(count))
        #     x4= np.random.multivariate_normal(mean=vec*(1+self.label_margin_ratio),cov=np.eye(2)/(2/np.sqrt(2.))/2,size=(count))
        #     x=np.vstack((x1,x2,x3,x4)).T
        rotation_matrix = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4)], [np.sin(np.pi / 4), np.cos(np.pi / 4)]])
        x = rotation_matrix @ x
        x[0, :] /= x[0, :].max()
        x[1, :] /= x[1, :].max()
        x_sgn = np.sign(x)
        x = x * self.dispersion
        x = x + np.array([[self.mask_d], [self.mask_d]]) * x_sgn
        x[:,
        np.linalg.norm(np.abs(x) - np.array([[self.mask_d], [self.mask_d]]), 1, axis=0) > self.dispersion / np.sqrt(
            2.)] *= (1 + self.label_margin_ratio/2.)
        x[:,np.linalg.norm(np.abs(x) - np.array([[self.mask_d], [self.mask_d]]), 1, axis=0) <= self.dispersion / np.sqrt(
            2.)] *= (1 - self.label_margin_ratio/2.)
        return (x, self.generate_label(x))

    def generate_first_rule_data(self):
        x = self.generate_data()[0]
        x = np.abs(x)
        x = x * (np.random.randint(0, 2, size=x.shape[1]) * 2 - 1)
        return x, self.generate_label(x)

    def switch(self):
        self.is_rule_1 = not self.is_rule_1

    def generate_second_rule_data(self):
        x = self.generate_data()[0]
        x = np.abs(x)
        mask_0 = np.random.randint(0, 2, size=x.shape[1]) * 2 - 1
        mask_1 = -mask_0

        x = x * (np.vstack((mask_0, mask_1)))

        return x, self.generate_label(x)

    def generate_label(self, x):
        # diameter = 1 + np.power(self.mask_d, 2)
        cond_1 = np.linalg.norm(np.abs(x) - np.array([[self.mask_d], [self.mask_d]]), 1,axis=0) > self.dispersion / np.sqrt(2.)
        cond_2 = x[1, :] < 0
        return np.logical_xor(cond_1, cond_2)

    def __iter__(self):
        current_rule = self.is_rule_1
        for i in range(self.n_batch):
            if current_rule:
                x, y = self.generate_first_rule_data()
            else:
                x, y = self.generate_second_rule_data()
            yield torch.Tensor(x.T).requires_grad_(), torch.Tensor(y)

    def __len__(self):
        return self.n_batch

    def plot_data(self, x=None, y=None, n=None):
        plt.hlines(xmin=-1, xmax=1, y=0, color='black')
        plt.vlines(ymin=-1, ymax=1, x=0, color='black')
        if x is None and y is None:
            x, y = self.generate_data(n)
        for i in range(y.size):
            plt.scatter(x[0, i], x[1, i], s=1, c='red' if y[i] == 1 else 'blue')
        plt.show()

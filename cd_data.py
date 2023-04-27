import numpy as np
import torch
import matplotlib.pyplot as plt

class CDataLoader:
    def __init__(self, batch_size=10,  mask_d=0.1, disperssion=1,n_batch=50):
        self.batch_size = batch_size
        self.is_rule_1 = True
        self.mask_d = mask_d
        self.dispersion = disperssion
        self.n_batch=n_batch
    def generate_data(self):
        polaric_rep = np.random.random((2, self.batch_size))  # polaric representation
        polaric_rep[0, :] = polaric_rep[0, :] * (1 - self.mask_d) + self.mask_d
        polaric_rep[1, :] = polaric_rep[1, :] * 2 * np.pi
        x = self._transform_polaric_rp(polaric_rep)
        return (x * self.dispersion, self.generate_label(x))

    def _transform_polaric_rp(self, polaric_rep):
        x = np.empty_like(polaric_rep)
        x[0, :] = polaric_rep[0, :] * np.cos(polaric_rep[1, :])
        x[1, :] = polaric_rep[0, :] * np.sin(polaric_rep[1, :])
        return x

    def generate_first_rule_data(self):
        polaric_rep = np.random.random((2, self.batch_size))  # polaric representation
        polaric_rep[0, :] = polaric_rep[0, :] * (1 - self.mask_d) + self.mask_d
        polaric_rep[1, :] = polaric_rep[1, :] * np.pi
        polaric_rep[1, polaric_rep[1, :] > np.pi / 2] += 0.5 * np.pi
        x = self._transform_polaric_rp(polaric_rep)
        return x * self.dispersion, self.generate_label(x)
    def switch(self):
        self.is_rule_1=not self.is_rule_1
    def generate_second_rule_data(self):
        polaric_rep = np.random.random((2, self.batch_size))  # polaric representation
        polaric_rep[0, :] = polaric_rep[0, :] * (1 - self.mask_d) + self.mask_d
        polaric_rep[1, :] = polaric_rep[1, :] * np.pi
        polaric_rep[1, polaric_rep[1, :] > np.pi / 2] += 0.5 * np.pi
        polaric_rep[1, :] += np.pi / 2
        x = self._transform_polaric_rp(polaric_rep)
        return x * self.dispersion, self.generate_label(x)

    def generate_label(self, x):
        # diameter = 1 + np.power(self.mask_d, 2)
        diameter = 0.95 + np.power(self.mask_d, 2)
        cond_1 = np.linalg.norm(x, 2, axis=0) > np.sqrt(diameter / (2.))
        cond_2 = x[1, :] < 0
        return np.logical_xor(cond_1, cond_2)

    def __iter__(self):
        current_rule= self.is_rule_1
        for i in range(self.n_batch):
            if current_rule:
                x,y = self.generate_first_rule_data()
            else:
                x, y = self.generate_second_rule_data()
            yield torch.Tensor(x.T).requires_grad_(),torch.Tensor(y)
    def __len__(self):
        return self.n_batch
    def plot_data(self, x=None, y=None):
        plt.hlines(xmin=-1, xmax=1, y=0, color='black')
        plt.vlines(ymin=-1, ymax=1, x=0, color='black')
        if x is None and y is None:
            x, y = self.generate_data()
        for i in range(y.size):
            plt.scatter(x[0, i], x[1, i], s=1, c='red' if y[i] == 1 else 'blue')
        plt.show()

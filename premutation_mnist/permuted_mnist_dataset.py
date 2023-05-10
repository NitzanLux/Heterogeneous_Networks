import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class PermutedMNIST(Dataset):
    def __init__(self, train=True, permute: [bool, np.ndarray] = False):
        self.mnist = torchvision.datasets.MNIST(root='./data', train=train, download=True,
                                                transform=transforms.ToTensor())
        self.train = train
        if isinstance(permute, np.ndarray):
            self.permutation = permute
        elif permute:
            self.permutation = np.random.permutation(28 * 28)
        else:
            self.permutation = np.arange(28 * 28)

    def __getitem__(self, index):
        image, label = self.mnist[index]
        image = image.view(-1, 28 * 28).numpy()
        permuted_image = image[:, self.permutation]
        label_ohv = torch.zeros((1, 10))
        label_ohv[:, label] = 1
        return torch.from_numpy(permuted_image).view(28 * 28), label

    def __len__(self):
        return len(self.mnist)

    def display_sample(self, index, inverted=False):
        image, label = self[index]
        inv = self.inverse_permutation()
        image = image.squeeze().numpy()
        if inverted:
            image = image.flatten()
            image = image[inv].reshape((28, 28))
        plt.imshow(image, cmap='gray')
        plt.title(f"Label: {label}")
        plt.show()

    # Naive Python3 Program to
    # find inverse permutation.

    # Function to find inverse permutations
    def inverse_permutation(self):
        inverse = np.zeros_like(self.permutation)
        for i, c in enumerate(self.permutation):
            inverse[c] = i
        return inverse

    def get_dataloader(self, batch_size, num_workers=1, shuffle=True):
        data_loader = torch.utils.data.DataLoader(self,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers)
        return data_loader
# first_dataset=PermutedMNIST(False)
# img_number=np.random.randint(0,1000)
# first_dataset.display_sample(img_number,True)
# second_dataset=PermutedMNIST(False,np.random.randint(1,1000))
# second_dataset.display_sample(img_number,False)
# second_dataset.display_sample(img_number,True)
# second_dataset=PermutedMNIST(False,np.random.randint(1,1000))
# second_dataset.display_sample(img_number,True)

import numpy as np
import os
# import torch

# from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, join
import sys


class MNIST2KDataset():
    """MNIST2k dataset."""

    def __init__(self, transform=None):
        """
        Args:
            txt_file (string): Path to the txt file with images.
            label_file (string): Path to the txt file with classes.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

        # Check with m-nist
        txt_file = "mnist2500_X.txt"
        label_file = "mnist2500_labels.txt"

        # Add module location
        txt_file = os.path.join(__location__, txt_file)
        label_file = os.path.join(__location__, label_file)

        self.imgs = np.loadtxt(txt_file)
        self.labels = np.array(np.loadtxt(label_file).tolist(), dtype=int)
        assert(len(self.imgs)==len(self.labels))
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        img = self.imgs[idx]
        label = self.labels[idx]
        sample = {'image': img, 'landmarks': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class TripletMNIST2K():
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, mnist_dataset):

        self.test_labels = mnist_dataset.labels
        self.test_data = mnist_dataset.imgs
        # generate fixed triplets for testing
        self.labels_set = set(self.test_labels)
        self.label_to_indices = {label: np.where(self.test_labels == label)[0]
                                    for label in self.labels_set}
        
        self.merge_label_indicies_by_group()

        random_state = np.random.RandomState(29)

        triplets = [(i,
                        random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                        random_state.choice(self.label_to_indices[
                                                np.random.choice(
                                                    list(self.labels_set - set([self.test_labels[i].item()]))
                                                )
                                            ])
        )
                    for i in range(len(self.test_data))]
        self.test_triplets = triplets

    def merge_label_indicies_by_group(self):
        outs = [0,1]
        composite = [4,6,8,9]
        prime = [2,3,5,7]

        groups = [outs, composite, prime]

        for group in groups:
            new_label_list = []
            for num in group:
                new_label_list.append(self.label_to_indices[num])
            new_label_list = np.concatenate(new_label_list)
            for num in group:
                self.label_to_indices[num] = new_label_list

    def __getitem__(self, index):

        anchor = self.test_triplets[index][0]
        positive = self.test_triplets[index][1]
        negative = self.test_triplets[index][2]

        return (anchor, positive, negative)

    def __len__(self):
        return len(self.mnist_dataset)


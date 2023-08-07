from abc import abstractmethod
import torch
import pdb
import random

from .clusterer import Clusterer

class RandomClusterer(Clusterer):
    clusterer_name = "random"

    def set_partitions(self, train_dl, valid_dl, test_dl):
        for dl,split in zip([train_dl, valid_dl, test_dl],["train", "valid", "test"]):
            
            indices = [i for i in range(len(dl.dataset.targets))]
            random.shuffle(indices)
            partition_size = len(indices) // len(self.partitions)

            for cidx in range(len(self.partitions)):
                self.partitions[cidx][split] = indices[cidx*partition_size:(cidx+1)*partition_size]

class ClassClusterer(Clusterer):
    clusterer_name = "class"

    def set_partitions(self, train_dl, valid_dl, test_dl):
        for cidx in range(len(self.partitions)):
            for dl,split in zip([train_dl, valid_dl, test_dl],["train", "valid", "test"]):
                self.partitions[cidx][split] =  [i for i,x in enumerate(dl.dataset.targets) if x == cidx]
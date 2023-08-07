from abc import abstractmethod
import random
from ..datasets import get_loader

import torch
import numpy as np
import pdb

class Clusterer():
    def __init__(self, cfg, writer, device, transforms):
        self.partitions = [
            {split: [] for split in self.loader_splits}
            for _ in range(cfg["num_clusters"])
        ]
        
        self.cfg = cfg
        self.writer = writer
        self.device = device
        self.transforms = transforms

        self.cluster_dataloaders = []

        # Caching for inference of cluster for datapoint
        self.idx_to_cluster = {}
    
    @property
    def num_datapoints(self):
        return sum([len(cluster["train"]) for cluster in self.partitions])

    @property
    def loaders(self):
        return [self.train_dataloader, self.valid_dataloader, self.test_dataloader]

    @property
    def val_is_test(self): 
        return (self.valid_dataloader.dataset.inputs == self.test_dataloader.dataset.inputs).all().item()

    @property
    def loader_splits(self):
        return ["train", "valid", "test"]
    
    @property
    def cluster_save_name(self):
        return self.cfg["cluster_partitions_save"] \
                if self.cfg["cluster_partitions_save"] is not None else self.clusterer_name+"_clusters"
    
    def get_cluster_idx(self, idx, split):
        return self.idx_to_cluster[split][idx]
        
    def load_id_estimates(self):
        try:
            self.partitions = self.writer.load_checkpoint(self.clusterer_name+"_clusters", "cpu")
            print("Loaded ID estimates from checkpoint")
        except:
            print("Could not load ID estimates from checkpoint")

    def get_cluster_weights(self, normalized=False):
        cws = [len(cluster["train"]) for cluster in self.partitions]
        if normalized:
            total_cw = sum(cws)
            return [cw / total_cw for cw in cws]
        else:
            return cws

    def get_cluster(self):
        weights = self.get_cluster_weights()
        return random.choices([i for i in range(len(weights))], weights=weights)[0]

    def _build_subset_dataset(self, split_loader, split, partition):
        images = split_loader.dataset.x[partition]
        labels = split_loader.dataset.targets[partition]
        return split_loader.dataset.__class__(split_loader.dataset.name, split, images, labels, transforms=self.transforms)

    def set_dataloaders(self):
        """"Populates cluster dataloaders"""
        for cidx,partitions in enumerate(self.partitions):
    
            subset_datasets = [
                self._build_subset_dataset(split_loader, split, partitions[split])
                    for split_loader,split in zip(
                        self.loaders,self.loader_splits
                    )
                ]
                
            subset_dataloaders = {
                split: get_loader(dataset, self.device, dl.batch_size, dl.drop_last)
                for dataset, dl, split in zip(subset_datasets, self.loaders, self.loader_splits)
            }

            self.cluster_dataloaders.append(subset_dataloaders)

    def cluster_batch(self, idxs, split):
        return [self.idx_to_cluster[split][idx.item()] for idx in idxs]
    
    def write_cluster_stats(self):
        # Length of each partition
        for cidx,partition in enumerate(self.partitions): 
            self.writer.write_scalar(f"cluster_train_size", len(partition["train"]), cidx)
            self.writer.write_scalar(f"cluster_test_size", len(partition["test"]), cidx)

        # Similarity to classes
        labels = {split:dl.dataset.targets.cpu() for split,dl in zip(self.loader_splits,self.loaders)}
        num_classes = labels["train"].max()+1
     
        cluster_to_class_dist = {i:[0 for i in range(num_classes)] for i in range(num_classes)}

        for cidx,partition in enumerate(self.partitions):
            for split in ["train", "test"]:
                for label in (partition[split]):
                    cluster_to_class_dist[cidx][labels[split][label]] += 1
            for class_label in range(num_classes):
                self.writer.write_scalar(f"cluster_{cidx}_class_dist", cluster_to_class_dist[cidx][class_label], class_label)
        
        # Accuracy 
        for split_idx,split in enumerate(self.loader_splits):
            cluster_labels = torch.zeros(labels[split].shape)
            for cidx,partition in enumerate(self.partitions):
                for idx in partition[split]:
                    cluster_labels[idx] = cidx

            reference_labels = {}

            # For loop to run through each label of cluster label
            for i in range(self.cfg["num_clusters"]):
                index  = np.where(cluster_labels == i,1,0)
                try:
                    num = np.bincount(labels[split][index==1]).argmax()
                except:
                    num = 0
                # print(i, num, np.bincount(labels[split][index==1]))
                reference_labels[i] = num

            cluster_class_assignment = torch.zeros(labels[split].shape)
            for i in range(len(cluster_class_assignment)):
                cluster_class_assignment[i] = reference_labels[cluster_labels[i].item()]

            accuracy = (labels[split]==cluster_class_assignment).to(torch.float32).mean().item()
            self.writer.write_scalar(f"cluster_accuracy_0train_1valid_2test", accuracy, split_idx)


    def set_idx_to_cluster_cache(self, split):
        self.idx_to_cluster[split] = {}
        for cidx,partition in enumerate(self.partitions): 
            for idx in partition[split]:
                self.idx_to_cluster[split][idx] = cidx

    def load_clusters(self):
        try:
            self.partitions = self.writer.load_checkpoint(self.cluster_save_name, "cpu", absolute_path=True)
            print(f"Loaded cluster partitions from checkpoint {self.cluster_save_name}")
        except:
            print("Could not load cluster partitions from checkpoint")

    @property
    def has_partitions(self):
        dataset_lengths = [loader.dataset.targets.shape[0] for loader in self.loaders]
        partition_lenghts = [sum( [len(partition[split]) for partition in self.partitions] ) for split in self.loader_splits]
        return dataset_lengths == partition_lenghts
    
    @property
    def save_clusters(self): return self.cfg["cluster_partitions_save"] is not None

    @abstractmethod
    def set_super_dataloaders(self, train_dl, valid_dl, test_dl):
        """Initializes self.partitions for input dataloader"""
        self.train_dataloader = train_dl
        self.valid_dataloader = valid_dl
        self.test_dataloader = test_dl

    @abstractmethod
    def initialize_clusters(self, train_dl, valid_dl, test_dl):
        """Initializes self.partitions for input dataloader"""
        self.train_dataloader = train_dl
        self.valid_dataloader = valid_dl
        self.test_dataloader = test_dl
        self.load_clusters()

        if not self.has_partitions:
            print("Running clusterer")
            self.set_partitions(train_dl, valid_dl, test_dl)

        self.set_idx_to_cluster_cache("test")
        
        self.write_cluster_stats() 
        
        if self.save_clusters: self.writer.write_checkpoint(self.cluster_save_name, self.partitions, absolute_path=True)

        self.set_dataloaders()

    def get_partition(self, cluster_idx):
        "Returns {train,valid,test} partitions for given cluster"
        return self.partitions[cluster_idx]
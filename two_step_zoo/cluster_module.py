import os
import pdb
import math
import torch
import torch.nn as nn
from .datasets.supervised_dataset import SupervisedDataset
from .utils import batch_or_dataloader
from tqdm import tqdm
import shutil

class BaseClusterModule(nn.Module):

    def __init__(self, child_modules, clusterer):
        super().__init__()
        self.child_modules = nn.ModuleList(child_modules)
        self.module_id = "clustering_module"
        self.clusterer = clusterer
    
    @property
    def module_type(self):
        return self.module_id
    
    @property
    def model_type(self):
        return self.module_id

    def get_cluster_dataset(self, cluster_idx, split):
        return self.clusterer.cluster_dataloaders[cluster_idx][split]

    def sample(self, n_samples):
        samples = []
        cidxs = [self.clusterer.get_cluster() for _ in range(n_samples)]
        for cidx in range(len(self.clusterer.partitions)):
            num_to_sample = sum([1 for i in cidxs if i == cidx])

            if num_to_sample > 0:
                samples.append( self.child_modules[cidx].sample(n_samples=num_to_sample) )
    
        return torch.concat(samples, dim=0).detach()
    
    @batch_or_dataloader(pass_idx=False, pass_label=False)
    def log_prob(self, x, split="test"):
        log_probs = []
        num_clusters = len(self.clusterer.partitions)
        weights = self.clusterer.get_cluster_weights(normalized=True)
        log_probs = []
        for cidx in range(num_clusters):
            log_prob = self.child_modules[cidx].log_prob(x)
            log_prob += math.log(weights[cidx])
            log_probs.append(log_prob)
        log_probs = torch.cat(log_probs, dim=1)
        log_probs = torch.logsumexp(log_probs, dim=1, keepdim=True)
        return log_probs

    @batch_or_dataloader(pass_idx=True, pass_label=True)
    def loss(self, x, idx=None, split="test"):
        log_probs = []
        for batch_idx in range(x[0].shape[0]):
            cidx = self.clusterer.get_cluster_idx(idx[batch_idx].item(), split=split)
            log_probs.append(self.child_modules[cidx].loss(x[0][batch_idx].unsqueeze(0)))
        return torch.stack(log_probs)
    
    def get_sample_dataset(self, num_samples, batch_size=32):
        if not self.sample_dataset_stale:
            return self.sample_dataset
        
        images = []
        
        with torch.no_grad():
            for i in tqdm(range(0, num_samples, batch_size), desc="Generating sample dataset"):
            
                if num_samples - i < batch_size:
                    curr_batch_size = num_samples - i
                else:
                    curr_batch_size = batch_size

                images.append(self.sample(curr_batch_size).cpu())

        images = torch.cat(images, dim=0)

        sample_dataset = SupervisedDataset("gen_samples", "train", images)

        self.sample_dataset = sample_dataset
        self.sample_dataset_stale = False

        return self.sample_dataset
    
    def cleanup(self):
        return


class ClusterModule(BaseClusterModule):

    def log_prob(self, dataloader, split="test"):
        log_probs = []
        num_clusters = len(self.clusterer.partitions)
        for cidx in range(num_clusters):
            idxs = self.clusterer.partitions[cidx][split]
            inputs = torch.stack([dataloader.dataset.inputs[idx] for idx in idxs]).to(self.device)
            log_probs.append(self.child_modules[cidx].log_prob(inputs))
        return torch.cat(log_probs, 0)

    @batch_or_dataloader(pass_idx=True, pass_label=True)
    def rec_error(self, x, idx=None, *args, **kwargs):
        cidxs = self.clusterer.cluster_batch(idx, split="test")
        errors = []
        for idx,cidx in enumerate(cidxs):
            error = self.child_modules[cidx].generalized_autoencoder.rec_error(x[0][idx].unsqueeze(0), idx=idx, *args, **kwargs)
            errors.append(error)
        return torch.tensor(errors)

    @property
    def device(self):
        return self.child_modules[0].generalized_autoencoder.device
    
class SingleClusterModule(BaseClusterModule):

    @property
    def device(self):
        return self.child_modules[0].device

class MemEfficientSingleClusterModule(SingleClusterModule, nn.Module):
    def __init__(self, child_modules, clusterer, module_save_dir):
        nn.Module.__init__(self) # TODO: investigate multiple inheritance
        self.child_modules = child_modules
        self.module_id = "clustering_module"
        self.clusterer = clusterer
        self.module_save_dir = module_save_dir

        # Enforce one component in memory at a time
        self.current_module_info = None
        self.current_trainer = None

        # Make save directory for modules
        os.makedirs(module_save_dir, exist_ok=True)

    @property
    def device(self):
        return self.cluster_module_device
    
    def get_cluster_module(self, cidx):
        module_info = self.child_modules[cidx]

        module = module_info["get_module_fn"](
            cfg=module_info["cfg"],
            data_dim=module_info["data_dim"],
            data_shape=module_info["data_shape"],
            train_dataset_size=module_info["train_dataset_size"]
        )

        if module_info["instantiated"]:

            checkpoint = torch.load(f"{self.module_save_dir}/module_{cidx}")
            module.load_state_dict(checkpoint)

        else:

            self.child_modules[cidx]["instantiated"] = True

        module.train(self.training)

        return module
    
    def save_current_component(self):
        # TODO: ensure everything needed gets saved in state dict
        torch.save(self.current_trainer.module.state_dict(), \
            f"{self.module_save_dir}/module_{self.current_module_info['cluster_component']}")
    
    def switch_component(self, trainer):
        if trainer == self.current_trainer: return 

        component_info = trainer.module
        cidx = component_info["cluster_component"]
        new_component = self.get_cluster_module(cidx).to(self.device)

        # Delete current component from memory
        if self.current_module_info is not None:
            self.save_current_component()
            self.current_trainer.remove_module(self.current_module_info)
        
        self.current_trainer = trainer 
        self.current_trainer.set_module(new_component)
        self.current_module_info = component_info
    
    def sample(self, n_samples):
        samples = []
        cidxs = [self.clusterer.get_cluster() for _ in range(n_samples)]
        for cidx in range(len(self.clusterer.partitions)):
            num_to_sample = sum([1 for i in cidxs if i == cidx])

            if num_to_sample > 0:
                self.switch_component(self.trainers[cidx])

                samples.append( self.current_trainer.module.sample(n_samples=num_to_sample) )

        return torch.concat(samples, dim=0).detach()
    
    @batch_or_dataloader(pass_idx=False, pass_label=False)
    def log_prob(self, x, split="test"):
        assert False, "Todo efficient cluster log prob"
        log_probs = []
        num_clusters = len(self.clusterer.partitions)
        weights = self.clusterer.get_cluster_weights(normalized=True)
        log_probs = []
        for cidx in range(num_clusters):
            log_prob = self.child_modules[cidx].log_prob(x)
            log_prob += math.log(weights[cidx])
            log_probs.append(log_prob)
        log_probs = torch.cat(log_probs, dim=1)
        log_probs = torch.logsumexp(log_probs, dim=1, keepdim=True)
        return log_probs

    @batch_or_dataloader(pass_idx=True, pass_label=True)
    def loss(self, x, idx=None, split="test"):
        log_probs = []
        for batch_idx in range(x[0].shape[0]):
            cidx = self.clusterer.get_cluster_idx(idx[batch_idx].item(), split=split)
            self.switch_component(self.trainers[cidx])
            log_probs.append(self.current_trainer.module.loss(x[0][batch_idx].unsqueeze(0)))
        return torch.stack(log_probs)
    
    def cleanup(self):
        print("Cleaning up cluster module memory dump")
        shutil.rmtree(self.module_save_dir)
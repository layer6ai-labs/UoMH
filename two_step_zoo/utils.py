import pdb
import torch
from torch.utils.data import DataLoader
import functools
import numpy as np
import torch.nn as nn
import os
import pickle

def get_distribution(base_distribution, num_mixture_components, distribution_mean_spacing, latent_dim, device):
    if "completely_learned" in base_distribution:
        mixture_weights = nn.Parameter(torch.ones(num_mixture_components),requires_grad=True)
        means = nn.Parameter(torch.arange((-num_mixture_components // 2)*distribution_mean_spacing, (num_mixture_components // 2 + num_mixture_components % 2)*distribution_mean_spacing, distribution_mean_spacing)[:,None].repeat(1,latent_dim).to(torch.float32),requires_grad=True)
        stds = nn.Parameter(torch.ones_like(means),requires_grad=True)
    elif "learned" in base_distribution:
        mixture_weights = nn.Parameter(torch.ones(num_mixture_components),requires_grad=True)
        means = torch.arange((-num_mixture_components // 2)*distribution_mean_spacing, (num_mixture_components // 2 + num_mixture_components % 2)*distribution_mean_spacing, distribution_mean_spacing)[:,None].repeat(1,latent_dim).to(torch.float32).to(device)
        stds =torch.ones_like(means)
    else:
        mixture_weights = torch.ones(num_mixture_components).to(device)
        means = torch.arange((-num_mixture_components // 2)*distribution_mean_spacing, (num_mixture_components // 2 + num_mixture_components % 2)*distribution_mean_spacing, distribution_mean_spacing)[:,None].repeat(1,latent_dim).to(torch.float32).to(device)
        stds = torch.ones_like(means)
    
    return mixture_weights,means,stds


def batch_or_dataloader(pass_idx=False, pass_label=False, agg_func=torch.cat):
    def decorator(batch_fn):
        '''
        Decorator for methods in which the first arg (after `self`) can either be
        a batch or a dataloader.

        The method should be coded for batch inputs. When called, the decorator will automatically
        determine whether the first input is a batch or dataloader and apply the method accordingly.
        '''
        @functools.wraps(batch_fn)
        def batch_fn_wrapper(ref, batch_or_dataloader, idx=None, label=None, **kwargs):
            if isinstance(batch_or_dataloader, DataLoader): # Input is a dataloader
                assert idx is None, "Indices cannot be passed along with dataloaders!"
               
                if pass_idx:
                    list_out = [batch_fn(ref, (batch,label) if pass_label else batch, idx=idx, **kwargs)
                                for batch, label, idx in batch_or_dataloader]
                
                else:
                    list_out = [batch_fn(ref, (batch,label) if pass_label else batch, **kwargs)
                                for batch, label, _ in batch_or_dataloader]
             
                if list_out and type(list_out[0]) in (list, tuple):
                    # Each member of list_out is a tuple/list; re-zip them and output a tuple
                    return tuple(agg_func(out) for out in zip(*list_out))
                else:
                    # Output is not a tuple
                    return agg_func(list_out)

            else: # Input is a batch
                return batch_fn(ref, batch_or_dataloader, **kwargs)

        return batch_fn_wrapper

    return decorator

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

def deterministic_shuffle(x):
    is_tensor = False
    if type(x) == torch.Tensor:
        is_tensor = True
        x = np.array(x)

    current_state = np.random.get_state()
    np.random.seed(0)
    np.random.shuffle(x)
    np.random.set_state(current_state)

    if is_tensor:
        x = torch.tensor(x)
        
    return x

def clip_dataset(dataset, max_samples):
    all_inds = np.arange(len(dataset))
    all_inds = deterministic_shuffle(all_inds)
    if max_samples > len(all_inds):
        rand_inds = all_inds
    else:
        rand_inds = all_inds[:max_samples]
    dataset = torch.utils.data.Subset(dataset, rand_inds)
    return dataset

def pickle_exists(name, prefix): 
    return os.path.exists(f'{prefix}/{name}.pickle')

def save_pickle(name, prefix, object):
    os.makedirs(prefix + "/" + "/".join(name.split("/")[:-1]), exist_ok=True)

    with open(f'{prefix}/{name}.pickle', 'wb') as handle: 
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(name, prefix):
    with open(f'{prefix}/{name}.pickle', 'rb') as handle:
        object = pickle.load(handle)
    
    return object
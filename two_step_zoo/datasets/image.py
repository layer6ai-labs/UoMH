import os
from pathlib import Path
import pdb
from typing import Any, Tuple
import pickle

import pandas as pd
import PIL
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.datasets
import torchvision.transforms as torchvision_transforms
from two_step_zoo.datasets.supervised_dataset import SupervisedDataset

from tqdm import tqdm
from ..utils import deterministic_shuffle

class CenterCropLongEdge(object):
  """Crops the given PIL Image on the long edge.
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
  """
  def __call__(self, img):
    """
    Args:
        img (PIL Image): Image to be cropped.
    Returns:
        PIL Image: Cropped image.
    """
    return torchvision_transforms.functional.center_crop(img, min(img.size))

  def __repr__(self):
    return self.__class__.__name__

class CelebA(Dataset):
    '''
    CelebA PyTorch dataset
    The built-in PyTorch dataset for CelebA is outdated.
    '''

    def __init__(self, root: str, role: str = "train"):
        self.root = Path(root)
        self.role = role
        
        self.transform = torchvision_transforms.Compose([
            torchvision_transforms.Resize((64, 64)),
            torchvision_transforms.ToTensor(),
        ])

        celeb_path = lambda x: self.root / x

        role_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        splits_df = pd.read_csv(celeb_path("list_eval_partition.csv"))
        self.filename = splits_df[splits_df["partition"] == role_map[self.role]]["image_id"].tolist()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_path = (self.root / "img_align_celeba" /
                    "img_align_celeba" / self.filename[index])
        X = PIL.Image.open(img_path)
        X = self.transform(X)

        return X, 0

    def __len__(self) -> int:
        return len(self.filename)
    
    def to(self, device):
        return self

def image_tensors_to_dataset(dataset_name, dataset_role, images, labels, transforms):
    images = images.to(dtype=torch.get_default_dtype())
    labels = labels.long()
    return SupervisedDataset(dataset_name, dataset_role, images, labels, transforms)

# Returns tuple of form `(images, labels)`. Both are uint8 tensors.
# `images` has shape `(nimages, nchannels, nrows, ncols)`, and has
# entries in {0, ..., 255}
def get_raw_image_tensors(dataset_name, train, data_root, class_ind):
    data_dir = os.path.join(data_root, dataset_name)

    if dataset_name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root=data_dir, train=train, download=True)
        images = torch.tensor(dataset.data).permute((0, 3, 1, 2))
        labels = torch.tensor(dataset.targets)
    
    elif dataset_name == "cifar100":
        dataset = torchvision.datasets.CIFAR100(root=data_dir, train=train, download=True)
        images = torch.tensor(dataset.data).permute((0, 3, 1, 2))
        labels = torch.tensor(dataset.targets)

    elif dataset_name == "svhn":
        dataset = torchvision.datasets.SVHN(root=data_dir, split="train" if train else "test", download=True)
        images = torch.tensor(dataset.data)
        labels = torch.tensor(dataset.labels)

    elif dataset_name in ["mnist", "fashion-mnist"]:
        dataset_class = {
            "mnist": torchvision.datasets.MNIST,
            "fashion-mnist": torchvision.datasets.FashionMNIST
        }[dataset_name]
        dataset = dataset_class(root=data_dir, train=train, download=True)
        images = dataset.data.unsqueeze(1)
        labels = dataset.targets

    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    if class_ind != -1:
        print("Restricting dataset to class:", class_ind)
        class_idxs = labels == class_ind
        labels = labels[class_idxs]
        images = images[class_idxs]

    return images.to(torch.uint8), labels.to(torch.uint8)

def get_torchvision_datasets(dataset_name, data_root, valid_fraction, class_ind, transforms):
    images, labels = get_raw_image_tensors(dataset_name, train=True, data_root=data_root, class_ind=class_ind)

    perm = torch.arange(images.shape[0])
    perm = deterministic_shuffle(perm)
    print("Torchvision dataset first inds of perm:", perm[:5])

    shuffled_images = images[perm]
    shuffled_labels = labels[perm]

    valid_size = int(valid_fraction * images.shape[0])
    valid_images = shuffled_images[:valid_size]
    valid_labels = shuffled_labels[:valid_size]
    train_images = shuffled_images[valid_size:]
    train_labels = shuffled_labels[valid_size:]

    train_dset = image_tensors_to_dataset(dataset_name, "train", train_images, train_labels, transforms)
    valid_dset = image_tensors_to_dataset(dataset_name, "valid", valid_images, valid_labels, transforms)
    
    test_images, test_labels = get_raw_image_tensors(dataset_name, train=False, data_root=data_root, class_ind=class_ind)
    test_dset = image_tensors_to_dataset(dataset_name, "test", test_images, test_labels, transforms)

    return train_dset, valid_dset, test_dset

def get_image_datasets(dataset_name, data_root, make_valid_dset, valid_fraction, class_ind=-1, transforms=None):
    if not make_valid_dset: valid_fraction = 0
    
    return get_torchvision_datasets(dataset_name, data_root, valid_fraction, class_ind, transforms)
import pdb
from sys import set_asyncgen_hooks
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from .image import get_image_datasets
from .tabular import get_tabular_datasets
from .generated import get_generated_datasets
from .supervised_dataset import EmptySupervisedDataset, SupervisedDataset


def get_loaders_from_config(cfg, device):
    """
    Wrapper function providing frequently-used functionality.

    Updates `cfg` with dataset information.
    """
    train_loader, valid_loader, test_loader = get_loaders(
        dataset=cfg["dataset"],
        device=device,
        data_root=cfg.get("data_root", "data/"),
        make_valid_loader=cfg["make_valid_loader"],
        valid_fraction=cfg["valid_fraction"] if cfg["make_valid_loader"] else None,
        train_batch_size=cfg["train_batch_size"],
        valid_batch_size=cfg["valid_batch_size"],
        test_batch_size=cfg["test_batch_size"],
        class_ind=cfg["class_ind"],
        transforms=cfg["transforms"]
    )

    train_dataset = train_loader.dataset.x
    cfg["train_dataset_size"] = train_dataset.shape[0]
    cfg["data_shape"] = tuple(train_dataset.shape[1:])
    cfg["data_dim"] = int(np.prod(cfg["data_shape"]))

    if not cfg["make_valid_loader"]:
        valid_loader = test_loader
        print("WARNING: Using test loader for validation")

    return train_loader, valid_loader, test_loader


def get_loaders(
        dataset,
        device,
        data_root,
        make_valid_loader,
        valid_fraction,
        train_batch_size,
        valid_batch_size,
        test_batch_size,
        class_ind,
        transforms
):
    if dataset in ["celeba", "mnist", "fashion-mnist", "cifar10", "cifar100", "svhn"] or "imagenet" in dataset:
        train_dset, valid_dset, test_dset = get_image_datasets(dataset, data_root, make_valid_loader, valid_fraction, class_ind, transforms)

    elif dataset in ["miniboone", "hepmass", "power", "gas", "bsds300"]:
        train_dset, valid_dset, test_dset = get_tabular_datasets(dataset, data_root)
        
    elif dataset in ["sphere", "klein", "two_moons"]:
        train_dset, valid_dset, test_dset = get_generated_datasets(dataset)

    else:
        raise ValueError(f"Unknown dataset {dataset}")
    
    train_loader = get_loader(train_dset, device, train_batch_size, drop_last=True)

    if make_valid_loader:
        valid_loader = get_loader(valid_dset, device, valid_batch_size, drop_last=False)
    else:
        valid_loader = None

    test_loader = get_loader(test_dset, device, test_batch_size, drop_last=False)

    return train_loader, valid_loader, test_loader
    
def get_empty_loader():
    dataset = EmptySupervisedDataset()
    return get_loader(dataset, "cpu", 1, False)

def get_loader(dset, device, batch_size, drop_last, set_device=True):

    if set_device:
        collate_fn = lambda x: tuple(x_.to(device) for x_ in default_collate(x))
    else:
        collate_fn = default_collate
 
    return DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn
    )


def get_embedding_loader(embeddings, batch_size, drop_last, role):
    dataset = SupervisedDataset(
        name="embeddings",
        role=role,
        x=embeddings
    )
    return get_loader(dataset, embeddings.device, batch_size, drop_last)


def remove_drop_last(loader, device=None):
    dset = loader.dataset
    return get_loader(dset, device if device is not None else dset.x.device, loader.batch_size, False)
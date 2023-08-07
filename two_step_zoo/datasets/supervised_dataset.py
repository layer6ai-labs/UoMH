import pdb
import torch
from typing import Any, Tuple
import torchvision.transforms as t

torchvision_transforms = {
    "horizontal_flip": t.RandomHorizontalFlip
}

class SupervisedDataset(torch.utils.data.Dataset):
    ''' 
    Generic implementation of torch Dataset
    '''
    def __init__(self, name, role, x, y=None, transforms=None):
        if y is None:
            y = torch.zeros(x.shape[0]).long()

        assert x.shape[0] == y.shape[0]
        assert role in ["train", "valid", "test"]

        self.name = name
        self.role = role

        self.x = x
        self.y = y

        if transforms is not None:
            self.transforms = []
            for transform in transforms:
                self.transforms.append(
                    torchvision_transforms[transform["module"]](**transform["params"])
                )
            self.transforms = t.Compose(self.transforms)
        else:
            self.transforms = None
    
    @property
    def targets(self):
        return self.y
    
    @property
    def inputs(self):
        return self.x
    
    @property
    def device(self):
        return self.y.device

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        image = self.x[index]

        if self.transforms is not None:
            image = self.transforms(image)
  
        return image, self.y[index], index

    def to(self, device):
        return SupervisedDataset(
            self.name,
            self.role,
            self.x.to(device),
            self.y.to(device)
        )

class EmptySupervisedDataset(SupervisedDataset):
    def __init__(self, name=None, role=None, x=None, y=None):
        super().__init__("empty_dataset", "test", torch.zeros([1]))
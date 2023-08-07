import pickle

from .clusterer import Clusterer
import pdb

class IDClusterer(Clusterer):
    def __init__(self, cfg, writer, device, transforms):
        super().__init__(cfg, writer, device, transforms)
        self.clusterer_name = "id"

        assert cfg["partitions_save"] is not None

        self.partitions_save = cfg["partitions_save"]

    def set_partitions(self, train_dl, valid_dl, test_dl):
        with open(self.partitions_save, 'rb') as handle:
            self.partitions = pickle.load(handle)
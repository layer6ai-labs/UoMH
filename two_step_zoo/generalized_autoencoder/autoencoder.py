import pdb
from . import GeneralizedAutoEncoder


class AutoEncoder(GeneralizedAutoEncoder):
    model_type = "ae"

    def loss(self, x):
        return self.rec_error(x).mean()
import pdb
from tqdm import tqdm
import numpy as np
import torch
from pytorch_fid import fid_score
import pdb

class InceptionHelper():
    def __init__(self, module, gt_loader, gen_samples, gen_batch_size, old_fid) -> None:
        self.module = module
        self.gt_loader = gt_loader
        self.gen_samples = gen_samples
        self.gen_batch_size = gen_batch_size
        self.inception = fid_score.InceptionV3().to(module.device).eval()
        self.old_fid = old_fid

    def gen_loader(self):
        # TODO: consider refactoring this into a `sample_loader` method in TwoStepDensityEstimator
        for i in range(0, self.gen_samples, self.gen_batch_size):
            if self.gen_samples - i < self.gen_batch_size:
                batch_size = self.gen_samples - i
            else:
                batch_size = self.gen_batch_size

            yield self.module.sample(batch_size), None, None

    def get_inception_features(self, im_loader=None):
        if im_loader:
            loader_len = len(self.gt_loader)
            loader_type = "ground truth"
        else:
            loader_len = self.gen_samples // self.gen_batch_size
            loader_type = "generated"
            im_loader = self.gen_loader()

        feats = []
        paused = False
        for batch, _, _ in tqdm(im_loader, desc=f"Getting {loader_type} features", leave=False, total=loader_len):
            # Convert grayscale to RGB
            if batch.ndim == 3:
                batch.unsqueeze_(1)
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)

            if not self.old_fid:    # Used for version consistency, old_fid should be False
                batch = (batch / 255)

            with torch.no_grad():
                batch_feats = self.inception(batch)[0]

            batch_feats = batch_feats.squeeze().cpu().numpy()
            feats.append(batch_feats)

        return np.concatenate(feats)

    def compute_inception_stats(self, im_loader=None):
        # Compute mean and covariance for generated and ground truth iterables
        feats = self.get_inception_features(im_loader)
        mu = np.mean(feats, axis=0)
        sigma = np.cov(feats, rowvar=False)

        return mu, sigma
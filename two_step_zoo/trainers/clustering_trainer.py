import matplotlib.pyplot as plt
import torch
import torchvision.utils

import pdb

class BaseClusteringTrainer:
    """
    Base class for training a clustering two-step module
    """

    def __init__(
            self,
            module,
            child_trainers,
            evaluator,
            writer,
            train_loader,
            valid_loader,
            test_loader,

            checkpoint_load_list,
            memory_efficient,
            
            only_test=False 
    ):
        self.module = module
        self.child_trainers = child_trainers
        self.writer = writer
        self.evaluator = evaluator
        self.only_test = only_test

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.memory_efficient = memory_efficient

        for trainer in child_trainers:
            if hasattr(trainer.module, "whitening_transform") and trainer.module.whitening_transform:
                trainer.whitening_loader = train_loader

        if self.memory_efficient:
            self.module.trainers = child_trainers

        self.load_checkpoint(checkpoint_load_list)

    def train(self):
        raise NotImplementedError("Implement train function in child classes")

    def write_checkpoint(self, tag):
        for trainer in self.child_trainers:
            trainer.write_checkpoint(tag)

    def load_checkpoint(self, checkpoint_load_list):
    
        for trainer in self.child_trainers:
            
            if hasattr(trainer, "is_twostep_trainer") and trainer.is_twostep_trainer: continue

            if self.memory_efficient:
                self.module.switch_component(trainer)

            for ckpt in checkpoint_load_list:
                try:
                    trainer.load_checkpoint(ckpt)
                    break
                except FileNotFoundError:
                    print(f"Did not find {ckpt} {trainer.module.module_id} checkpoint")


class DisjointSequentialClusteringTrainer(BaseClusteringTrainer):
    """Class for fully training a clustering model cluster-by-cluster"""
    def train(self):
        
        if not self.only_test:
            for trainer in self.child_trainers:
                self.train_component(trainer)

        try:
            self.sample_and_record()
        except AttributeError:
            print("No sample method available")

        test_results = self.evaluator.test()
        self.record_dict("test", test_results, save=True)
        self.module.cleanup()
    
    def train_component(self, trainer):
        if not self.memory_efficient:
            trainer.train()
        else:
            self.module.switch_component(trainer)
            trainer.train()

    def sample_and_record(self):
        NUM_SAMPLES = 64
        GRID_ROWS = 8

        with torch.no_grad():
            imgs = self.evaluator.module.sample(NUM_SAMPLES)
            imgs.clamp_(self.child_trainers[0].module.data_min.to(imgs.device), self.child_trainers[0].module.data_max.to(imgs.device))
            grid = torchvision.utils.make_grid(imgs, nrow=GRID_ROWS, pad_value=1, normalize=True, scale_each=True)
            grid_permuted = grid.permute((1,2,0))

            plt.figure()
            plt.axis("off")
            plt.imshow(grid_permuted.detach().cpu().numpy())

            self.writer.write_image(self.module.module_id+"/samples", grid, global_step=self.child_trainers[0].epoch)

    def record_dict(self, tag_prefix, value_dict, save=False):
        for k, v in value_dict.items():
            print(f"clusterer {k}: {v:.4f}")
            self.writer.write_scalar(f"cluster_{tag_prefix}_{k}", v, 0)

        if save:
            self.writer.write_json(
                f"cluster_{tag_prefix}_metrics",
                {k: v.item() for k, v in value_dict.items()}
            )
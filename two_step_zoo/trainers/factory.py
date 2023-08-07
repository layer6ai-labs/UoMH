import pdb
from .single_trainer import SingleTrainer
from .two_step_trainer import SequentialTrainer, AlternatingEpochTrainer, AlternatingIterationTrainer
from .clustering_trainer import DisjointSequentialClusteringTrainer
from ..evaluators import get_evaluator
import math

def get_single_trainer(
        module,
        writer,
        cfg,
        train_loader,
        valid_loader,
        test_loader,
        evaluator,
        only_test=False,
):
    return SingleTrainer(
        module=module,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        writer=writer,
        max_epochs=cfg["max_epochs"],
        early_stopping_metric=cfg["early_stopping_metric"],
        max_bad_valid_epochs=cfg["max_bad_valid_epochs"],
        max_grad_norm=cfg["max_grad_norm"],
        conditioning=cfg["conditioning"],
        evaluator=evaluator,
        only_test=only_test,
        epoch_sample_every=cfg.get("epoch_sample_every", math.inf)
    )


def get_trainer(
        two_step_module,

        writer,

        gae_cfg,
        de_cfg,
        shared_cfg,

        train_loader,
        valid_loader,
        test_loader,

        gae_evaluator,
        de_evaluator,
        shared_evaluator,

        load_best_valid_first,

        pretrained_gae_path,
        freeze_pretrained_gae,

        only_test=False
):
    gae_trainer = get_single_trainer(
        module=two_step_module.generalized_autoencoder,
        writer=writer,
        cfg=gae_cfg,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        evaluator=gae_evaluator,
        only_test=only_test,
    )
    de_trainer = get_single_trainer(
        module=two_step_module.density_estimator,
        writer=writer,
        cfg=de_cfg,
        train_loader=None,
        valid_loader=None,
        test_loader=None,
        evaluator=de_evaluator,
        only_test=only_test,
    )

    checkpoint_load_list = ["best_valid", "latest"] if load_best_valid_first else ["latest", "best_valid"]

    if shared_cfg["sequential_training"]:
        return SequentialTrainer(
            gae_trainer=gae_trainer,
            de_trainer=de_trainer,
            writer=writer,
            evaluator=shared_evaluator,
            checkpoint_load_list=checkpoint_load_list,
            pretrained_gae_path=pretrained_gae_path,
            freeze_pretrained_gae=freeze_pretrained_gae,
            only_test=only_test
        )

    elif shared_cfg["alternate_by_epoch"]:
        trainer_class = AlternatingEpochTrainer
    else:
        trainer_class = AlternatingIterationTrainer

    return trainer_class(
        two_step_module=two_step_module,
        gae_trainer=gae_trainer,
        de_trainer=de_trainer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        writer=writer,
        max_epochs=shared_cfg["max_epochs"],
        early_stopping_metric=shared_cfg["early_stopping_metric"],
        max_bad_valid_epochs=shared_cfg["max_bad_valid_epochs"],
        max_grad_norm=shared_cfg["max_grad_norm"],
        evaluator=shared_evaluator,
        checkpoint_load_list=checkpoint_load_list,
        pretrained_gae_path=pretrained_gae_path,
        freeze_pretrained_gae=freeze_pretrained_gae,
        only_test=only_test
    )

def get_two_step_trainer(
    two_step_module,
    writer,
    gae_cfg,
    de_cfg,
    shared_cfg,
    train_loader,
    valid_loader,
    test_loader,
    cluster_cfg=None,
    load_best_valid_first=False,
    pretrained_gae_path="",
    pretrained_decoder_subpath="",
    freeze_pretrained_gae=None,
    only_test=False
):
    gae_evaluator = get_evaluator(
        two_step_module.generalized_autoencoder,
        valid_loader=valid_loader, test_loader=valid_loader,
        train_loader=train_loader,
        valid_metrics=gae_cfg["valid_metrics"],
        test_metrics=gae_cfg["test_metrics"],
        **gae_cfg.get("metric_kwargs", {}),
    )
    de_evaluator = get_evaluator(
        two_step_module.density_estimator,
        valid_loader=None, test_loader=None, # Loaders must be updated later by the trainer
        train_loader=train_loader,
        valid_metrics=de_cfg["valid_metrics"],
        test_metrics=de_cfg["test_metrics"],
        **de_cfg.get("metric_kwargs", {}),
    )

    shared_evaluator = get_evaluator(
        two_step_module,
        train_loader=train_loader, valid_loader=train_loader, test_loader=train_loader,
        valid_metrics=shared_cfg["valid_metrics"],
        test_metrics=shared_cfg["test_metrics"],
        **shared_cfg.get("metric_kwargs", {}),
    )

    trainer = get_trainer(
        two_step_module=two_step_module,
        writer=writer,
        gae_cfg=gae_cfg,
        de_cfg=de_cfg,
        shared_cfg=shared_cfg,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        gae_evaluator=gae_evaluator,
        de_evaluator=de_evaluator,
        shared_evaluator=shared_evaluator,
        load_best_valid_first=load_best_valid_first,
        pretrained_gae_path=pretrained_gae_path,
        freeze_pretrained_gae=freeze_pretrained_gae,
        only_test=only_test
    )

    return trainer

def get_single_clustering_trainer(
    clustering_module,
    writer,
    cfg,
    cluster_cfg,
    train_loader,
    valid_loader,
    test_loader,
    only_test,
    load_best_valid_first=False

):

    trainers = []
    for cidx,module in enumerate(clustering_module.child_modules):
        evaluator = get_evaluator(
            module,
            train_loader=clustering_module.get_cluster_dataset(cidx, "train"), 
            valid_loader=clustering_module.get_cluster_dataset(cidx, "valid"), 
            test_loader=clustering_module.get_cluster_dataset(cidx, "test"),
            valid_metrics=cfg["valid_metrics"],
            test_metrics=cfg["test_metrics"],
            **cfg.get("metric_kwargs", {}),
        )
        trainer =  get_single_trainer(
                module,
                writer,
                cfg,
                train_loader=clustering_module.get_cluster_dataset(cidx, "train"),
                valid_loader=clustering_module.get_cluster_dataset(cidx, "valid"),
                test_loader=clustering_module.get_cluster_dataset(cidx, "test"),
                evaluator=evaluator,
                only_test=False,
        )
        
        trainer.cluster_component = cidx

        trainers.append(trainer)
    
    clustering_evaluator = get_evaluator(
        clustering_module,
        train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
        valid_metrics=cluster_cfg["valid_metrics"],
        test_metrics=cluster_cfg["test_metrics"],
        **cluster_cfg.get("metric_kwargs", {}),
    )

    if cluster_cfg["trainer"] == "DisjointSequential":
        trainer_class = DisjointSequentialClusteringTrainer
    else:
        raise ValueError(f"Unknown trainer {cluster_cfg['DisjoingSequential']}")

    checkpoint_load_list = ["best_valid", "latest"] if load_best_valid_first else ["latest", "best_valid"]

    return trainer_class(
        clustering_module,
        trainers,
        clustering_evaluator,
        writer,
        train_loader,
        valid_loader,
        test_loader,
        memory_efficient=cluster_cfg["memory_efficient"],
        checkpoint_load_list=checkpoint_load_list,
        only_test=only_test
    )

def get_clustering_trainer(
    clustering_module,
    writer,
    gae_cfg,
    de_cfg,
    shared_cfg,
    cluster_cfg,
    train_loader,
    valid_loader,
    test_loader,
    load_best_valid_first=False,
    only_test=False
):
    two_step_trainers = []
    for cidx,two_step_module in enumerate(clustering_module.child_modules):
        two_step_trainer = get_two_step_trainer(
                two_step_module=two_step_module,
                writer=writer,
                gae_cfg=gae_cfg,
                de_cfg=de_cfg,
                shared_cfg=shared_cfg,
                train_loader=clustering_module.get_cluster_dataset(cidx, "train"),
                valid_loader=clustering_module.get_cluster_dataset(cidx, "valid"),
                test_loader=clustering_module.get_cluster_dataset(cidx, "test"),
                cluster_cfg=cluster_cfg,
                only_test=only_test
            )
        
        two_step_trainer.cluster_component = cidx

        two_step_trainers.append(two_step_trainer)
    
    clustering_evaluator = get_evaluator(
        clustering_module,
        train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
        valid_metrics=cluster_cfg["valid_metrics"],
        test_metrics=cluster_cfg["test_metrics"],
        **cluster_cfg.get("metric_kwargs", {}),
    )

    if cluster_cfg["trainer"] == "DisjointSequential":
        trainer_class = DisjointSequentialClusteringTrainer
    else:
        raise ValueError(f"Unknown trainer {cluster_cfg['DisjoingSequential']}")
    
    checkpoint_load_list = ["best_valid", "latest"] if load_best_valid_first else ["latest", "best_valid"]

    return trainer_class(
        clustering_module,
        two_step_trainers,
        clustering_evaluator,
        writer,
        train_loader,
        valid_loader,
        test_loader,
        checkpoint_load_list=checkpoint_load_list,
        only_test=only_test,
        memory_efficient=cluster_cfg["memory_efficient"],
    )
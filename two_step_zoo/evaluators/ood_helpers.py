import os
import json
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from ..datasets import remove_drop_last


def ood_acc(
        module,
        is_test_loader,
        oos_test_loader,
        is_train_loader,
        oos_train_loader,
        savedir,
        low_dim=False,
        cache=None,
    ):
    # NOTE: `is` = in-sample, `oos` = out-of-sample
    score_fn = module.low_dim_log_prob if low_dim else module.log_prob

    def get_log_prob(dataloader, name):
        dataloader = remove_drop_last(dataloader)
        log_prob = np.zeros(dataloader.dataset.x.shape[0])

        ind = 0
        for batch, _, _ in tqdm(dataloader, leave=False, desc=name):
            with torch.no_grad():
                log_prob_batch = score_fn(batch)
                new_ind = ind + len(log_prob_batch)
                log_prob[ind:new_ind] = log_prob_batch.detach().cpu().numpy().squeeze()
                ind = new_ind

        np.save(os.path.join(savedir, f"{name}_lowdim_{low_dim}.npy"), log_prob)

        return log_prob

    loaders = [is_train_loader, is_test_loader, oos_train_loader, oos_test_loader]
    loader_names = ["is_train", "is_test", "oos_train", "oos_test"]

    log_probs = []
    for loader, name in zip(loaders, loader_names):
        log_probs.append(get_log_prob(loader, name))

    _, classification_rate = get_ood_threshold_and_classification_rate(*log_probs)

    return classification_rate


def get_ood_threshold_and_classification_rate(
    is_train_log_prob,
    is_test_log_prob,
    oos_train_log_prob,
    oos_test_log_prob
):
    def make_dataset(arr, ones):
        labels = np.ones(arr.shape[0]) if ones else np.zeros(arr.shape[0])
        return np.stack((arr, labels), axis=1)

    is_train_dset = make_dataset(is_train_log_prob, ones=True)
    oos_train_dset = make_dataset(oos_train_log_prob, ones=False)
    train_dataset = np.concatenate((is_train_dset, oos_train_dset), axis=0)

    decision_stump = DecisionTreeClassifier(max_depth=1)
    decision_stump = decision_stump.fit(train_dataset[:,0,np.newaxis], train_dataset[:,1])

    threshold = decision_stump.tree_.threshold[0]

    # NOTE:
    #   1. We want higher log probability to correspond to higher likelihood of being in-sample
    #   2. We need to account for imbalance in number of datapoints in each dataset
    is_gt_threshold = np.sum(is_test_log_prob > threshold)
    oos_lt_threshold = np.sum(oos_test_log_prob <= threshold)
    num_is_test = is_test_log_prob.shape[0]
    num_oos_test = oos_test_log_prob.shape[0]

    adjusted_num_correct = is_gt_threshold + num_is_test/num_oos_test * oos_lt_threshold
    classification_rate = adjusted_num_correct / (2*num_is_test)

    # # NOTE: Old approach which did not account for differing dataset sizes
    # predictions = test_dataset[:,0] > threshold
    # classification_rate = np.mean(predictions == test_dataset[:,1])

    return threshold, classification_rate


def plot_ood_histogram(
        range,
        threshold,
        is_test_log_prob,
        oos_test_log_prob,
        is_dataset_name,
        oos_dataset_name,
        title,
        decision_height,
        bins=100
):
    label_map = {
        "fashion-mnist": "FMNIST",
        "mnist": "MNIST",
        "cifar10": "CIFAR-10",
        "svhn": "SVHN"
    }

    _, ax = plt.subplots(1, 1)

    plt.plot([threshold, threshold], [0, decision_height], "--",
        linewidth=5, color="orange", label="Decision Boundary")
    plt.hist(oos_test_log_prob, color="b", bins=bins, histtype="stepfilled",
        alpha=0.7, label=label_map[oos_dataset_name], range=range)
    plt.hist(is_test_log_prob, color="g", bins=bins, histtype="stepfilled",
        alpha=0.7, label=label_map[is_dataset_name], range=range)

    xlabel = "$\log p_Z$"
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel("Number of samples", fontsize=15)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc=2, fontsize=12)

    plt.title(title, fontsize=18)
    # plt.show()


def plot_ood_histogram_from_run_dir(
        run_dir,
        low_dim,
        range,
        title,
        decision_height,
        bins=100
):
    dataset_names = ["is_train", "is_test", "oos_train", "oos_test"]

    log_probs = {}
    for name in dataset_names:
        log_probs[name] = np.load(os.path.join(run_dir, f"{name}_lowdim_{low_dim}.npy"))

    threshold, classification_rate = get_ood_threshold_and_classification_rate(*log_probs.values())
    print(f"Classification rate: {100*classification_rate:.2f}%")

    try:
        with open(os.path.join(run_dir, "shared_config.json"), "r") as f:
            is_dataset = json.load(f)["dataset"]
    except FileNotFoundError:
        with open(os.path.join(run_dir, "config.json"), "r") as f:
            is_dataset = json.load(f)["dataset"]

    ood_dataset_map = {
        "mnist": "fashion-mnist",
        "fashion-mnist": "mnist",
        "svhn": "cifar10",
        "cifar10": "svhn"
    }

    oos_dataset = ood_dataset_map[is_dataset]

    plot_ood_histogram(
        range=range,
        threshold=threshold,
        is_test_log_prob=log_probs["is_test"],
        oos_test_log_prob=log_probs["oos_test"],
        is_dataset_name=is_dataset,
        oos_dataset_name=oos_dataset,
        title=title,
        decision_height=decision_height,
        bins=bins
    )
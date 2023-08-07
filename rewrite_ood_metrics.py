#!/usr/bin/env python3

import sys
import os
import json
import numpy as np

from two_step_zoo.evaluators.ood_helpers import get_ood_threshold_and_classification_rate


run_dir = sys.argv[1]


rates = {}
for low_dim in [True, False]:
    dataset_names = ["is_train", "is_test", "oos_train", "oos_test"]

    log_probs = {}
    try:
        for name in dataset_names:
            log_probs[name] = np.load(os.path.join(run_dir, f"{name}_lowdim_{low_dim}.npy"))
    except FileNotFoundError:
        print("No low dim ood")
        continue

    _, classification_rate = get_ood_threshold_and_classification_rate(*log_probs.values())

    rates["likelihood_ood_acc"+low_dim*"_low_dim"] = classification_rate


with open(os.path.join(run_dir, "fixed_ood_metrics.json"), "w") as f:
    json.dump(rates, f, sort_keys=True, indent=4)
import os
import time
import sys

old_fid = True
debug = False

bigger_dsets = ["svhn", "cifar10", "cifar100"]
model = "vae"
name = "cluster"

for iter in range(3):
    for cluster_method in ["class", "agglomerative", "random"]:
        for dataset in ["cifar10", "svhn", "cifar100", "mnist", "fashion-mnist"]:
            time.sleep(1)
            log_folder = "paper_reproduce" if not debug else "paper_reproduce_debug"

            if dataset in bigger_dsets:
                cmd = f"python single_cluster_main.py \
                --dataset {dataset} \
                --model {model} \
                --run-name {log_folder}/{model}_{name}/{dataset}_{cluster_method}_{iter} \
                --config test_metrics=\'[]\' \
                --cluster-config metric_dataset_save={dataset}/twentyk_dset_allks \
                --cluster-config cluster_id_metric_dataset_save={dataset}/twentyk_dset_allks_kmeans \
                --cluster-config clustered_id_samples_save=paper_results/{model}_baselines/{dataset}_{iter}  \
                --config scale_data=True \
                --cluster-config cluster_method={cluster_method} \
                --cluster-config cluster_partitions_save=runs/clusters/{dataset}/{cluster_method} \
                --cluster-config test_metrics=\'[\"fid\"]\' \
                --config early_stopping_metric=loss"

            else:
                cmd = f"python single_cluster_main.py \
                --dataset {dataset} \
                --model {model} \
                --run-name {log_folder}/{model}_{name}/{dataset}_{cluster_method}_{iter} \
                --config test_metrics=\'[]\' \
                --cluster-config metric_dataset_save={dataset}/twentyk_dset_allks \
                --cluster-config cluster_id_metric_dataset_save={dataset}/twentyk_dset_allks_kmeans \
                --cluster-config clustered_id_samples_save=paper_results/{model}_baselines/{dataset}_{iter}  \
                --config scale_data=True \
                --config encoder_hidden_dims='[512,512]' \
                --config decoder_hidden_dims='[512,512]' \
                --cluster-config cluster_method={cluster_method} \
                --cluster-config cluster_partitions_save=runs/clusters/{dataset}/{cluster_method} \
                --cluster-config test_metrics=\'[\"fid\"]\' \
                --config early_stopping_metric=loss"
            
            if debug:
                print(cmd)
                os.system(cmd)
                exit() 
            else:
                print(cmd)
                os.system(cmd)
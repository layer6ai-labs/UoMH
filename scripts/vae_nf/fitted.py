import os
import pdb

debug = False
device = 0

model = ["vae", "flow"]

for iter in range(3):
    for cluster_method in ["class", "agglomerative"]:
        for dataset in ["cifar10", "mnist", "fashion-mnist", "svhn", "cifar100"]:
            log_folder = "paper_reproduce" if not debug else "paper_reproduce_debug"

            cmd = f"python3 cluster_main.py \
            --dataset {dataset} \
            --gae-model {model[0]} \
            --de-model {model[1]} \
            --run-name {log_folder}/vae_flow_fitted/{dataset}_{cluster_method}_{iter} \
            --gpu-id {device} \
            --cluster-config test_metrics=\'[\"fid\"]\' \
            --cluster-config metric_dataset_save={dataset}/twentyk_dset_allks \
            --cluster-config cluster_id_metric_dataset_save={dataset}//twentyk_dset_allks_kmeans \
            --cluster-config clustered_id_samples_save=paper_results/vae_flow_fitted/{dataset}_{iter}  \
            --de-config scale_data=False \
            --gae-config lr=0.001 \
            --de-config lr=0.001 \
            --cluster-config cluster_method={cluster_method} \
            --de-config whitening_transform=True \
            --de-config scale_data=False \
            --cluster-config cluster_partitions_save=runs/clusters/{dataset}/{cluster_method} \
            --cluster-config id_estimates_save=runs/id_estimates/{dataset}/{cluster_method} \
            --cluster-config id_estimator=mle \
            --gae-config early_stopping_metric=loss \
            --gae-config valid_metrics=\'[\"loss\"]\' \
            --gae-config max_bad_valid_epochs=30 \
            --gae-config use_lr_scheduler=True"
            
            if debug:
                print(cmd)
                os.system(cmd)
                exit() 
            else:
                print(cmd)
                os.system(cmd)
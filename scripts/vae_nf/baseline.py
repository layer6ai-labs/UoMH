import os
import pdb

debug = False
device = 0

model = ["vae", "flow"]

for iter in range(3):
    for dataset in ["cifar10", "mnist", "fashion-mnist", "svhn", "cifar100"]:
        log_folder = "paper_reproduce" if not debug else "paper_reproduce_debug"

        cmd = f"python3 main.py \
        --dataset {dataset} \
        --gae-model {model[0]} \
        --de-model {model[1]} \
        --run-name {log_folder}/vae_flow_baseline/{dataset}_{iter} \
        --gpu-id {device} \
        --shared-config test_metrics=\'[\"log_likelihood\"]\' \
        --de-config test_metrics=\'[\"loss\"]\' \
        --gae-config test_metrics=\'[\"loss\"]\' \
        --de-config valid_metrics=\'[\"loss\"]\' \
        --gae-config valid_metrics=\'[\"loss\"]\' \
        --cluster-config metric_dataset_save={dataset}/twentyk_dset_allks \
        --cluster-config cluster_id_metric_dataset_save={dataset}//twentyk_dset_allks_kmeans \
        --cluster-config clustered_id_samples_save=paper_results/vae_flow_baseline/{dataset}_{iter}  \
        --de-config scale_data=False \
        --gae-config lr=0.001 \
        --de-config lr=0.001 \
        --de-config whitening_transform=True \
        --de-config scale_data=False \
        --gae-config early_stopping_metric=loss \
        --load-best-valid-first \
        --load-pretrained-gae \
        --freeze-pretrained-gae"
        
        if debug:
            print(cmd)
            os.system(cmd)
            exit() 
        else:
            print(cmd)
            os.system(cmd)
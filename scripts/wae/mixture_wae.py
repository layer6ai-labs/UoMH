import os

debug = False

bigger_dsets = ["svhn", "cifar10", "cifar100"]
model = "wae"
name = "mixture"

for iter in range(3):
    for dataset in ["mnist", "fashion-mnist", "svhn", "cifar100", "cifar10"]:
        log_folder = "paper_reproduce" if not debug else "paper_reproduce_debug"

        if dataset in bigger_dsets:
            cmd = f"python single_main.py \
            --dataset {dataset} \
            --model {model} \
            --run-name {log_folder}/{model}_{name}/{dataset}_{iter} \
            --config test_metrics=\'[\"fid\"]\' \
            --cluster-config metric_dataset_save={dataset}/twentyk_dset_allks \
            --cluster-config cluster_id_metric_dataset_save={dataset}/twentyk_dset_allks_kmeans \
            --cluster-config clustered_id_samples_save=paper_results/{model}_baselines/{dataset}_{iter}  \
            --config scale_data=True \
            --is-gae \
            --config use_lr_scheduler=False \
            --cluster-config test_metrics=\'[\"fid\"]\' \
            --config base_distribution=mixture_of_gaussians \
            --config distribution_mean_spacing=3 \
            --config early_stopping_metric=l2_reconstruction_error"
            
        else:
            cmd = f"python single_main.py \
            --dataset {dataset} \
            --model {model} \
            --run-name {log_folder}/{model}_{name}/{dataset}_{iter} \
            --config test_metrics=\'[\"fid\"]\' \
            --cluster-config metric_dataset_save={dataset}/twentyk_dset_allks \
            --cluster-config cluster_id_metric_dataset_save={dataset}/twentyk_dset_allks_kmeans \
            --cluster-config clustered_id_samples_save=paper_results/{model}_baselines/{dataset}_{iter}  \
            --config scale_data=True \
            --config encoder_hidden_dims='[512]' \
            --config decoder_hidden_dims='[512]' \
            --is-gae \
            --config use_lr_scheduler=False \
            --cluster-config test_metrics=\'[\"fid\"]\' \
            --config base_distribution=mixture_of_gaussians \
            --config distribution_mean_spacing=3 \
            --config early_stopping_metric=l2_reconstruction_error"
        
        if debug:
            print(cmd)
            os.system(cmd)
            exit() 
        else:
            print(cmd)
            os.system(cmd)
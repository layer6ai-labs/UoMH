import os

debug = False

bigger_dsets = ["svhn", "cifar10", "cifar100"]
model = "wae"

for iter in range(3):
    for conditioning in ["agglomerative", "class"]:
        for dataset in ["mnist", "cifar10", "fashion-mnist", "svhn", "cifar100"]:
            log_folder = "paper_reproduce" if not debug else "paper_reproduce_debug"

            conditioning_dimension = 10 if dataset != "cifar100" else 100

            if dataset in bigger_dsets:
                cmd = f"python single_main.py \
                --dataset {dataset} \
                --model {model} \
                --run-name {log_folder}/{model}_cond_{conditioning}/{dataset}_{iter} \
                --config test_metrics=\'[\"fid\"]\' \
                --cluster-config metric_dataset_save={dataset}/twentyk_dset_allks \
                --cluster-config cluster_id_metric_dataset_save={dataset}/twentyk_dset_allks_kmeans \
                --cluster-config clustered_id_samples_save=paper_results/{model}_baselines/{dataset}_{iter}  \
                --config scale_data=True \
                --is-gae \
                --config use_lr_scheduler=False \
                --cluster-config test_metrics=\'[\"fid\"]\' \
                --config conditioning={conditioning} \
                --cluster-config cluster_partitions_save=runs/clusters/{dataset}/{conditioning} \
                --config conditioning_dimension={conditioning_dimension}"

            else:
                cmd = f"python single_main.py \
                --dataset {dataset} \
                --model {model} \
                --run-name {log_folder}/{model}_cond_{conditioning}/{dataset}_{iter} \
                --config test_metrics=\'[\"fid\"]\' \
                --cluster-config metric_dataset_save={dataset}/twentyk_dset_allks \
                --cluster-config cluster_id_metric_dataset_save={dataset}/twentyk_dset_allks_kmeans \
                --cluster-config cluster_partitions_save=runs/clusters/{dataset}/{conditioning} \
                --cluster-config clustered_id_samples_save=paper_results/{model}_baselines/{dataset}_{iter}  \
                --config scale_data=True \
                --config encoder_hidden_dims='[512]' \
                --config decoder_hidden_dims='[512]' \
                --is-gae \
                --config use_lr_scheduler=False \
                --cluster-config test_metrics=\'[\"fid\"]\' \
                --config encoder_hidden_dims='[512]' \
                --config decoder_hidden_dims='[512]' \
                --config conditioning={conditioning} \
                --config conditioning_dimension={conditioning_dimension}"
            
            if debug:
                print(cmd)
                os.system(cmd)
                exit() 
            else:
                print(cmd)
                os.system(cmd)
from datetime import datetime

def get_cluster_config(dataset):
    return {
        "trainer": "DisjointSequential",
        "cluster_method": "class",
        "num_clusters": 10,
        "cluster_norm": 255.,
        "cluster_partitions_save": None,
        "partitions_save": None,

        "id_estimates_save": None,
        "id_estimator": None,
        "id_estimate_num_datapoints_per_class": 5000,
        "max_k": 15,
        "id_est_batch_size": 256,
        "n_id_est_workers": 0,
        "eval_every_k": True,
        "latent_k": 10,
        "pfix": True,

        "metric_dataset_save": str(datetime.now(tz=None)),
        "cluster_id_metric_dataset_save": str(datetime.now(tz=None)),
        "clustered_id_samples_save": str(datetime.now(tz=None)),

        "memory_efficient": False,
        "module_save_dir": "memory_efficient_dump",

        "valid_metrics": ["l2_reconstruction_error"],
        "test_metrics": ["l2_reconstruction_error", "clustered_id", "fid", "id"],
    }
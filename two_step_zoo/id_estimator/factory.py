from .estimator import MLEIDEstimator

def get_id_estimator(cluster_cfg, writer):

    if cluster_cfg["id_estimator"] == "mle":
        id_estimator = MLEIDEstimator(cluster_cfg, writer)
    else:
        raise ValueError(f"Unknown ID estimator {cluster_cfg['id_estimator']}")

    return id_estimator
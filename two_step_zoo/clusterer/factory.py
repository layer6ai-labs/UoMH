from .class_clusterer import ClassClusterer,RandomClusterer
from .scipy_clusterer import AgglomerativeClusterer, KMeansClusterer,AgglomerativeSLClusterer
from .id_clusterer import IDClusterer

def get_clusterer(cluster_cfg, writer, device, transforms):
    if cluster_cfg["cluster_method"] == "class":
        clusterer = ClassClusterer(cluster_cfg, writer, device, transforms)
    elif cluster_cfg["cluster_method"] == "agglomerative":
        clusterer = AgglomerativeClusterer(cluster_cfg, writer, device, transforms)
    elif cluster_cfg["cluster_method"] == "kmeans":
        clusterer = KMeansClusterer(cluster_cfg, writer, device, transforms)
    elif cluster_cfg["cluster_method"] == "single_linkage":
        clusterer = AgglomerativeSLClusterer(cluster_cfg, writer, device, transforms)
    elif cluster_cfg["cluster_method"] == "id":
        clusterer = IDClusterer(cluster_cfg, writer, device, transforms)
    elif cluster_cfg["cluster_method"] == "random":
        clusterer = RandomClusterer(cluster_cfg, writer, device, transforms)
    else:
        raise ValueError(f"Unknown clusterer {cluster_cfg['cluster_method']}")
    
    return clusterer
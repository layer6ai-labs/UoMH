from sklearn.cluster import AgglomerativeClustering,MiniBatchKMeans

from abc import abstractmethod
import torch
import pdb

from .clusterer import Clusterer

class ScipyClusterer(Clusterer):
    def __init__(self, cfg, writer, device, transforms):
        super().__init__(cfg, writer, device, transforms)
        self.clusterer_name = "agglomerative"

    def set_partitions(self, train_dl, valid_dl, test_dl):
        train_images = train_dl.dataset.inputs.cpu() / self.cfg["cluster_norm"]
        valid_images = valid_dl.dataset.inputs.cpu() / self.cfg["cluster_norm"]
        test_images = test_dl.dataset.inputs.cpu() / self.cfg["cluster_norm"]

        num_samples = train_images.shape[0]
        cluster_inputs = train_images.reshape(num_samples,-1)
        self.clusterer.fit(cluster_inputs)
        self.cluster_labels = self.clusterer.labels_

        for cidx in range(len(self.partitions)):
            self.partitions[cidx]["train"] =  [i for i,x in enumerate(self.cluster_labels) if x == cidx]
        
        self.post_train_cluster()

        for images,split in zip([valid_images, test_images], ["valid", "test"]):
            images = images.reshape(images.shape[0], -1)
            clusters = {i: [] for i in range(self.cfg["num_clusters"])}
            for idx,image in enumerate(images):
                cluster = self.cluster_image(image)
                clusters[cluster].append(idx)

            for cidx,cluster in clusters.items():
                self.partitions[cidx][split] = cluster



    def post_train_cluster(self): return
        
class AgglomerativeClusterer(ScipyClusterer):
    """
    AgglomerativeClusterer with ward linkage
    """
    def __init__(self, cfg, writer, device, transforms):
        super().__init__(cfg, writer, device, transforms)
        # self.iter = 0
        self.clusterer = MiniBatchKMeans(n_clusters=cfg["num_clusters"])#AgglomerativeClustering(n_clusters=cfg["num_clusters"])
    
    def post_train_cluster(self):
        train_data = self.train_dataloader.dataset.inputs.cpu() / self.cfg["cluster_norm"]
        
        self.train_clusters = [
            torch.stack([train_data[idx] for idx in partition["train"]])
            for partition in self.partitions
        ]

    def cluster_image(self, image):
        assert hasattr(self, "train_clusters")
        image = image[None,:]
        ss_values = []

        for cluster in self.train_clusters:
            cluster_size = cluster.shape[0]
            cluster_mean = cluster.reshape(cluster.shape[0], -1).mean(0)
            sum_squared_diffs = (cluster_size / (1 + cluster_size)) * ((cluster_mean-image)**2).sum()
            ss_values.append(sum_squared_diffs)

        return torch.argmin(torch.tensor(ss_values)).item()

class KMeansClusterer(ScipyClusterer):
    def __init__(self, cfg, writer, device, transforms):
        super().__init__(cfg, writer, device, transforms)
        self.clusterer = MiniBatchKMeans(n_clusters=cfg["num_clusters"])

    def cluster_image(self, image):
        image = image[None,:]
        return torch.argmin(((image-self.clusterer.cluster_centers_)**2).mean(1)).item()

class AgglomerativeSLClusterer(ScipyClusterer):
    """
    AgglomerativeClusterer with single linkage
    """
    def __init__(self, cfg, writer, device, transforms):
        super().__init__(cfg, writer, device, transforms)
        self.clusterer = AgglomerativeClustering(n_clusters=cfg["num_clusters"], linkage="single")
    
    def post_train_cluster(self):
        train_data = self.train_dataloader.dataset.inputs.cpu() / self.cfg["cluster_norm"]
        
        self.train_clusters = [
            torch.stack([train_data[idx] for idx in partition["train"]])
            for partition in self.partitions
        ]

    def cluster_image(self, image):
        assert hasattr(self, "train_clusters")
        image = image[None,:]
        dist_values = []
        for cluster in self.train_clusters:
            min_dist = ((image-cluster.reshape(cluster.shape[0], -1))**2).sum(1).min(0) 
            dist_values.append(min_dist[0].item())
        return torch.argmin(torch.tensor(dist_values)).item()
from sklearn.cluster import SpectralClustering, KMeans, DBSCAN, AgglomerativeClustering
from spectralcluster import SpectralClusterer
import hdbscan
import numpy as np
import torch

class ClusterModule():
    """
    Class is used to select the clustering choice and get clustering labels

    Editors note: Yes, this is not the best implimentation of this. My bad on that one hahaha
    """
    def __init__(self, feature_list, choice = 'KMeans', n_cluster = 2) -> None:
        """
        Initialises the clustering module

        Parameters
        ----------
            feature_list: Numpy array 
                Numpy array of the features or embeddings for clustering
            choice: String
                Choice of the clustering methods that are stated in the module
                Choose from ['KMeans', 'Spectral', 'Agglomerative', 'Google_Spectral', 'hdbscan', 'DBScan']
            n_cluster: Integar
                Number of clusters (If known, if not n_cluster = 0)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.name = choice

        if n_cluster == 0:
            self.n_cluster = 2
        else:
            self.n_cluster = n_cluster

        match choice:

            case 'KMeans':
                if n_cluster == 0:
                    self.n_cluster = self.elbow_method(feature_list)
                self.clusterer = KMeans(n_clusters=self.n_cluster, random_state=0, n_init="auto").fit(feature_list)

            case 'Spectral':
                self.clusterer = SpectralClustering(n_clusters=self.n_cluster, random_state=0,
                                                    assign_labels='cluster_qr', affinity= 'nearest_neighbors').fit(feature_list)
                
            case 'Agglomerative':
                self.clusterer = AgglomerativeClustering(n_clusters = self.n_cluster).fit(feature_list)

            case 'Google_Spectral':
                self.clusterer = SpectralClusterer(
                            min_clusters=2,
                            max_clusters=7,
                            autotune=None,
                            laplacian_type=None,
                            refinement_options=None,
                            custom_dist="cosine")
                self.features = np.array(feature_list)

            case 'hdbscan':
                self.clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
                self.features = np.array(feature_list)

            case 'DBScan':
                #Not in use since eps and min_samples hard to define
                self.clusterer = DBSCAN(eps=3, min_samples=2).fit(feature_list)

            case _:
                print('Error: Clustering choice not found')
    
    def get_labels(self):
        """
        Get the labels for each embedding in a list form

        Returns
        ----------
            labels: Python list
                List with all the labels from each embedding
        """

        match self.name:

            case 'Google_Spectral':

                return self.clusterer.predict(self.features)
            
            case 'hdbscan':
                
                return self.clusterer.fit_predict(self.features)

        return self.clusterer.labels_
    
    def elbow_method(self, feature_list):
        """
        Performs elbow method on the given feature_list

        Parameters
        ----------
            feature_list: Numpy array 
                Numpy array of the features or embeddings for clustering
        Returns
        ----------
            index: Integar
                index for the ideal k value for kmeans clustering
        """

        distortions = []
        index = 0
        max = 0
        K = range(1,10)
        for k in K:
            kmeanModel = KMeans(n_clusters=k, n_init="auto")
            kmeanModel.fit(feature_list)
            distortions.append(kmeanModel.inertia_)

        for x in range(len(distortions)-1):
            
            if (max < distortions[x] - distortions[x+1]):
                max = distortions[x] - distortions[x+1]
                index = x+2

        return index
    


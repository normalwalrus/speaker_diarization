from sklearn.cluster import SpectralClustering, KMeans, DBSCAN, AgglomerativeClustering
import torch

class ClusterModule():
    def __init__(self, feature_list, choice = 'KMeans', n_cluster = 2) -> None:
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
                #TODO how to predict
                #Nearest_Neighbours seem to perform the best
                self.clusterer = SpectralClustering(n_clusters=self.n_cluster, random_state=0,
                                                    assign_labels='cluster_qr', affinity= 'nearest_neighbors').fit(feature_list)
                
            case 'Agglomerative':
                self.clusterer = AgglomerativeClustering(n_clusters = self.n_cluster).fit(feature_list)

            case 'DBScan':
                #Not in use since eps and min_samples hard to define
                self.clusterer = DBSCAN(eps=3, min_samples=2).fit(feature_list)

            case _:
                print('Error: Clustering choice not found')
    
    def get_labels(self):

        return self.clusterer.labels_
    
    def elbow_method(self, feature_list):

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
    


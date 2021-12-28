from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np

class TextSumKmeans():

    def __init__(self, sents_emb) -> None:
        '''
            sents_emb: list sentence embedding
        '''
        self.sents_emb = sents_emb

    def get_index_from_kmeans(self):
        """
        Input a list of embeded sentence vectors
        Output an list of indices of sentence in the paragraph, represent the clustering of key sentences
        Note: Kmeans is used here for clustering
        """
        # sqrt(len)
        n_clusters = np.ceil(len(self.sents_emb)**0.5)

        # ratio: 0.3
        # n_clusters = np.ceil(len(sents_emb)*0.3)
        kmeans = KMeans(n_clusters=int(n_clusters))
        kmeans = kmeans.fit(self.sents_emb)
        sum_index,_ = pairwise_distances_argmin_min(kmeans.cluster_centers_, self.sents_emb, metric='cosine')
        sum_index = sorted(sum_index)
        return sum_index


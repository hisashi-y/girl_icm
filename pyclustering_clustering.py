import matplotlib.pyplot as plt
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import SIMPLE_SAMPLES
from sklearn.decomposition import PCA
import torch
import pickle

with open('sentence_embedding_pairs.bin', 'rb') as f:
    data = pickle.load(f)
labels = []
embeddings = []
for i, j in data:
    labels.append(i)
    embeddings.append(j.tolist())


pca = PCA(n_components=3)
pca.fit(embeddings)
transformed = pca.transform(embeddings)
print(len(transformed))

# Prepare initial centers - amount of initial centers defines amount of clusters from which X-Means will
# start analysis.
amount_initial_centers = 2
initial_centers = kmeans_plusplus_initializer(transformed, amount_initial_centers).initialize()
# Create instance of X-Means algorithm. The algorithm will start analysis from 2 clusters, the maximum
# number of clusters that can be allocated is 20.
xmeans_instance = xmeans(transformed, initial_centers, 20)
xmeans_instance.process()
# Extract clustering results: clusters and their centers
clusters = xmeans_instance.get_clusters()
centers = xmeans_instance.get_centers()
# Visualize clustering results
visualizer = cluster_visualizer()
visualizer.set_canvas_title('girls in COCA')
visualizer.append_clusters(clusters, transformed, markersize=5)
visualizer.append_cluster(centers, None, marker='*', markersize=70)
visualizer.show()
visualizer.save('girl_cluster_3d.png')

# print('centers:', centers)
# print('clusters:', clusters)

# for i in range(100):
#     label_idx = clusters[1][i]
#     print('----beginning of {}st sentence----'.format(i))
#     print(labels[label_idx])

import numpy as np
import matplotlib.pyplot as plt
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import SIMPLE_SAMPLES
from sklearn.decomposition import PCA
import torch
import pickle

labels = [1000, 1001]
transformed = []
transformed.append(np.array([1, 1], dtype=float))
transformed.append(np.array([5, 5], dtype=float))
print(transformed[0].shape)
for i in range(98):
    labels.append(i)
    if i <= 48:
        transformed.append(np.random.uniform(0.5, 1.5, (2, )))
    else:
        transformed.append(np.random.uniform(4.5, 5.5, (2, )))

transformed = np.array(transformed)
# print(transformed)

# pca = PCA(n_components=2)
# pca.fit(embeddings)
# transformed = pca.transform(embeddings)
# print(transformed)
# print(len(transformed))




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
visualizer.set_canvas_title('students')
visualizer.append_clusters(clusters, transformed, markersize=5)
visualizer.append_cluster(centers, None, marker='*', markersize=10)
visualizer.show()
visualizer.save('students.png')

# print('centers:', centers)
# print(centers[0])
print('clusters:', clusters)

print('1, 1付近のクラスタとそのラベルを表示')
for i in clusters[-1]:
    print('label:', labels[i])
    print('embedding:', transformed[i])


# for i in range(100):
#     label_idx = clusters[1][i]
#     print('----beginning of {}st sentence----'.format(i))
#     print(labels[label_idx])

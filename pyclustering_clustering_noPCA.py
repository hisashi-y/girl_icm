import numpy as np
import matplotlib.pyplot as plt
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import SIMPLE_SAMPLES
from sklearn.decomposition import PCA
from sklearn import preprocessing
import torch
import pickle
import random
import faiss

# (sentence, embedding) のtupleのリストを読み込む
with open('sentence_embedding_pairs.bin', 'rb') as f:
    data = pickle.load(f)
# ドメインの偏りをなくすためにシャッフル
random.shuffle(data)

labels = []
embeddings = []
for i, j in data:
    labels.append(i)
    embeddings.append(j.tolist())

labels = np.array(labels)
embeddings = np.array(embeddings)

ss = preprocessing.StandardScaler()
embeddings = ss.fit_transform(embeddings)

print(len(embeddings))
# PCAの訓練用にデータの25%を分割
train, test = np.split(embeddings, [int(len(embeddings) * 0.25)])
train_label, test_label = np.split(labels, [int(len(labels) * 0.25)])
# print(len(train))
# print(len(test))

# # PCAで3次元まで削減
# pca = PCA(n_components=3)
# # PCAの訓練
# pca.fit(train)
# transformed = pca.transform(test)
# print(len(transformed))

# Prepare initial centers - amount of initial centers defines amount of clusters from which X-Means will
# start analysis.
amount_initial_centers = 2
initial_centers = kmeans_plusplus_initializer(test, amount_initial_centers).initialize()
# Create instance of X-Means algorithm. The algorithm will start analysis from 2 clusters, the maximum
# number of clusters that can be allocated is 20.
xmeans_instance = xmeans(test, initial_centers, 20)
xmeans_instance.process()
# Extract clustering results: clusters and their centers
clusters = xmeans_instance.get_clusters()
centers = xmeans_instance.get_centers()
# Visualize clustering results
# visualizer = cluster_visualizer()
# visualizer.set_canvas_title('girls in COCA')
# visualizer.append_clusters(clusters, transformed, markersize=5)
# visualizer.append_cluster(centers, None, marker='*', markersize=70)
# visualizer.show()
# visualizer.save('girl_cluster_3d_split_std_shuffle.png')

# print('centers:', centers)
# print('clusters:', clusters)
# print('num of clusters:', len(centers))
# print(len(centers) == len(clusters))

# # for i in range(100):
# #     label_idx = clusters[1][i]
# #     print('----beginning of {}st sentence----'.format(i))
# #     print(labels[label_idx])

# print('last element of original label:', labels[-20])
# print('last element of original embeddings:', embeddings[-20])
# print('last element of test label:', test_label[-20])
# print('last element of test embedding:', test[-20])

# print('xmeans_instance.predict(transformed):', xmeans_instance.predict(transformed))

for i in range(len(centers)):
    print(f'--the number {i} cluster and its centroid--')
    centroid = centers[i]
    distances = []
    for j in clusters[i]: # そのcentroidが属するclusterの各成員に対して
        distance = np.linalg.norm(centroid - test[j])
        distances.append(distance)
    distances = np.array(distances)
    # 距離の小さい順
    indice = distances.argsort()
    # print('distances:', distances)
    # print('indice', indice)
    for idx in indice[:10]:
        label_idx = clusters[i][idx]
        print('distance:', distances[idx])
        label_sentence = test_label[label_idx]
        print('the beginning of sentence:')
        print(label_sentence)


import pickle
import torch
from sklearn.manifold import TSNE
import time
import faiss
import numpy as np

with open('/Users/hisashi-y/python codes/NLP lab/girl_icm/sentence_embedding_pairs.bin', 'rb') as f:
    lst = pickle.load(f)
X = []
for i, j in lst:
    X.append(j)

query_sentence_embedding = lst[0][1]

print(type(X))
X = torch.stack(X, dim= 0)
print(X)

# tsne = TSNE(n_components=2, random_state=0)
# redduced_embeddings = tsne.fit_transform(X)

# class FaissKMeans:
#     def __init__(self, n_clusters=8, n_init=10, max_iter=300):
#         self.n_clusters = n_clusters
#         self.n_init = n_init
#         self.max_iter = max_iter
#         self.kmeans = None
#         self.cluster_centers_ = None
#         self.inertia_ = None
#
#     def fit(self, X, y):
#         self.kmeans = faiss.Kmeans(d=X.shape[1],
#                                    k=self.n_clusters,
#                                    niter=self.max_iter,
#                                    nredo=self.n_init)
#         self.kmeans.train(X.astype(np.float32))
#         self.cluster_centers_ = self.kmeans.centroids
#         self.inertia_ = self.kmeans.obj[-1]
#
#     def predict(self, X):
#         return self.kmeans.index.search(X.astype(np.float32), 1)[1]

# s = time.time()
# k = 2
# n_init = 10
# max_iter = 300
X = X.detach().numpy().copy().astype(np.float32)
query_sentence_embedding = query_sentence_embedding.detach().numpy().copy().astype(np.float32)
# print(X)
# kmeans = faiss.Kmeans(d=3072, k=k, niter=max_iter, nredo=n_init)
# kmeans.train(X)
#
# e = time.time()
# print("Training time = {}".format(e - s))
#
# s = time.time()
# kmeans.index.search(X, 1)[1]
# e = time.time()
# print("Prediction time = {}".format((e - s) / len(y_test)))
print(X.ndim)
print(query_sentence_embedding.ndim)

index = faiss.index_factory(, 'Flat')
index.train(X)
index.add(X)
distances, neighbors = index.search(query_sentence_embedding, 1)

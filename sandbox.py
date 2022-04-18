import numpy as np
import pickle
import torch
# import faiss

n = 4
d = 3072
k = 4

# xb = np.random.random((n, d)).astype('float32')
# print(xb)
#

# with open('sentence_embedding_pairs.bin', 'rb') as f:
#     lst = pickle.load(f)

# print(len(lst))

# lst = [2, 4, 0, 250]

# for i in lst:
#     try:
#         print(250 / i)
#     except ZeroDivisionError:
#         print('error')
#         print('i in this iteration:', i)

with open('sentence_embedding_pairs.bin', 'rb') as f:
    lst = pickle.load(f)

counter = 0
while counter < 20:
    print('counter:', counter)
    counter += 1
    print(lst[counter])

# embeddings = []
# labels = []
# for i, j in lst:
#     labels.append(i)
#     embeddings.append(j)

# tensor = torch.stack(embeddings, dim = 0)
# # print(tensor.shape)

# tensor = tensor.detach().numpy().copy().astype('float32')
# # print(tensor.size)
# query = lst[2][1].detach().numpy().copy().astype('float32')
# print(query)
# new_query = query.reshape(1, -1)
# print(new_query)

# index = faiss.index_factory(d, 'Flat')
# index.train(tensor)
# index.add(tensor)
# distances, neighbors = index.search(query.reshape(1, -1).astype('float32'), k)

# print('distance:', distances )
# print('neighbors', neighbors)

# # for i in range(4):
# #     idx = int(np.where(neighbors==i)[1])
# #     print(idx)
# #     sentence = labels[idx]
# #     print(f'number {i+1} nearest neighbor is: {sentence}')

# idx = int(np.where(neighbors==0)[1])
# print(idx)
# sentence = labels[idx]
# print(sentence)

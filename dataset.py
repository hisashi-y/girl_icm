import pickle

lst = []
for i in range(4):
    s = input()
    lst.append(s)

with open('example_sentences.bin', 'wb') as f:
    pickle.dump(lst, f)

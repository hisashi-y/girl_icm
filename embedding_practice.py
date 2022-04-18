import torch
from torch import nn
from transformers import BertTokenizer, BertModel
import json
import pickle
# import matplotlib.pyplot as plt
# %matplotlib inline

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

with open('girl_genres.json', 'r') as f:
    dict = json.load(f)

# dict = {'I saw a pretty cute girl at school.': 'acad', 'There goes a ugly girl in the street.': 'blog'}

# sentences = []
# for sentence in lst:
#     tagged_sentence = "[CLS] " + sentence + " [SEP]"
#     sentences.append(tagged_sentence)
#
# # print(sentences)
#
# tokenized_sentences = []
# for sentence in sentences:
#     tokenized_text = tokenizer.tokenize(sentence)
#     tokenized_sentences.append(tokenized_text)

# 単文を想定しているので複数の文を含むテキストに対応させる必要がある
def token_idx_pair(sentence):
    marked_text = "[CLS] " + sentence + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Display the words with their indeces.
    # for tup in zip(tokenized_text, indexed_tokens):
    #     print('{:<12} {:>6,}'.format(tup[0], tup[1]))

    segments_ids = [1] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    model = BertModel.from_pretrained('bert-base-uncased',output_hidden_states = True, # Whether the model returns all hidden-states.
    )
    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    with torch.no_grad():

        try:
            outputs = model(tokens_tensor, segments_tensors)
        except RuntimeError:
            print('Runtime Error')
            print('sentence:', sentence)
            print('tokens_tensor:', tokens_tensor)
            print('segments_tensors', segments_tensors)
            return None, None
        hidden_states = outputs[2]

    # print ("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
    # layer_i = 0
    # print ("Number of batches:", len(hidden_states[layer_i]))
    # batch_i = 0
    # print ("Number of tokens:", len(hidden_states[layer_i][batch_i]))
    # token_i = 0
    # print ("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))
    # For the 5th token in our sentence, select its feature values from layer 5.

    token_embeddings = torch.stack(hidden_states, dim = 0)
    # print(token_embeddings.size())
    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    # print(token_embeddings.size())
    token_embeddings = token_embeddings.permute(1,0,2)

    token_vecs_cat = []
    for token in token_embeddings:
        # `token` is a [12 x 768] tensor
        # Concatenate the vectors (that is, append them together) from the last
        # four layers.
        # Each layer vector is 768 values, so `cat_vec` is length 3,072.
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)

        # Use `cat_vec` to represent `token`.
        token_vecs_cat.append(cat_vec)
    # print ('Shape is: %d x %d' % (len(token_vecs_cat), len(token_vecs_cat[0])))
    girl_index = -1
    for i, token_str in enumerate(tokenized_text):
        if token_str == 'girl' or token_str == 'girls':
            girl_index = int(i)
    # Sanity Check: girlが含まれてなかったケース（含まれているべきなのに）
    if girl_index == -1:
        print('the word girl is not in this sentence')
        print('the sentence for reference:', sentence)
        return None, None
    return sentence, token_vecs_cat[girl_index]


sentence_embedding_pairs = []
for sentence, genre in dict.items():
    # print('---Beginning of the text this time--')
    label_sentence, embedding = token_idx_pair(sentence)
    if label_sentence == None:
        # print('the word girl is not in this sentence and continue')
        # print('the sentence for your reference:', label_sentence)
        continue
    else:
        sentence_embedding_pairs.append((label_sentence, embedding))

# for sentence, embedding in sentence_embedding_pairs:
#     print('girl in this context:', sentence, 'word embedding is:', str(embedding))

cos_sim = nn.CosineSimilarity(dim=0)

# print(sentence_embedding_pairs[2][1])

print('similartity between 1st and 2nd sentence:', cos_sim(sentence_embedding_pairs[0][1], sentence_embedding_pairs[1][1]))
# print('similartity between 1st and 3nd sentence:', cos_sim(sentence_embedding_pairs[0][1], sentence_embedding_pairs[2][1]))
# print('similartity between 1st and 4th sentence:', cos_sim(sentence_embedding_pairs[0][1], sentence_embedding_pairs[3][1]))
# print('similartity between 2nd and 3nd sentence:', cos_sim(sentence_embedding_pairs[1][1], sentence_embedding_pairs[2][1]))
# print('similartity between 2nd and 4th sentence:', cos_sim(sentence_embedding_pairs[1][1], sentence_embedding_pairs[3][1]))
# print('similartity between 3rd and 4th sentence:', cos_sim(sentence_embedding_pairs[2][1], sentence_embedding_pairs[3][1]))
print('sanity check')
print('similartity between 1st and 1nd sentence:', cos_sim(sentence_embedding_pairs[0][1], sentence_embedding_pairs[0][1]))

print('sentence_embedding_pairs', sentence_embedding_pairs)
print('# of sentences:', len(sentence_embedding_pairs))

with open('sentence_embedding_pairs.bin', 'wb') as f:
    pickle.dump(sentence_embedding_pairs, f)

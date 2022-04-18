import spacy
import pickle

# with open('/Users/hisashi-y/python codes/NLP lab/girl_icm/example_sentences.bin', 'rb') as f:
#     lst = pickle.load(f)

nlp = spacy.load('en_core_web_trf')

def get_embeddings(text):
    return nlp(text).vector


# data = {}
# for sentence in lst:
#     data[sentence] = get_embeddings(sentence)
#
# print(data)

print(get_embeddings('I have a pen.'))

from sentence_transformers import SentenceTransformer
from sentence_transformers import models
import os
model_path = '/home/molin/Downloads/bert-base-uncased'
if os.path.exists(os.path.join(model_path, 'config.json')):
    print('yes')

model = SentenceTransformer('bert-base-nli-mean-tokens')

sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.', 
    'The quick brown fox jumps over the lazy dog.']
sentence_embeddings = model.encode(sentences)
import numpy as np
print(np.array(sentence_embeddings).shape)

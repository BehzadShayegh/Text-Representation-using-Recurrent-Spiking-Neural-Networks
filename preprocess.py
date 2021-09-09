#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install normalise

import nltk
# nltk.download('brown')
# nltk.download('names')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('universal_tagset')
# nltk.download('stopwords')
# nltk.download('punkt')

from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer
from normalise import normalise
import numpy as np

porter=PorterStemmer()
tokenizer = TreebankWordTokenizer()
stem_words = np.vectorize(porter.stem)

from tqdm.notebook import tqdm


# In[2]:


import pandas as pd
df = pd.read_csv('data.csv')
df.head()


# In[11]:


import json

data = []
targets = []
for (i, row) in tqdm(df.iterrows(), total=df.shape[0]):
    try:
        doc = row.short_description
        if type(doc) is not str:
            continue
        edited = stem_words(
            tokenizer.tokenize(
                ' '.join(
                    normalise(
                        tokenizer.tokenize(
                            doc.lower()
                        ),
                        verbose=False
                    )
                )
            )
        ).tolist()
        data.append(edited)
        targets.append(row.category)
    except:
        continue


# In[15]:


train, test = data[:20000], data[20000:]
train_targets, test_targets = targets[:20000], targets[20000:]

json.dump(train, open('data.json','w'))
json.dump(train_targets, open('targets.json','w'))

json.dump(test, open('test_data.json','w'))
json.dump(test_targets, open('test_targets.json','w'))

words = [word for doc in data for word in doc]
words, counts = np.unique(words, return_counts=True)
words = words[counts>1]
words = ['<UKN>','<s>','</s>']+list(words)
json.dump(words, open('words.json','w'))

vocab = {w:i for i,w in enumerate(words)}
vocab['<PAD>'] = -1
json.dump(vocab, open('vocab.json','w'))


# In[ ]:





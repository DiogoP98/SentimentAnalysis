#!/usr/bin/env python
# coding: utf-8

# In[1]:


import string

import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm
import math
import preprocessor as p
import numpy as np


# # Load Data
# 
# - Read in the data and then shuffle

# In[2]:


file_num = 2
df = pd.read_csv(f"{file_num}.csv",keep_default_na=False)
df = df[~df['reviewText'].str.contains("\.jpg|\.png|\.jpeg|\.tiff|\.gif|\.bmp|\.heif", regex=True, na=False)]
old = df.copy()
text = df.reviewText
num_entries = len(df)


# Initial Data cleaning
# 

# In[8]:


def tweet_clean(num_entries):
    for index, review in tqdm(enumerate(text[0:num_entries])):
        clean_text = p.clean(review)
        df.reviewText[index] = clean_text


# In[9]:




tweet_clean(num_entries)


# In[10]:


# Helper function to print tokens in a sentence 

def print_tokens(doc):
    for token in doc:
        token_text = token.text
        token_pos = token.pos_
        token_dep = token.dep_
        token_vec = token.vector
        print("{:<12}{:<10}{:<10}".format(token_text, token_pos, token_dep))
    print(token.vector)


# You need to download the spacy model with:
# 
# python -m spacy download en_core_web_md

# In[11]:


nlp = spacy.load("en_core_web_md")


# # Clean Text
# 
# 1. Pronouns are removed from the text - lemmtaisation
# 2. Then stopwords and punctuation are removed
# 3. Then the removal of spaces and numbers and undetected punctuation
# 4. Removal of URLS and Emails
# 5. The whole sentence is then spell checked and corrected

# In[12]:


def tokenizer(sentence):
    punctuation = string.punctuation
    stopwords = list(STOP_WORDS)
    filterTokens = [ word for word in sentence if word.lemma_ != "-PRON-" ]
    filterTokens = [ word for word in filterTokens if word.text not in stopwords and word.text not in punctuation ]
    filterTokens = [ word for word in filterTokens if word.pos_ not in ["SPACE","PUNCT","NUM"]]
    filterTokens = [ word for word in filterTokens if not word.like_email ]
    return filterTokens


# In[13]:


def clean_text(num_entries):
    for index, review in tqdm(enumerate(text[0:num_entries])):
        doc = nlp(review)
        clean_token = tokenizer(doc)
        sentence = ' '.join([x.text.lower() for x in clean_token])
        df.reviewText[index] = sentence
        


# Clean all the text and put into a list

# In[14]:


clean_text(num_entries)


# Save the cleaned data

# In[ ]:


df.to_csv(f'kindle_reviews_cleaned_{file_num}.csv', index=False)


# In[ ]:


# Just to test 

# twitter - 62 ops/sec - 3 hours
# spacy clean - 30 ops/sec - 8 hours

# 11 hours

# 6 threads = 11/6 = 1.86 hours

# df_2 = pd.read_csv("kindle_reviews_cleaned.csv")


# In[4]:



# def index_marks(nrows, chunk_size):
#     return range(chunk_size, math.ceil(nrows / chunk_size) * chunk_size, chunk_size)
# 
# 
# def split(dfm, chunk_size):
#     indices = index_marks(dfm.shape[0], chunk_size)
#     return np.split(dfm, indices)
# 
# chunks = split(df, 163765)


# In[6]:


# for inx, c in enumerate(chunks):
#     c.to_csv(f"data_split/{inx}.csv")


# In[ ]:





#!/usr/bin/env python
# coding: utf-8
import csv

import pandas as pd
import spacy
import string
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm
from spacymoji import Emoji





df = pd.read_csv(f"../kindle_reviews.csv")
df = df[df['reviewText'].notna()]
df = df[~df['reviewText'].str.contains(".jpg|.png|.jpeg|.tiff|.gif|.bmp|.heif", regex=True, na=False)]
text = df.reviewText
num_entries = 50 #len(df)

disable = ['vectors', 'textcat', 'ner', 'parser']
nlp = spacy.load("en_core_web_md", disable = disable)
nlp.tokenizer.rules = {key: value for key, value in nlp.tokenizer.rules.items() if "'" not in key and "’" not in key and "‘" not in key}
emoji = Emoji(nlp)
nlp.add_pipe(emoji, first=True)
columns= ['overall', 'verified', 'reviewTime', 'reviewerID', 'asin', 'style',
       'reviewerName', 'reviewText', 'summary', 'unixReviewTime', 'vote',
       'image']

columns = ['asin', 'helpful', 'overall', 'reviewText', 'reviewTime',
       'reviewerID', 'reviewerName', 'summary', 'unixReviewTime']

# # Clean Text
# 
# 1. Pronouns are removed from the text - lemmtaisation
# 2. Then stopwords and punctuation are removed
# 3. Then the removal of spaces and numbers and undetected punctuation
# 4. Removal of URLS and Emails
# 5. The whole sentence is then spell checked and corrected

punctuation = string.punctuation
def tokenizer(sentence):
    #filterTokens = [ word for word in sentence if word.lemma_ != "-PRON-"]
    #filterTokens = [word for word in filterTokens if word.pos_ not in STOP_WORDS]
    filterTokens = [ word for word in sentence if word.pos_ not in ["SPACE","NUM"]]
    filterTokens = [ word for word in filterTokens if not word.like_email]
    filterTokens = [word for word in filterTokens if not word.like_url]
    filterTokens = [word for word in filterTokens if not word._.is_emoji]
    return filterTokens


@profile
def clean_text(num_entries):

    with open('mycsvfile.csv', 'w') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for index, review in tqdm(enumerate(text[0:num_entries])):
            doc = nlp(review)
            clean_token = tokenizer(doc)
            sentence = ' '.join([x.text.lower() for x in clean_token])
            sentence = sentence.translate(str.maketrans('', '', punctuation))
            new_sentence = sentence.translate(str.maketrans('', '', '&;'))
            df.at[index, 'reviewText'] = new_sentence
            row = df.iloc[[index]]
            dict_to_add = {key: row[key].values[0] for key in columns}
            w.writerow(dict_to_add)

        




# Clean all the text and put into a list


clean_text(num_entries)


# Save the cleaned data

#df.to_csv(f'new_clean.csv', index=False)








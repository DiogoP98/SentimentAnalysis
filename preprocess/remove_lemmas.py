import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm
from nltk.stem import PorterStemmer

def clean():
    clean_text_3(num_entries)
    df.to_csv("cleaned_again_3.csv", index=False)


def clean_text_3(num_entries):
    for index, review in tqdm(enumerate(text[0:num_entries])):
        if type(review) is str:
            clean_token = remove_lemmas(list(review.split()))
            sentence = ' '.join(clean_token)
            df.reviewText[index] = sentence


def remove_lemmas(words):
    for inx,w in enumerate(words):
        words[inx] = ps.stem(w)
    return words


if __name__ == '__main__':
    ps = PorterStemmer()
    df = pd.read_csv("../cleaned_again_2.csv")
    text = df.reviewText
    num_entries = len(df)
    clean()

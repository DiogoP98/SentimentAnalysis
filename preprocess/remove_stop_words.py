import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm


def clean():
    clean_text_2(num_entries)
    df.to_csv("cleaned_again.csv", index=False)


def tokenizer_2(sentence):
    stopwords = list(STOP_WORDS)
    filterTokens = [word for word in sentence if word not in stopwords]
    return filterTokens


def clean_text_2(num_entries):
    for index, review in tqdm(enumerate(text[0:num_entries])):
        clean_token = tokenizer_2(list(review.split()))
        sentence = ' '.join(clean_token)
        df.reviewText[index] = sentence


if __name__ == '__main__':
    df = pd.read_csv("../kindle_reviews_cleaned.csv")
    text = df.reviewText
    num_entries = len(df)
    clean()

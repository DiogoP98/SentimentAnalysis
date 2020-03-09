import pandas as pd
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import string


def sample_dataset(file='kindle_reviews.csv'):
	df = pd.read_csv(file)
	df = df.sample(frac = 1, random_state=0)
	df = df.rename(columns={'Unnamed: 0': 'Id'})

	column_names = df.columns
	ratings = [1,2,3,4,5]
	sampled_dataset = pd.DataFrame(columns=column_names)

	for rating in ratings:
		books_rating = df.loc[df['overall'] == rating]
		books_rating = books_rating.sample(n=20000, random_state=0)
		sampled_dataset = sampled_dataset.append(books_rating)

	sampled_dataset = sampled_dataset.sample(frac = 1, random_state=0)
	return sampled_dataset

df = sample_dataset()
df.to_csv('kindle_reviews_sm.csv')

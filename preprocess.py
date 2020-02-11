import pandas as pd
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import string

df = pd.read_table('TripAdvisorUKRestaurant-max_MF.txt')

ratings = df.iloc[:,0]
reviews = df.iloc[:,1]

ratings_distribution = ratings.value_counts()

print(ratings_distribution)
print(len(reviews))

nlp = English()
tokenizer = Tokenizer(nlp.vocab)
	
table = str.maketrans('', '', string.punctuation)

for review in reviews:
	tokens = tokenizer(review)
	for token in tokens:
		print(token.orth_)
	stripped = [token.orth_ for token in tokens if not token.is_punct | token.is_space]
	for token in stripped:
		print(token)
	break
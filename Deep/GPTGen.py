import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import AML.Deep.utils as utils
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
import isbntools
import isbnlib
import pandas as pd
from tqdm import tqdm
logging.basicConfig(level=logging.INFO)
from AML.Deep.fine_tune import tokenize, split_data
model, tokenizer = utils.setup_model('GPT2',1)
#data = utils.get_data()
reviews = pd.read_csv("Datasets\\new_clean_sm.csv")
print(reviews.shape)
reviews.head()

##

reviews['summary'] = reviews['summary'].fillna('') #remove nan vals and replace them with ''
reviews['reviewText'] = reviews['reviewText'].fillna('')
reviews['reviewText'] = reviews['summary'] + ' ' + reviews['reviewText']
sums = reviews['summary'].values
isbns = reviews['asin'].values

# keeping only relevant columns and calculating sentence lengths
reviews = reviews[['reviewText', 'overall']]
reviews.columns = ['reviewText', 'overall']
reviews['review_length'] = reviews['reviewText'].apply(lambda x: len(x.split()))
reviews.head()
print(isbnlib.is_isbn10(isbns),isbnlib.is_isbn13(isbns))
isbn10 = []
isbn13 = []
for isbn in isbns:
    truth =(isbnlib.is_isbn10(isbn) or isbnlib.is_isbn13(isbn))
    if(truth) :
        isbn10.append(isbn)
        isbn13.append(isbn)
print(isbnlib.is_isbn10['1511435933'])
firstbook = isbnlib.meta(['1511435933'])
firstbook['title']
# Load pre-trained model tokenizer (vocabulary)


pre = 'my review of a Lamp.'
# Encode a text inputs
text = "Who was Jim Henson ? Jim Henson was a"
indexed_tokens = tokenizer.encode(text)

# Convert indexed tokens in a PyTorch tensor
tokens_tensor = torch.tensor([indexed_tokens])

# Load pre-trained model (weights)
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set the model in evaluation mode to deactivate the DropOut modules
# This is IMPORTANT to have reproducible results during evaluation!
model.eval()

# If you have a GPU, put everything on cuda
tokens_tensor = tokens_tensor.to('cuda')
model.to('cuda')

# Predict all tokens
with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]

# get the predicted next sub-word (in our case, the word 'man')
predicted_index = torch.argmax(predictions[0, -1, :]).item()
predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
assert predicted_text == 'Who was Jim Henson? Jim Henson was a man'
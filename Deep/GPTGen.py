import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import AML.Deep.utils as utils
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
from goodreads import client
import isbntools
import isbnlib
import numpy as np
import pandas as pd
import pickle
import time
from time import sleep
from goodreads import client
import threading
gc = client.GoodreadsClient('NvBjhVw5nHK7h1dVieSkA','mFbtG6YNnRjY2MW5mko4SBAqOoCg792R7hC3mFn4Y')
gc2 = client.GoodreadsClient('fMR7HGS1g3D0WiNKed2A','oE4J6Kx7fbfnkU86Ldq24pleIOMiwyStCVt0lBIxI')
from tqdm import tqdm
logging.basicConfig(level=logging.INFO)
from AML.Deep.fine_tune import tokenize, split_data
model, tokenizer = utils.setup_model('GPT2',1)
#data = utils.get_data()
reviews = pd.read_csv("Datasets\\new_clean_sm.csv")
print(reviews.shape)
reviews.head()
l = threading.Lock()
##
threads = 2
reviews['summary'] = reviews['summary'].fillna('') #remove nan vals and replace them with ''
reviews['reviewText'] = reviews['reviewText'].fillna('')
reviews['reviewText'] = reviews['summary'] + ' ' + reviews['reviewText']
sums = reviews['summary'].values
isbns = reviews['asin'].values.tolist()
np.array_split(isbns, threads)
print(len(isbns))
isbns = list(dict.fromkeys(isbns))#remove dups
print(len(isbns))
# keeping only relevant columns and calculating sentence lengths
reviews = reviews[['reviewText', 'overall']]
reviews.columns = ['reviewText', 'overall']
reviews['review_length'] = reviews['reviewText'].apply(lambda x: len(x.split()))
reviews.head()
import threading
print(isbnlib.is_isbn10(isbns),isbnlib.is_isbn13(isbns))
#b= gc.search_books(['B00BN0T8ZO'])[0].title.tolist()

try:
    isbntitle = pickle.load(open( "isbn_title.pkl", "rb" ) )
except:
    print('does not exist')



i = 0
for isbn in isbns:
    if isbn not in isbntitle:
        try:
            title = client.search_books(isbn)[0].title
            l.acquire()
            isbntitle[isbn] = title
            l.release()
        except:
            print('fail')
        #sleep(1)
    if(i%10==0):
        print('saving books')
        f = open(f"isbn_title{num}.pkl","wb")
        pickle.dump(isbntitle,f)
        f.close()
    i+=1


isbntitle = {
}
isbn10 = []
isbn13 = []





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
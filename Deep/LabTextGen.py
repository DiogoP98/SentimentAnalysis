from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch
from torch.nn import functional as F
from torch import optim
import random
import sys
import numpy as np
import pandas as pd
from numpy import save
import AML.Deep.utils as utils
import io

# text = io.open('Datasets\\new_clean_sm.csv', encoding='utf-8').read().lower()
df = pd.read_csv("Datasets\\new_clean_sm.csv")

print(df.shape)
temp = df[['reviewText', 'overall']]
ones = temp.loc[temp['overall'].isin([1])]
twos = temp.loc[temp['overall'].isin([2])]
threes = temp.loc[temp['overall'].isin([3])]
fours = temp.loc[temp['overall'].isin([4])]
fives = temp.loc[temp['overall'].isin([5])]
ones = ones['reviewText'].values
twos = twos['reviewText'].values
threes = threes['reviewText'].values
fours = fours['reviewText'].values
fives = fives['reviewText'].values
maxlen = 40
step = 3
try:
    ones = np.load(r'Datasets\ones.npy', allow_pickle=True)
    twos = np.load(r'Datasets\twos.npy', allow_pickle=True)
    threes = np.load(r'Datasets\threes.npy', allow_pickle=True)
    fours = np.load(r'Datasets\fours.npy', allow_pickle=True)
    fives = np.load(r'Datasets\fives.npy', allow_pickle=True)
except:
    save(r'Datasets\ones.npy', ones)
    save(r'Datasets\twos.npy', twos)
    save(r'Datasets\threes.npy', threes)
    save(r'Datasets\fours.npy', fours)
    save(r'Datasets\fives.npy', fives)

try:
    oneswords =  np.load(r'Datasets\oneswords.npy'   , allow_pickle=True)
    twoswords =  np.load(r'Datasets\twoswords.npy'   , allow_pickle=True)
    threeswords= np.load(r'Datasets\threeswords.npy' , allow_pickle=True)
    fourswords = np.load(r'Datasets\fourswords.npy'  , allow_pickle=True)
    fiveswords = np.load(r'Datasets\fiveswords.npy'  , allow_pickle=True)
except:
    oneswords = ''
    twoswords= ''
    threeswords= ''
    fourswords= ''
    fiveswords= ''
    print(ones[0])
    for reviews in ones:
        oneswords=  oneswords+str(reviews)
    oneswords = oneswords.split(' ')
    for reviews in twos:
        twoswords= twoswords+str(reviews)
    twoswords = twoswords.split(' ')
    for reviews in threes:
        threeswords=threeswords+str(reviews)
    threeswords = threeswords.split(' ')
    for reviews in fours:
        fourswords=fourswords+str(reviews)
    fourswords = fourswords.split(' ')
    for reviews in fives:
        fiveswords=  fiveswords+str(reviews)
    fiveswords = fiveswords.split(' ')

    save(r'Datasets\oneswords.npy'  , oneswords)
    save(r'Datasets\twoswords.npy'  , twoswords)
    save(r'Datasets\threeswords.npy', threeswords)
    save(r'Datasets\fourswords.npy' , fourswords)
    save(r'Datasets\fiveswords.npy' , fiveswords)
text = oneswords
chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


def encode(inp):
    # encode the characters in a tensor
    x = torch.zeros(maxlen, dtype=torch.long)
    for t, char in enumerate(inp):
        x[t] = char_indices[char]
    return x


def decode(ten):
    s = ''
    for v in ten:
        s += indices_char[v]
    return s


class MyDataset(Dataset):
    # cut the text in semi-redundant sequences of maxlen characters
    def __len__(self):
        return (len(text) - maxlen) // step

    def __getitem__(self, i):
        inp = text[i*step: i*step + maxlen]
        out = text[i*step + maxlen]

        x = encode(inp)
        y = char_indices[out]

        return x, y


class CharPredictor(nn.Module):
    def __init__(self):
        super(CharPredictor, self).__init__()
        self.emb = nn.Embedding(len(chars), 8)
        self.lstm = nn.LSTM(8, 128, batch_first=True)
        self.lin = nn.Linear(128, len(chars))

    def forward(self, x):
        x = self.emb(x)
        lstm_out, _ = self.lstm(x)
        out = self.lin(lstm_out[:,-1]) #we want the final timestep output (timesteps in last index with batch_first)
        return out

def sample(logits, temperature=1.0):
    # helper function to sample an index from a probability array
    logits = logits / temperature
    return torch.multinomial(F.softmax(logits, dim=0), 1)


import torchbearer
from torchbearer import Trial
from torchbearer.callbacks.decorators import on_end_epoch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

@on_end_epoch
def create_samples(state):
    with torch.no_grad():
        epoch = -1
        if state is not None:
            epoch = state[torchbearer.EPOCH]

        print()
        print('----- Generating text after Epoch: %d' % epoch)

        start_index = random.randint(0, len(text) - maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print()
            print()
            print('----- diversity:', diversity)

            generated = ''
            sentence = text[start_index:start_index + maxlen - 1]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            print()
            sys.stdout.write(generated)

            inputs = encode(sentence).unsqueeze(0).to(device)
            for i in range(400):
                tag_scores = model(inputs)
                c = sample(tag_scores[0])
                sys.stdout.write(indices_char[c.item()])
                sys.stdout.flush()
                inputs[0, 0:inputs.shape[1] - 1] = inputs[0, 1:]
                inputs[0, inputs.shape[1] - 1] = c
        print()


data = MyDataset()
loader = DataLoader(data, batch_size=128)
loss_function = nn.CrossEntropyLoss()
model = CharPredictor()
print(model.parameters())

optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
trial = Trial(model, optimizer, metrics=['loss', 'accuracy'], callbacks=[create_samples])
trial.run(epochs=10)

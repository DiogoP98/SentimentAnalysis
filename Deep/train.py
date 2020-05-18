import spacy
import torchtext
import torchbearer
import torch
import dataframe_dataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torchbearer import Trial
from tqdm import tqdm

import utils
import LSTM

spacy.load('en')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.optim as optim

class Iter:
    def __init__(self, it):
        self.it = it
    def __iter__(self):
        for batch in self.it:
            yield (batch.text, batch.label.unsqueeze(1))
    def __len__(self):
        return len(self.it)

def pre_process():
    TEXT = torchtext.data.Field(tokenize='spacy', lower=True, include_lengths=True)
    LABEL = torchtext.data.LabelField(dtype=torch.float)

    df, num_classes = utils.get_data()

    train, test = train_test_split(df, test_size=0.2)
    val, test = train_test_split(test, test_size=0.5)

    print("Splitting the data")
    train_data, val_data, test_data = dataframe_dataset.DataFrameDataset.splits(
        text_field=TEXT, label_field=LABEL, train_df=train, val_df=val, test_df=test)
    print("Finised Splitting the data")

    BATCH_SIZE = 64

    train_iterator, valid_iterator, test_iterator = torchtext.data.BucketIterator.splits(
        (train_data, val_data, test_data), 
        batch_size=BATCH_SIZE,
        device=device,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True)
    
    TEXT.build_vocab(train_data, max_size=25000)
    LABEL.build_vocab(train_data)

    print(LABEL.vocab.stoi)

    return len(TEXT.vocab), Iter(train_iterator), Iter(valid_iterator), Iter(test_iterator)

def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)

def train(model, model_path, selected_model, train_loader, val_loader, test_loader):
    print("Started Training")
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 5

    # torchbearer_trial2 = Trial(model, optimizer, criterion, metrics=['acc', 'loss']).to(device)
    # torchbearer_trial2.with_generators(train_generator=train_loader, val_generator=val_loader, test_generator=test_loader)
    # torchbearer_trial2.run(epochs=num_epochs)

    for epoch in range(num_epochs):
        with tqdm(train_loader, total=len(train_loader), desc='train', position=0, leave=True) as t:
            for (inputs, lengths), labels in train_loader:
                labels = labels.squeeze().long()
                #labels.requires_grad = True
                optimizer.zero_grad()

                logits = model(inputs, lengths)
                loss = criterion(logits, labels)
                loss.backward()
                print(logits)
                running_accuracy = utils.accuracy(labels, logits)
                t.set_postfix(accuracy='{:05.3f}'.format(running_accuracy), loss='{:05.3f}'.format(loss))
                t.update()
                optimizer.step()

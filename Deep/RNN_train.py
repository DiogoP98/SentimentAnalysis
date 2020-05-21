import spacy
import torchtext
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim

import utils
import RNN
import os
import dataframe_dataset

spacy.load('en')
device = "cuda:0" if torch.cuda.is_available() else "cpu"

class Iter:
    def __init__(self, it):
        self.it = it
    def __iter__(self):
        for batch in self.it:
            yield (batch.text, batch.label.unsqueeze(1))
    def __len__(self):
        return len(self.it)

def pre_process(three_class_problem):
    TEXT = torchtext.data.Field(tokenize='spacy', lower=True, include_lengths=True)
    LABEL = torchtext.data.LabelField(dtype=torch.float)

    df, num_classes = utils.get_data()

    if three_class_problem:
        df, num_classes = utils.three_class_problem(df)

    train, test = train_test_split(df, test_size=0.2)
    val, test = train_test_split(test, test_size=0.5)

    train_data, val_data, test_data = dataframe_dataset.DataFrameDataset.splits(
        text_field=TEXT, label_field=LABEL, train_df=train, val_df=val, test_df=test)

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

    return len(TEXT.vocab), Iter(train_iterator), Iter(valid_iterator), Iter(test_iterator), num_classes

def train(model, model_path, selected_model, class_problem, train_loader, val_loader):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    num_epochs = 10

    scheduler =optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        running_accuracy = 0
        running_loss = 0
        count = 0

        with tqdm(train_loader, total=len(train_loader), desc='train', position=0, leave=True) as t:
            for (inputs, lengths), labels in train_loader:
                labels = labels.squeeze().long()
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                logits = model(inputs, lengths)
                loss = criterion(logits, labels)
                loss.backward()

                running_loss += loss.item()
                running_accuracy += utils.accuracy(labels, logits)
                count += 1

                t.set_postfix(accuracy='{:05.3f}'.format(running_accuracy/count), loss='{:05.3f}'.format(running_loss/count))
                t.update()
                optimizer.step()
        scheduler.step()

        validation(model, val_loader)
    
    torch.save({
            'model_state_dict': model.state_dict(),
    }, model_path + selected_model + str(class_problem) + ".pth")

def validation(model, val_loader):
    model.eval()
    count = 0
    running_accuracy = 0

    with torch.no_grad():
        with tqdm(val_loader, total=len(val_loader), desc='val', position=0, leave=True) as t:
            for (inputs, lengths), labels in val_loader:
                labels = labels.squeeze().long()
                inputs, labels = inputs.to(device), labels.to(device)

                logits = model(inputs, lengths)
                running_accuracy += utils.accuracy(labels, logits)
                count += 1
                t.set_postfix(accuracy='{:05.3f}'.format(running_accuracy/count))
                t.update()

def test(model, saving_path, selected_model, class_problem, test_loader):
    if os.path.exists(saving_path + selected_model + str(class_problem) + '.pth'):
        checkpoint = torch.load(saving_path + selected_model + str(class_problem) + ".pth", map_location=device)
    else:
        raise ValueError('No file with the pretrained model selected')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    running_accuracy = 0
    count = 0

    with torch.no_grad():
        for (inputs, lengths), labels in tqdm(test_loader, total=len(test_loader)):
            labels = labels.squeeze().long()
            inputs, labels = inputs.to(device), labels.to(device)

            logits = model(inputs, lengths)
            running_accuracy += utils.accuracy(labels, logits)
            count += 1
    
    print(f"Test accuracy: {running_accuracy/count}")

    
 

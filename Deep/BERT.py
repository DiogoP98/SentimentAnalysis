import transformers
import torch
import pandas as pd
from torch.utils.data import TensorDataset, random_split, DataLoader
import numpy as np 
import progressbar
import time
import datetime

def getData():
    df = pd.read_csv('../kindle_reviews.csv', keep_default_na=False)
    df = df.rename(columns={'Unnamed: 0': 'Id'})
    
    return df

def tokenize(df):
    print("Started Tokenizer")
    print("-----------------")
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)

    ids = []
    masks = []
    labels = []

    dfsize = len(df)
    barsize = int(dfsize/100000)

    bar = progressbar.ProgressBar(maxval=barsize, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    curr_sample = 0

    for review, label in zip(df['reviewText'], df['overall']):
        if dfsize%100000 == 0:
            bar.update(curr_sample+1)
            curr_sample += 1
        encoded = tokenizer.encode_plus(review, add_special_tokens= True,
                            max_length = 512, pad_to_max_length = True,
                            return_attention_mask= True, return_tensors='pt')
        
        ids.append(encoded['input_ids'])
        masks.append(encoded['attention_mask'])
        labels.append(label - 1)
    
    bar.update(barsize)
    bar.finish()

    ids = torch.cat(ids)
    masks = torch.cat(masks)
    labels = torch.tensor(labels)

    print("-----------------")
    print("Finished Tokenizer\n\n")
    return ids, masks, labels

def split_data(ids, masks, labels):
    dataset = TensorDataset(ids, masks, labels)
    size = len(dataset)

    train_size = int(0.8*size)
    val_test_size = int(0.1*size)
    train, val, test = random_split(dataset, [train_size, val_test_size, val_test_size])

    train_data = DataLoader(train, batch_size=128, shuffle=True)
    val_data = DataLoader(val, batch_size=128, shuffle=True)
    test_data = DataLoader(test, batch_size=128, shuffle=True)

    return train_data, val_data, test_data

def accuracy(labels, predictions):
    predictions = np.argmax(predictions, axis=1).flatten()
    labels = labels.flatten()
    size = len(labels)

    return np.sum(predictions == labels) / size

def training(train_data, val_data, test_data):
    print("Started Training")
    print("-----------------")
    model = transformers.BertForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels = 5,output_attentions = False, output_hidden_states = False)
    
    #Adam optimizer with weight decay fix
    optimizer = transformers.AdamW(model.parameters(), lr = 5e-5, eps = 1e-8)
    epochs = 2
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = len(train_data) * epochs)
    
    bar = progressbar.ProgressBar(maxval=len(train_data), \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    for epoch in range(epochs):
        total = time.time()
        print("Epoch " + str(epoch + 1) + " progress:")
        bar.start()
        train_loss = 0
        curr_batch = 0
        model.train()

        for step, batch in enumerate(train_data):
            bar.update(curr_batch+1)
            curr_batch += 1
            model.zero_grad()
            loss, logits = model(batch[0], token_type_ids=None,
                            attention_mask=batch[1], 
                            labels=batch[2])
            train_loss += loss.item()
            loss.backward()

            #prevents "exploding gradients" problem
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
    
        total = time.time() - total
        print("Elapsed Time: " + str(datetime.timedelta(seconds=int(round((total))))))
        print("Average training loss: " + str(train_loss/len(train_data)))
        bar.finish()

        model.eval()

        val_acc = 0
        val_loss = 0

        for batch in val_data:
            while(torch.no_grad()):
                loss, logits = model(batch[0], token_type_ids=None,
                                attention_mask=batch[1], 
                                labels=batch[2])
            
            val_loss += loss.item()
            val_acc += accuracy(batch[2], logits)
        
        print("Average validation loss: " + str(val_loss/len(val_data)))
        print("Average validation accuracy: " + str(val_acc/len(val_data)))

    print("-----------------")
    print("Ended Training")

    print("\nStarted Testing")
    print("-----------------")

    model.eval()

    test_acc = 0

    for batch in test_data:
        while(torch.no_grad()):
            _, logits = model(batch[0], token_type_ids=None,
                            attention_mask=batch[1], 
                            labels=batch[2])
        
        test_acc += accuracy(batch[2], logits)

    print("Average test accuracy: " + str(val_acc/len(test_data)))
    print("-----------------")
    print("Ended Testing")


if __name__ == "__main__":
    df = getData()
    ids, masks, labels = tokenize(df)
    train_data, val_data, test_data = split_data(ids, masks, labels)
    training(train_data, val_data, test_data)

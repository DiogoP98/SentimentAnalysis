import transformers
import torch
import pandas as pd
from torch.utils.data import TensorDataset, random_split, DataLoader
import numpy as np 
from progress.bar import IncrementalBar
import time
import datetime
from sklearn.metrics import matthews_corrcoef

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def getData():
    df = pd.read_csv('../new_clean_sm.csv', keep_default_na=False)
    df = df[df['reviewText'].notna()]
    df = df.rename(columns={'Unnamed: 0': 'Id'})

    return df

def tokenize(df):
    print("**Started Tokenizer**")
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)

    ids = []
    masks = []
    labels = []

    bar = IncrementalBar('Review', max = len(df))

    for review, label in zip(df['reviewText'], df['overall']):
        bar.next()
        encoded = tokenizer.encode_plus(review, add_special_tokens= True,
                            max_length = 512, pad_to_max_length = True,
                            return_attention_mask= True, return_tensors='pt')
        
        ids.append(encoded['input_ids'])
        masks.append(encoded['attention_mask'])
        labels.append(label)

    bar.finish()

    ids = torch.cat(ids)
    masks = torch.cat(masks)
    labels = torch.tensor(labels)

    print("**Finished Tokenizer**\n\n")
    return ids, masks, labels

def split_data(ids, masks, labels):
    dataset = TensorDataset(ids, masks, labels)
    size = len(dataset)

    train_size = int(0.8*size)
    val_size = int(0.1*size)
    test_size = size - train_size - val_size
    train, val, test = random_split(dataset, [train_size, val_size, test_size])

    train_data = DataLoader(train, batch_size=50, shuffle=True)
    val_data = DataLoader(val, batch_size=50, shuffle=True)
    test_data = DataLoader(test, batch_size=50, shuffle=True)

    return train_data, val_data, test_data

def accuracy(labels, predictions):
    predictions = np.argmax(predictions, axis=1).flatten()
    labels = labels.flatten()
    size = len(labels)

    return np.sum(predictions == labels) / size

def mcc(labels, predictions):
    labels = np.concatenate(labels, axis=0)

    predictions = np.concatenate(predictions, axis=0)
    predictions = np.argmax(predictions, axis=1).flatten()    

    return matthews_corrcoef(labels, predictions)


def training(train_data, val_data, test_data):
    print("**Started Training**")
    print("Training using " + device)
    
    model = transformers.BertForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels = 6,output_attentions = False, output_hidden_states = False)
    model = model.to(device)
    
    #Adam optimizer with weight decay fix
    optimizer = transformers.AdamW(model.parameters(), lr = 5e-5, eps = 1e-8)
    epochs = 2
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = len(train_data) * epochs)

    for epoch in range(epochs):
        total = time.time()
        print("\nEpoch " + str(epoch + 1) + "/" + str(epochs))
        train_loss = 0
        curr_batch = 0
        model.train()

        bar = IncrementalBar('Batch', max = len(train_data))
        batch_n = 0

        for step, batch in enumerate(train_data):
            batch_n += 1
            if batch_n % 10 == 0:
                torch.save(model.state_dict(), 'Bert_trained.pth')
            bar.next()
            model.zero_grad()

            loss, logits = model(batch[0].to(device), token_type_ids=None,
                            attention_mask=batch[1].to(device), labels=batch[2].to(device))
            train_loss += loss.item()
            loss.backward()

            #prevents "exploding gradients" problem
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
        
        total = time.time() - total
        print("\nElapsed Time: " + str(datetime.timedelta(seconds=int(round((total))))))
        print("Average training loss: " + str(train_loss/len(train_data)))
        bar.finish()

        model.eval()

        val_acc = 0
        val_loss = 0

        print("\n**Running Validation batches**")
        bar = IncrementalBar('Batch', max = len(val_data))
        for batch in val_data:
            bar.next()

            with torch.no_grad():
                loss, logits = model(batch[0].to(device), token_type_ids=None,
                                attention_mask=batch[1].to(device), 
                                labels=batch[2].to(device))
            
            logits = logits.detach().cpu().numpy()
            val_loss += loss.item()
            val_acc += accuracy(batch[2].numpy(), logits)
        
        bar.finish()
        print("Average validation loss: " + str(val_loss/len(val_data)))
        print("Average validation accuracy: " + str(val_acc/len(val_data)))

    print("**Ended Training**\n")
    torch.save(model.state_dict(), 'Bert_trained.pth')

    print("\n**Started Testing**")
    print("-----------------")

    model.eval()

    test_acc = 0

    true_labels = []
    predicted_labels = []

    for batch in test_data:
        with torch.no_grad():
            output = model(batch[0].to(device), token_type_ids=None,
                            attention_mask=batch[1].to(device))
        
        logits = output[0]
        logits = logits.detach().cpu().numpy()

        true_labels.append(batch[2].numpy())
        predicted_labels.append(logits)

        test_acc += accuracy(batch[2].numpy(), logits)

    print("Test Matthews correlation coefficient: " + str(mcc(true_labels, predicted_labels)))
    print("Average test accuracy: " + str(test_acc/len(test_data)))
    print("**Ended Testing**")


if __name__ == "__main__":
    df = getData()
    ids, masks, labels = tokenize(df)
    train_data, val_data, test_data = split_data(ids, masks, labels)
    training(train_data, val_data, test_data)

import numpy as np
from sklearn.metrics import matthews_corrcoef
import pandas as pd
import torch

def checkpoint(model, optimizer, scheduler, selected_model):
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
    }, selected_model+"_finetuned.pth")

def get_data():
    df = pd.read_csv('../new_clean_sm_100000.csv', keep_default_na=False)
    df = df[df['reviewText'].notna()]
    df = df.rename(columns={'Unnamed: 0': 'Id'})

    return df

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

def setup_seeds():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True

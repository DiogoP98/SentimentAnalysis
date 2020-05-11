import numpy as np
from sklearn.metrics import matthews_corrcoef
import pandas as pd
import torch
from transformers import XLNetModel, XLNetTokenizer, XLNetForSequenceClassification
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
import argparse
import os
import platform

device = "cuda:0" if torch.cuda.is_available() else "cpu"
num_classes = 5
dir_path = os.path.dirname(os.path.realpath(__file__))

if platform.system() == 'Linux' or platform.system() == 'Darwin':
    dir_path += '/'
else:
    dir_path +=  '\\'

def setup_BERT():
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=num_classes, output_attentions = False, output_hidden_states = False)
    #model.classifier = nn.Linear(768, num_classes)

    #for param in model.parameters():
    #    param.requires_grad = False
    
    #model.classifier.weight.requires_grad = True #unfreeze last layer weights
    #model.classifier.bias.requires_grad = True #unfreeze last layer biases
    model = model.to(device)

    return model, tokenizer

def setup_XLNet():
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
    model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=num_classes, output_attentions = False, output_hidden_states = False)
    #model.logits_proj = nn.Linear(768, num_classes)

    #for param in model.parameters():
    #    param.requires_grad = False
    #model.logits_proj.weight.requires_grad = True #unfreeze last layer weights
    #model.logits_proj.bias.requires_grad = True #unfreeze last layer biases
    model = model.to(device)

    return model, tokenizer

def checkpoint(model, optimizer,scheduler, epoch, batch_num ,selected_model, save_path):
    torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'batch_num': batch_num
    }, save_path + selected_model+"_finetuned.pth")

def get_data():
    df = pd.read_csv(dir_path + '../new_clean_sm_100000.csv', keep_default_na=False)
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

def arg_parser():
    parser = argparse.ArgumentParser(description='Check checkpoints')
    parser.add_argument("--m", choices=["BERT", "XLNET"], required=True, type=str, help="Model")
    parser.add_argument("--c", choices=[True, False], required=False, type=bool, default=False, help="Checkpoints")
    parser.add_argument("--d", required=False, type=str, default=dir_path, help="DataLoader path")
    parser.add_argument("--mp", required=False, type=str, default=dir_path, help="Model path")
    args = parser.parse_args()

    return args.m, args.c, args.d, args.mp

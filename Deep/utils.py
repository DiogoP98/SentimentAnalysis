import numpy as np
from sklearn.metrics import matthews_corrcoef
import pandas as pd
import torch
from transformers import XLNetModel, XLNetTokenizer, XLNetForSequenceClassification
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
from transformers import RobertaModel, RobertaTokenizer, RobertaForSequenceClassification
from transformers import XLMModel, XLMTokenizer, XLMForSequenceClassification
import argparse
import os
import platform

device = "cuda:0" if torch.cuda.is_available() else "cpu"
<<<<<<< HEAD
num_classes = 3
=======
>>>>>>> 6d24b921ddb57b30af32d02ef75cf25630d2c3f2
dir_path = os.path.dirname(os.path.realpath(__file__))

if platform.system() == 'Linux' or platform.system() == 'Darwin':
    dir_path += '/'
else:
    dir_path +=  '\\'

def setup_BERT(num_classes):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=num_classes, output_attentions = False, output_hidden_states = False)
    #model.classifier = nn.Linear(768, num_classes)

    #for param in model.parameters():
    #    param.requires_grad = False
    
    #model.classifier.weight.requires_grad = True #unfreeze last layer weights
    #model.classifier.bias.requires_grad = True #unfreeze last layer biases
    model = model.to(device)

    return model, tokenizer

def setup_XLNet(num_classes):
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
    model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=num_classes, output_attentions = False, output_hidden_states = False)
    #model.logits_proj = nn.Linear(768, num_classes)

    #for param in model.parameters():
    #    param.requires_grad = False
    #model.logits_proj.weight.requires_grad = True #unfreeze last layer weights
    #model.logits_proj.bias.requires_grad = True #unfreeze last layer biases
    model = model.to(device)

    return model, tokenizer

def setup_Roberta(num_classes):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_classes, output_attentions = False, output_hidden_states = False)
    model = model.to(device)

    return model, tokenizer

def setup_XLM(num_classes):
    tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-enfr-1024', do_lower_case=True)
    model = XLMForSequenceClassification.from_pretrained('xlm-mlm-enfr-1024', num_labels=num_classes, output_attentions = False, output_hidden_states = False)
    model = model.to(device)

    return model, tokenizer

def checkpoint(model, optimizer,scheduler, epoch, batch_num ,selected_model, save_path, class_problem):
    torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'batch_num': batch_num
    }, save_path + selected_model + "_finetuned_" + class_problem + ".pth")

def get_data():
    df = pd.read_csv(dir_path + '../3_class.csv', keep_default_na=False)
    df = df[df['reviewText'].notna()]
    df = df.rename(columns={'Unnamed: 0': 'Id'})

    return df,5

def three_class_problem(df):
    df = df[df['overall'] != 2]
    df = df[df['overall'] != 4]
    df.loc[df['overall'] == 3, 'overall'] = 2
    df.loc[df['overall'] == 5, 'overall'] = 3

    return df,3

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
    parser.add_argument("--m", choices=["BERT", "XLNET", "ROBERTA", "XLM"], required=True, type=str, help="Model")
    parser.add_argument("--c", action='store_true', help="Use previous checkpoints")
    parser.add_argument("--d", required=False, type=str, default=dir_path, help="DataLoader path")
    parser.add_argument("--mp", required=False, type=str, default=dir_path, help="Model path")
    parser.add_argument("--tcp", action='store_true', help="Three Class Problem")
    args = parser.parse_args()


    return args.m.upper(), args.c, args.d, args.mp, args.tcp

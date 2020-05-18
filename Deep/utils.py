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

dir_path = os.path.dirname(os.path.realpath(__file__))

if platform.system() == 'Linux' or platform.system() == 'Darwin':
    dir_path += '/'
else:
    dir_path +=  '\\'

def setup_model(selected_model, num_classes):
    if selected_model == "BERT":
        model, tokenizer = setup_BERT(num_classes)
    elif selected_model == "XLNET":
        model, tokenizer = setup_XLNet(num_classes)
    elif selected_model == "ROBERTA":
        model, tokenizer = setup_Roberta(num_classes)

    return model, tokenizer

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

def checkpoint(model, optimizer,scheduler, epoch, batch_num ,selected_model, save_path, class_problem):
    torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'batch_num': batch_num
    }, save_path + selected_model + "_finetuned_" + class_problem + ".pth")

def load_checkpoint(model, optimizer, scheduler, selected_model, model_path, class_problem):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    filename = model_path + selected_model + "_finetuned_" + class_problem + ".pth"
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        batch_num = checkpoint['batch_num']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, scheduler, start_epoch, batch_num

def get_data():
    df = pd.read_csv(dir_path + '../new_clean_sm_100000.csv', keep_default_na=False)
    df = df[df['reviewText'].notna()]
    df = df.rename(columns={'Unnamed: 0': 'Id'})

    df = df[:2000]
    return df,5

def three_class_problem(df):
    df = df[df['overall'] != 2]
    df = df[df['overall'] != 4]
    df.loc[df['overall'] == 3, 'overall'] = 2
    df.loc[df['overall'] == 5, 'overall'] = 3

    return df,3

def accuracy(labels, predictions):
    predictions = predictions.detach().numpy()
    predictions = np.argmax(predictions, axis=1).flatten()
    labels = labels.flatten()
    size = len(labels)

    return np.sum(predictions == labels) / size

def mcc(labels, predictions):
    labels = np.concatenate(labels, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    predictions = np.argmax(predictions, axis=1).flatten()    

    return matthews_corrcoef(labels, predictions)

def setup_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def arg_parser():
    parser = argparse.ArgumentParser(description='Check checkpoints')
    parser.add_argument("--m", choices=["BERT", "XLNET", "ROBERTA", "LSTM"], required=True, type=str, help="Model")
    parser.add_argument("--c", action='store_true', help="Use previous checkpoints")
    parser.add_argument("--d", required=False, type=str, default=dir_path, help="DataLoader path")
    parser.add_argument("--mp", required=False, type=str, default=dir_path, help="Model path")
    parser.add_argument("--tcp", action='store_true', help="Three Class Problem")
    parser.add_argument("--t", action='store_true', help="Test Mode")
    args = parser.parse_args()


    return args.m.upper(), args.c, args.d, args.mp, args.tcp, args.t

import transformers
import torch
import utils
from torch.utils.data import TensorDataset, random_split, DataLoader
import numpy as np 
from progress.bar import IncrementalBar
import time
import datetime
from transformers import XLNetModel, XLNetTokenizer, XLNetForSequenceClassification
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
from torch import nn
import sys
import os
from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"
num_classes = 5


# SET model_path and dataloader_path to None if training from scratch

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

    for param in model.parameters():
        param.requires_grad = False
    model.logits_proj.weight.requires_grad = True #unfreeze last layer weights
    model.logits_proj.bias.requires_grad = True #unfreeze last layer biases
    model = model.to(device)

    return model, tokenizer


def tokenize(df, tokenizer):
    ids = []
    masks = []
    labels = []
    bar = IncrementalBar('Review', max = len(df))

    print("**Started Tokenizer**")

    for review, label in zip(df['reviewText'], df['overall']):
        bar.next()
        encoded = tokenizer.encode_plus(review, add_special_tokens= True,
                            max_length = 512, pad_to_max_length = True,
                            return_attention_mask= True, return_tensors='pt')
        
        ids.append(encoded['input_ids'])
        masks.append(encoded['attention_mask'])
        #labels: 1-5 --> 0-4
        labels.append(label - 1)

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
    test_size = size - (train_size + val_size)
    train, val, test = random_split(dataset, [train_size, val_size, test_size])

    #Review length             Maximum Batch Size
    #    64                            64
    #    128                           32
    #    256                           16
    #    320                           14
    #    384                           12
    #    512                           6

    batch_size = 5 
    train_data = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_data = DataLoader(val, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(test, batch_size=batch_size, shuffle=True)

    return train_data, val_data, test_data

def fine_tune(model, train_data, val_data, selected_model, use_checkpoint):
    print("**Started Fine Tune**")
    print("Using " + device)
    epoch = 0



    #Adam optimizer with weight decay fix
    optimizer = transformers.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = 5e-5, eps = 1e-8)
    epochs = 2
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = len(train_data) * epochs)

    start_epoch = 0
    if use_checkpoint:
        model, optimizer, scheduler, start_epoch, batch_num = load_checkpoint(model, optimizer, scheduler, selected_model)
        model = model.to(device)
        # now individually transfer the optimizer parts...
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        # reload the dataloader
        train_data = torch.load(dir_path + "train_dataloader.pth")
        val_data = torch.load(dir_path + "val_dataloader.pth")
       

    for epoch in range(start_epoch, epochs):
        if epoch > start_epoch and use_checkpoint == True :
            use_checkpoint = False
        total = time.time()
        print("\nEpoch " + str(epoch + 1) + "/" + str(epochs))
        train_loss = 0
        curr_batch = 0
        model.train()

        # bar = IncrementalBar('Batch', max = len(train_data))
        batch_n = 0

        for step, batch in enumerate(tqdm(train_data)):
            if use_checkpoint:
                if step < batch_num:
                    continue
            batch_n += 1
            if batch_n % 100 == 0:
                utils.checkpoint_2(model, optimizer, scheduler, epoch, batch_n, selected_model, model_path)
                # save the dataloader
                
                torch.save(train_data, dir_path + 'train_dataloader.pth')
                torch.save(val_data, dir_path + 'val_dataloader.pth')


            #bar.next()
            optimizer.zero_grad()

            loss, logits = model(batch[0].to(device), token_type_ids=None,
                            attention_mask=batch[1].to(device), labels=batch[2].to(device=device, dtype=torch.int64))
            train_loss += loss.item()
            loss.backward()

            #prevents "exploding gradients" problem
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
        
        total = time.time() - total
        print("\nElapsed Time: " + str(datetime.timedelta(seconds=int(round((total))))))
        print("Average loss: " + str(train_loss/len(train_data)))
        #bar.finish()

        run_validation(model, val_data)
    
    print("**Ended Fine Tune**\n")
    utils.checkpoint_2(model, optimizer, scheduler, epoch, batch_n ,selected_model, model_path)

def run_validation(model, val_data):
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
                                labels=batch[2].to(device=device, dtype=torch.int64))
            
            logits = logits.detach().cpu().numpy()
            val_loss += loss.item()
            val_acc += utils.accuracy(batch[2].numpy(), logits)
        
    bar.finish()
    print("Average validation loss: " + str(val_loss/len(val_data)))
    print("Average validation accuracy: " + str(val_acc/len(val_data)))


def testing(test_data, selected_model):
    checkpoint = torch.load(selected_model+"_finetuned.pth", map_location=device)

    if selected_model == "BERT":
        model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=num_classes, output_attentions = False, output_hidden_states = False)
    else:
        model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=num_classes, output_attentions = False, output_hidden_states = False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("\n**Started Testing**")
    print("-----------------")

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

        test_acc += utils.accuracy(batch[2].numpy(), logits)

    print("Test Matthews correlation coefficient: " + str(utils.mcc(true_labels, predicted_labels)))
    print("Average test accuracy: " + str(test_acc/len(test_data)))
    print("**Ended Testing**")



def load_checkpoint(model, optimizer, scheduler, filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    filename = dir_path + filename + "_finetuned.pth"
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


if __name__ == "__main__":

    dataloader_path = 'H:\\AdvancedMachineLearningCW\\Deep\\'
    model_path = 'H:\\AdvancedMachineLearningCW\\Deep\\'
    use_start_dataloader = True
    use_checkpoint = True
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "\\"

    
    arg = sys.argv
    if len(arg) == 1:
        raise Exception("No model specification provided")
    selected_model = sys.argv[1]
    selected_model = selected_model.upper()


    print("Fine tuning " + selected_model)

    utils.setup_seeds()
    df = utils.get_data()

    if selected_model == "BERT":
        model, tokenizer = setup_BERT()
    else:
        model, tokenizer = setup_XLNet()


    if use_start_dataloader or use_checkpoint:
        train_data = torch.load(dir_path + 'start_train_dataloader.pth')
        val_data = torch.load(dir_path + 'start_val_dataloader.pth')
        test_data = torch.load(dir_path + 'start_test_dataloader.pth')
    else:
        ids, masks, labels = tokenize(df, tokenizer)
        train_data, val_data, test_data = split_data(ids, masks, labels)
    if not use_start_dataloader:
        torch.save(train_data, dir_path + 'start_train_dataloader.pth')
        torch.save(val_data, dir_path + 'start_val_dataloader.pth')
        torch.save(test_data, dir_path + 'start_test_dataloader.pth')
    fine_tune(model, train_data, val_data, selected_model, use_checkpoint)
    testing(test_data, selected_model)

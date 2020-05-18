import transformers
import torch
import utils
from torch.utils.data import TensorDataset, random_split, DataLoader
import numpy as np
import time
import datetime
from torch import nn
import sys
import os
from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def tokenize(df, tokenizer):
    ids = []
    masks = []
    labels = []

    print("**Started Tokenizer**")

    for review, label in tqdm(zip(df['reviewText'], df['overall']), total=len(df['reviewText'])):        
        encoded = tokenizer.encode_plus(review, add_special_tokens= True,
                            max_length = 256, pad_to_max_length = True,
                            return_attention_mask= True, return_tensors='pt')
        
        ids.append(encoded['input_ids'])
        masks.append(encoded['attention_mask'])
        #labels: 1-5 --> 0-4
        labels.append(label - 1)

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

    batch_size = 5
    train_data = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_data = DataLoader(val, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(test, batch_size=batch_size, shuffle=True)

    return train_data, val_data, test_data

def fine_tune(model, train_data, val_data, selected_model, checkpoints, dataloader_path, model_path):
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
    if checkpoints:
        model, optimizer, scheduler, start_epoch, batch_num = utils.load_checkpoint(model, 
                                                        optimizer, scheduler, selected_model, model_path, class_problem)
        model = model.to(device)
        # now individually transfer the optimizer parts...
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    for epoch in range(start_epoch, epochs):
        if epoch > start_epoch and checkpoints == True:
            checkpoints = False
        total = time.time()
        print("\nEpoch " + str(epoch + 1) + "/" + str(epochs))
        train_loss = 0
        model.train()
        
        for step, batch in enumerate(tqdm(train_data)):
            if checkpoints:
                if step < batch_num:
                    continue

            if step % 1000 == 0:
                utils.checkpoint(model, optimizer, scheduler, epoch, step, selected_model, model_path, class_problem)

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

        run_validation(model, val_data)
    
    print("**Ended Fine Tune**\n")
    utils.checkpoint(model, optimizer, scheduler, epoch, len(train_data) ,selected_model, model_path, class_problem)

def run_validation(model, val_data):
    model.eval()
    val_acc = 0
    val_loss = 0

    print("\n**Running Validation batches**")
    for batch in val_data:
        with torch.no_grad():
            loss, logits = model(batch[0].to(device), token_type_ids=None,
                                attention_mask=batch[1].to(device), 
                                labels=batch[2].to(device=device, dtype=torch.int64))
            
            logits = logits.detach().cpu().numpy()
            val_loss += loss.item()
            val_acc += utils.accuracy(batch[2].numpy(), logits)
    
    print("Average validation loss: " + str(val_loss/len(val_data)))
    print("Average validation accuracy: " + str(val_acc/len(val_data)))

def testing(test_data, selected_model, model_path, num_classes):
    if os.path.exists(model_path + selected_model+ '_finetuned_' + class_problem + '.pth'):
        checkpoint = torch.load(model_path + selected_model+ "_finetuned_" + class_problem + ".pth", map_location=device)
    else:
        raise ValueError('No file with the pretrained model selected')

    model, _ = utils.setup_model(selected_model, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("\n**Started Testing**")
    print("-----------------")
    test_acc = 0
    true_labels = []
    predicted_labels = []

    for batch in tqdm(test_data, total=len(test_data)):
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

def run_finetune(selected_model, checkpoints, dataloader_path, model_path, three_class_problem, test_mode):
    print("Fine tuning " + selected_model)

    df, num_classes = utils.get_data()

    global class_problem

    if three_class_problem:
        df, num_classes = utils.three_class_problem(df)
        class_problem = '3'
    else:
        class_problem = '5'

    model, tokenizer = utils.setup_model(selected_model, num_classes)

    if checkpoints:
        train_data = torch.load(dataloader_path + 'train_dataloader' + class_problem + '.pth')
        val_data = torch.load(dataloader_path + 'val_dataloader' + class_problem + '.pth')
        test_data = torch.load(dataloader_path + 'test_dataloader' + class_problem + '.pth')
    else:
        ids, masks, labels = tokenize(df, tokenizer)
        train_data, val_data, test_data = split_data(ids, masks, labels)
        if not test_mode:
            torch.save(train_data, dataloader_path + 'train_dataloader' + class_problem + '.pth')
            torch.save(val_data, dataloader_path + 'val_dataloader' + class_problem + '.pth')
            torch.save(test_data, dataloader_path + 'test_dataloader' + class_problem + '.pth')
    
    if not test_mode:
        fine_tune(model, train_data, val_data, selected_model, checkpoints, dataloader_path, model_path)
    
    testing(test_data, selected_model, model_path, num_classes)
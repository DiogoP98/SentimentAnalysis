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

def tokenize(df, tokenizer, selected_model):
    ids = []
    masks = []
    labels = []

    print("**Started Tokenizer**")

    for review, label in tqdm(zip(df['reviewText'], df['overall']), total=len(df['reviewText'])):
        #solves Index Out of Range for Roberta
        if selected_model == "ROBERTA" and review == "":
            review = " "

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

def fine_tune(model, train_data, val_data, selected_model, checkpoints, saving_path):
    print("**Started Fine Tune**")
    print("Using " + device)
    epoch = 0

    #Adam optimizer with weight decay fix
    if model == "XLNET":
        optimizer = transformers.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = 5e-5, eps = 1e-8)
    elif model == "BERT":
        optimizer = transformers.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = 2e-5, eps = 1e-8)
    else:
        optimizer = transformers.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = 1e-5, eps = 1e-8)
    epochs = 2
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 50000,
                                            num_training_steps = len(train_data) * epochs)

    start_epoch = 0
    if checkpoints:
        model, optimizer, scheduler, start_epoch, batch_num = utils.load_checkpoint(model, 
                                                        optimizer, scheduler, selected_model, saving_path, class_problem)
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
        model.train()
        
        count = 0
        running_loss = 0
        running_acc = 0
        with tqdm(train_data, total=len(train_data), desc='train', position=0, leave=True) as t:
            for step, batch in enumerate(train_data): 
                if checkpoints:
                    if step < batch_num:
                        continue

                if step % 1000 == 0:
                    utils.checkpoint(model, optimizer, scheduler, epoch, step, selected_model, saving_path, class_problem)

                batch[0], batch[1], batch[2] = batch[0].to(device), batch[1].to(device), batch[2].to(device=device, dtype=torch.int64)
                
                optimizer.zero_grad()
                loss, logits = model(batch[0], token_type_ids=None,
                                attention_mask=batch[1], labels=batch[2])
                loss.backward()

                #prevents "exploding gradients" problem
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                running_acc += utils.accuracy(batch[2], logits)
                running_loss += loss.item()
                count += 1

                optimizer.step()
                scheduler.step()

                t.set_postfix(accuracy='{:05.3f}'.format(running_acc/count), loss='{:05.3f}'.format(running_loss/count))
                t.update()
        
        total = time.time() - total
        print("\nElapsed Time: " + str(datetime.timedelta(seconds=int(round((total))))))

        run_validation(model, val_data)
    
    print("**Ended Fine Tune**\n")
    utils.checkpoint(model, optimizer, scheduler, epoch, len(train_data) ,selected_model, saving_path, class_problem)

def run_validation(model, val_data):
    model.eval()
    val_acc = 0
    val_loss = 0

    print("\n**Running Validation batches**")
    with torch.no_grad():
        with tqdm(val_data, total=len(val_data), desc='validation', position=0, leave=True) as t:
            for batch in val_data:
                loss, logits = model(batch[0].to(device), token_type_ids=None,
                                    attention_mask=batch[1].to(device), 
                                    labels=batch[2].to(device=device, dtype=torch.int64))
                
                logits = logits.detach().cpu().numpy()
                val_loss += loss.item()
                val_acc += utils.accuracy(batch[2], logits)
    
    print("Average validation loss: " + str(val_loss/len(val_data)))
    print("Average validation accuracy: " + str(val_acc/len(val_data)))

def testing(test_data, selected_model, saving_path, num_classes):
    file_path = saving_path + selected_model+ '_finetuned_' + class_problem + '.pth'
    print(file_path)
    if os.path.exists(file_path):
        checkpoint = torch.load(file_path, map_location=device)
    else:
        raise ValueError('No file with the pretrained model selected')

    model, _ = utils.setup_model(selected_model, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)

    print("\n**Started Testing**")
    print("-----------------")
    test_acc = 0
    true_labels = []
    predicted_labels = []
    
    for batch in tqdm(test_data, total=len(test_data)):
        with torch.no_grad():
            batch[0], batch[1], batch[2] = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            output = model(batch[0], token_type_ids=None,
                            attention_mask=batch[1])
        
        logits = output[0]

        true_labels, predicted_labels = utils.concatenate_list(batch[2], logits, true_labels, predicted_labels)
        test_acc += utils.accuracy(batch[2], logits)
    
    if num_classes == 3:
        targets = ['Negative','Neutral','Positive']
    else:
        targets = ['1','2','3','4','5']
    
    utils.sklearn_metrics(true_labels, predicted_labels, targets)
    print(f"Test accuracy: {test_acc/len(test_data)}")
    print("**Ended Testing**")

def run_finetune(selected_model, checkpoints, saving_path, three_class_problem, test_mode):
    print("Using " + selected_model)

    df, num_classes = utils.get_data()

    global class_problem

    if three_class_problem:
        df, num_classes = utils.three_class_problem(df)
        class_problem = '3'
    else:
        class_problem = '5'

    model, tokenizer = utils.setup_model(selected_model, num_classes)
    ids, masks, labels = tokenize(df, tokenizer, selected_model)
    train_data, val_data, test_data = split_data(ids, masks, labels)
    
    if not test_mode:
        fine_tune(model, train_data, val_data, selected_model, checkpoints, saving_path)
    
    testing(test_data, selected_model, saving_path, num_classes)

import config
import Meter
import TweetDataset
import model
import engine

import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from sklearn import model_selection
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

def run():
    #Load data
    dfx = pd.read_csv(config.TRAINING_FILE).dropna().reset_index(drop=True)

    #Split data to training set and validation set
    df_train, df_valid = model_selection.train_test_split(
        dfx, 
        test_size=0.1, 
        random_state=42, 
        stratify=dfx.sentiment.values
    )    

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    #Define train dataset class
    train_dataset = TweetDataset.TweetDataset(
        tweet = df_train.text.values,
        sentiment = df_train.sentiment.values,
        target = df_train.selected_text.values)

    #Load training data using DataLoader  
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = config.TRAIN_BATCH_SIZE,
        num_workers=4
        )

    #Define validation dataset class
    valid_dataset = TweetDataset.TweetDataset(
        tweet = df_valid.text.values,
        sentiment = df_valid.sentiment.values,
        target = df_valid.selected_text.values
    )

    #Load validation data using DataLoader
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = config.VALID_BATCH_SIZE,
        num_workers=1
    )

    #Define device
    device = torch.device('cuda')

    #Define model
    bert_model = model.BertModel()

    #Convert model to device
    bert_model.to(device)

    #Define the optimizer parameters (weights and biases from model)
    #Get all of the model's parameters as a list of tuples
    param_optimizer = list(bert_model.named_parameters())

    #Weight Initialization from pretrained BERT
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    #Iterable of parameters to optimize, for AdamW optimizer
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    #
    num_train_steps = int(len(df_train)/config.TRAIN_BATCH_SIZE * config.EPOCHS)

    # Define the optimizer
    optimizer = AdamW(optimizer_parameters, lr=3e-5)

    # Define the learning rate schedule (update)
    scheduler =  get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    # Use multiple GPUs (using DataParallel)
    bert_model = nn.DataParallel(bert_model)

    #Evaluation
    best_jaccard = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_data_loader, bert_model, optimizer, device, scheduler)
        jaccard = engine.eval_fn(valid_data_loader, bert_model, device)
        print(f'Jaccard Score = {jaccard}')
        if jaccard > best_jaccard :
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_jaccard = jaccard

    #Display
    print('EPOCHS = ', config.EPOCHS)

''' run '''
if __name__ == '__main__':
    run()

 




    

    







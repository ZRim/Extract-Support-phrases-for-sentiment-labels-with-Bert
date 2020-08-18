import TweetDataset
import config
import model

import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn import model_selection
from tqdm import tqdm
import numpy as np


#Read training file
dataset_dir = config.TRAINING_FILE
dfx = pd.read_csv(dataset_dir).dropna().reset_index(drop=True)
#print(dfx.head())

#Split data into training and test datasets
df_train ,df_valid = model_selection.train_test_split(
    dfx, 
    test_size=0.99, 
    random_state=32, 
    shuffle=True, 
    stratify=dfx.sentiment.values)

df_train = df_train.reset_index(drop=True)
df_valid = df_valid.reset_index(drop=True)

#Train dataset (import class from file.py)
train_dataset = TweetDataset.TweetDataset(tweet=df_train.text.values, 
                                target=df_train.selected_text.values,
                                sentiment=df_train.sentiment.values)


'''Train '''
train_Dataloader = DataLoader(train_dataset, batch_size=10, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    












# #Define model
# model = model.BertModel()


# # tk0 = tqdm(train_Dataloader)

# # for bi, d in enumerate(tk0): #bi: batch index, d: dataset
# #     ids = d['ids']
# #     mask = d['mask']
# #     target_start = d['target_start']
# #     target_end = d['target_end']
# #     token_type_ids = d['token_type_ids']
# #     target = d['orig_target']
# #     tweet = d['orig_tweet']
# #     sentiment = d['orig_sentiment']

# #     outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
# #     print(outputs)





    
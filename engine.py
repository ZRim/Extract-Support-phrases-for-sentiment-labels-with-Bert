import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import Meter
import string

'''
 Loss function 
 
'''
def loss_fn(output1, output2, target1, target2): #in this case we have two outputs and two targets

    loss1 = nn.BCEWithLogitsLoss()(output1, target1)
    loss2 = nn.BCEWithLogitsLoss()(output2, target2)
    loss = loss1 + loss2

    return loss

''' 
Train function 

'''
def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train() #train mode
    losses = Meter.AverageMeter() #Initialize AverageMeter

    tk0 = tqdm(data_loader, total=len(data_loader))
    for bi, d in enumerate(tk0): # bi : batch index, d : dataset

        #Define variables
        ids = d['ids']
        mask = d['mask']
        token_type_ids = d['token_type_ids']
        target_start = d['targets_start']
        target_end = d['targets_end']
        orig_tweet = d['orig_tweet']
        orig_target = d['orig_target']
        orig_sentiment = d['orig_sentiment']
        tokens = d['tokens']
        padding_len = d['padding_len']

        #Convert variables to device
        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        target_start = target_start.to(device, dtype=torch.float)
        target_end = target_end.to(device, dtype=torch.float)
        padding_len = padding_len.to(device, dtype=torch.long)

        #Clear x.grad for every parameter x in the optimizer. 
        #It’s important to call this before loss.backward(), otherwise you’ll accumulate the 
        #gradients from multiple passes.

        optimizer.zero_grad()

        #Apply the model
        output1, output2 = model.BertModel(ids=ids, mask=mask, token_type_ids=token_type_ids)

        #Calculate the loss function
        loss = loss_fn(output1, output2, target_start, target_end)

        #Apply optimization
        loss.backward() #calculate gradient (computes dloss/dx for every parameter x which has requires_grad=True)
        optimizer.step() #perform w.data=w.data-lr*w.grad.data and b.data=b.data-lr*w.grad.data

        #Change the value of the learning rate
        #(If you don’t call it, the learning rate won’t be changed and stays at the initial value.)
        scheduler.step()

        #Update loss  (apply averageMeter function in Meter.py)
        losses.update(loss.item(), ids.size(0))  #The item() method extracts the loss’s value as a Python float.
        #ids.size(0) gives the current batch size (n=ids.size(0)).

        #loss.item() contains the loss of entire mini-batch, but divided by the batch size. 
        # That's why loss.item() is multiplied with batch size, given by inputs.size(0),
        #while calculating running_loss;
        #torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')

        tk0.set_postfix(loss=losses.avg) #give to me every iteration
        #Once we are in the loop, we can display the number of results discovered using the set_postfix() method.


''' 
Evaluation function (to predict on the test set)

'''
def eval_fn(data_loader, model, device):
    #Set model to evaluation mode
    model.eval()

    fin_output_start = []
    fin_output_end = []
    fin_tokens = []
    fin_padding_len = []

    for bi, d in enumerate(data_loader):
        #Define variables
        ids = d['ids']
        mask = d['mask']
        token_type_ids = d['token_type_ids']
        target_start = d['targets_start']
        target_end = d['targets_end']
        orig_tweet = d['orig_tweet']
        orig_target = d['orig_target']
        orig_sentiment = d['orig_sentiment']
        tokens = d['tokens']
        padding_len = d['padding_len']

        #Convert variables to device
        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        padding_len = padding_len.to(device, dtype=torch.long)
        target_start = target_start.to(device, dtype=torch.float)
        target_end = target_end.to(device, dtype=torch.float)

        

        #Apply the model
        output1, output2 = model.BertModel(ids=ids, mask=mask, token_type_ids=token_type_ids)

        #Convert Logits to probability scores and append result to a final output
        fin_output_start.append(torch.sigmoid(output1).cpu().detach().numpy()) #convert cuda tensor to cpu then to numpy
        fin_output_end.append(torch.sigmoid(output2).cpu().detach().numpy())

        fin_tokens.extend(tokens)
        fin_padding_len.extend(padding_len.cpu().detach.numpy().tolist())
        
        #Stack the sequence of outputs vertically to make a single array
        fin_output_start = np.vstack(fin_output_start)
        fin_output_end = np.vstack(fin_output_end)

        

        #We are getting probabilities between 0 and 1. We need to predict whether it is 0 or 1.
        #That means that we need to set a cut-off line (threshold) in our prediction.
        Threshold = 0.2

        jaccards = []

        for j in range(len(fin_tokens)):
            padding_len = fin_padding_len[j]
            

            if padding_len > 0 :
                mask_start = fin_output_start[j, :][:-padding_len] >= threshold
                mask_end = fin_output_end[j, :][:-padding_len] >= threshold

            else:
                mask_start = fin_output_start[j, :] >= threshold
                mask_end = fin_output_end[j, :] >= threshold

            mask = [0] * len(mask_start)
            idx_start = np.nonzero(mask_start)[0]
            idx_end = np.nonzero(mask_end)[0]

            if len(idx_start) > 0:
                idx_start = idx_start[0]
                if len(idx_end) > 0:
                    idx_end = idx_end[0]
                else:
                    idx_end = idx_start
            else:
                idx_start = 0
                idx_start = 0

            for mj in range(idx_start, idx_end):
                mask[mj] = 1

            output_tokens = [x for p, x in enumerate(tokens.split()) if mask[p] == 1]
            output_tokens = [x for x in output_tokens if x not in ('[CLS]', '[SEP]')]

            final_output = ''
            for ot in output_tokens:
                if ot.startswith('##'):
                    final_output = final_output[2:]
                elif len(ot)==1 and ot in string.punctuation():
                    final_output = final_output + ot
                else:
                    final_output = final_output + ' ' + ot
            final_output = final_output.strip()

            
            #prepare csv file
            submission = pd.read_csv('../input/sample_submission.csv')
            submission.selected_text.iloc[j] = final_output
            submission.to_csv('final_submission.csv', index=False)

            #jaccard
            jac = Meter.jaccard(orig_target.strip(), target_output.strip())
            jaccards.append(jac)

        mean_jac = np.mean(jaccards)












        






















    








        






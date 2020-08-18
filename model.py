import config
import torch
import transformers
import numpy as np
import torch.nn as nn

class BertModel(nn.Module):
    def __init__(self):
        super(BertModel, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH) #bert model
        self.l1 = torch.nn.Dropout(0.1)
        self.l2 = torch.nn.Linear(768,2) #number of outputs should be 2

    def forward(self, ids, mask, token_type_ids):
        last_hidden_state, pooler_output = self.bert(
            input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
        output = self.l1(last_hidden_state)
        logits = self.l2(output) #we have two outputs here
        start_logits, end_logits = logits.split(split_size=1, dim=-1) #dim=-1 to seperate columns
        #shape: (batch_size, num_tokens, 1), (batch_size, num_tokens, 1).

        start_logits = np.squeeze(start_logits, axis=-1)
        end_logits = np.squeeze(end_logits, axis=-1)
        #shape: (batch_size, num_tokens), (batch_size, num_tokens).
        
        return start_logits, end_logits

#I noticed here that I need two outputs for the linear layer l2. In tha case of a simple 
#classification model, the number of outputs is the number of output classes. In our case,
#the output is sub-text of the input text (or tweet). So, we need two outputs which are 
#two vectors of the same length of the input text and, : the first is 
#a vector that indicates the start position of the target and the second output is a vector 
#that indicates the end position of the target.

# let's define :
# input vector (input text): 00000000 
#outputs vectors:
#-first output vector: 00001000 (the 1 indicates the position where the target starts)
#-second output vector: 00000010 (the 1 indicates the position where the target ends).
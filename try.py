import  numpy as np
import config
import model
import torch

###
# #Tokenization
# inputs=config.TOKENIZER.encode('I am hoping for the best')
# print(inputs)
# ids = inputs.ids
# mask = inputs.attention_mask
# token_type_ids = inputs.type_ids
# tok_offsets = inputs.offsets
# print(tok_offsets)

###
tweet = 'Hi my name is Rim !!'
selected_text = 'is Rim'

inputs=config.TOKENIZER.encode(tweet, selected_text)
ids = inputs.ids
print(ids)
mask = inputs.attention_mask
token_type_ids = inputs.type_ids
tok_offsets = inputs.offsets
tokens = [config.TOKENIZER.id_to_token(id) for id in ids] 
print("tokens: ", tokens)

length = len(tokens) #including [cls] and [sep]
init_vector =  [0]*length

target = init_vector
ind = []
for id, tok in enumerate(tokens):
    if tok == '[SEP]':
        ind.append(id)

target_start = target.copy()
target_end = target.copy()

target_start[ind[0]+1] = 1  
print(target_start)

target_end[ind[1]-1] = 1
print(target_end)


# ### Try the code of Abichek
# len_sel_text = len(selected_text)
# idx0 = -1 
# idx1 = -1
# for ind in (i for i, e in enumerate(tweet) if e == selected_text[0]):
#     if tweet[ind: ind+len_sel_text] == selected_text:
#         idx0 = ind
#         idx1 = ind + len_sel_text - 1 #I added '-1' because len takes index+1
#         break

# #
# char_targets = [0] * len(tweet)
# #[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] : characters
# if idx0 != -1 and idx1 != -1 :
#     for j in range(idx0, idx1 + 1):
#         if tweet[j] != ' ':  #if not space
#             char_targets[j] = 1
# #[0,0,0,0,1,1,1,0,1,1,1,0,0,1,1,1,1,0,1,1,0,0,0] : the 0s in the middle are spaces

# #From characters to words: Define the targets variable: 
# targets = [0] * (len(tokens)-2) # -2 to remove [CLS] and [SEP] tokens
# #[0,0,0,0,0,0,0]: words
# for j, (offset1, offset2) in enumerate(tok_offsets):
#     if sum(char_targets[offset1:offset2]) > 0: #??
#         #then the hole token has target of 1, to match partial words
#         targets[j] = 1
# #[0,0,1,1,1,0,0]
# #=> we don't need char_targets anymore, we just need target.

# targets = [0] + targets + [0] #cls, sep
# targets_start = [0] * len(targets)
# targets_end = [0] *len(targets)

# #Create non zero variable:find the indices of the nonzero elements in targets
# non_zero = np.nonzero(targets)[0] 

# if len(non_zero) > 0: #if there are some values
#     targets_start[non_zero[0]] = 1
#     targets_end[non_zero[-1]] = 1 

# print(targets)
# print(targets_start)
# print(targets_end)
        

model = model.BertModel()

ids = torch.tensor([ids])
mask = torch.tensor([mask])
token_type_ids = torch.tensor([token_type_ids])

start_logits, end_logits = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
print('start_logits = ', start_logits)
print('end_logits = ', end_logits)

print(start_logits.shape)
print(end_logits.shape)
#shape: (batch_size, num_tokens, 1), (batch_size, num_tokens, 1).




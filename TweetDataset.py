import config
import torch

class TweetDataset:
    def __init__(self, tweet, target, sentiment): #self and features
        self.tweet = tweet
        self.target = target
        self.sentiment = sentiment
        self.tokenizer = config.TOKENIZER
        self.MAX_LEN = config.MAX_LEN

    #function specifies the size of the dataset. The DataLoader object then uses the __len__ function 
    #of the Dataset to create the batches.
    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, idx):
        
        '''
        CLEAN DATA

        '''
        tweet = str(self.tweet[idx])
        tweet = ' '.join(tweet.split()) #remove acceeding spaces

        target = str(self.target[idx])
        target = ' '.join(target.split())

        sentiment = str(self.sentiment[idx])
        sentiment = ' '.join(sentiment.split())

        '''
        PREPROCESSING DATA (WHAT WE NEED TO TRAIN OUR MODEL):

        We are required to:
        - Add special tokens to the start and end of each sentence.
        - Pad & truncate all sentences to a single constant length.
        - Explicitly differentiate real tokens from padding tokens with the “attention mask”.
        BERT has two constraints:
        - All sentences must be padded or truncated to a single, fixed length.
        - The maximum sentence length is 512 tokens. Padding is done with a special [PAD] token, 
        which is at index 0 in the BERT vocabulary. 

        '''

        inputs = self.tokenizer.encode(tweet, target)

        #The tokenizer returns a dictionary with all the arguments necessary for its corresponding model 
        #to work properly. 

        ids = inputs.ids #the position of the tokens in the dictionary: They are token indices, numerical representations of tokens building the sequences that will be used as input by the model.
        mask = inputs.attention_mask #for padding sequences: all the sequences shoulf have the same length => '1s' for the sequence and then '0s' to have the same length. 
        token_type_ids = inputs.type_ids #The token type IDs (also called segment IDs). 
        #They are a binary mask identifying the different sequences in the model ('0s' for the first segment and '1s' for the second one).
        
        tokens = [config.TOKENIZER.id_to_token(id) for id in ids] 

        #Preprocess the data: we need to represent our target as two output vectors:
        #-first output vector: 00001000 (the 1 indicates the position where the target starts),
        #-second output vector: 00000010 (the 1 indicates the position where the target ends).
        init_vector =  [0]*len(tokens) #including [CLS] and [SEP]

        target = init_vector

        ind = [] #positions of target (start and end)
        for id, tok in enumerate(tokens):
            if tok == '[SEP]':
                ind.append(id)

        #replace the start position by 1 and the end position by 1
        target_start = target.copy()
        target_end = target.copy()

        target_start[ind[0]+1] = 1  
        target_end[ind[1]-1] = 1

        ''' Pad everything you need: (all the outputs sould have the same length as max_len) '''
        padding_len = self.MAX_LEN - len(token_type_ids)

        ids = ids + [0] * padding_len
        mask = mask + [0] * padding_len
        token_type_ids = token_type_ids + [0] * padding_len


    
        # #Sentiment
        # sentiment = [1, 0, 0] #neutral
        # if self.sentiment[idx] == 'positive':
        #     sentiment = [0, 0, 1] #one_hot encoding
        # if self.sentiment[idx] == 'negative':
        #     sentiment = [0, 1, 0]

    
        return {'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'tokens': tokens,
                'padding_len': torch.tensor(padding_len, dtypr=torch.long),
                'target_start': torch.tensor(target_start, dtype=torch.long),
                'target_end': torch.tensor(target_end, dtype=torch.long),
                'orig_tweet': self.tweet[idx],
                'orig_target': self.target[idx],
                'orig_sentiment': self.sentiment[idx]
                }

        # PyTorch Tensor can run on either CPU or GPU. To run operations on the GPU, just cast the Tensor to a cuda datatype.






    

        

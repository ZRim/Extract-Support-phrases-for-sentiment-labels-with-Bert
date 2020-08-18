''' Define parameters'''
import transformers
import tokenizers
import os


MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10

#The pretarined model paths
BERT_PATH = '../input/bert-base-uncased'
MODEL_PATH = 'model.bin'

#The training dataset path
TRAINING_FILE = '../input/train.csv'

#Save the pretarined tokenizer
#TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, lowercase=True)

TOKENIZER = tokenizers.BertWordPieceTokenizer(
        os.path.join(BERT_PATH, 'vocab.txt'),
        add_special_tokens=True,
        lowercase=True
    )

#print(4)


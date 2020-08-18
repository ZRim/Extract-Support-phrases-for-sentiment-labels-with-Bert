''' 

Meters provide a way to keep track of important statistics in an online manner. 

The average of the batch losses will give you an estimate of the “epoch loss” during training. 
Since you are calculating the loss anyway, you could just sum it and calculate the mean after 
the epoch finishes. This training loss is used to see, how well your model performs on the training dataset. 


'''
import torch

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# result=[]

# valeur = torch.Tensor([10])
# average = AverageMeter()

# result = average.update(valeur.item(), 2)

# valeur = torch.Tensor([6])
# result = average.update(valeur.item(), 1)

# print(result)

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

    

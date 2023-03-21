import torch
import torch.nn as nn
import torch.nn.functional as F


#Cross entropy implementation
def crossEntropyLoss(outputs, labels):
    batch_size = outputs.size()[0]
    outputs = F.log_softmax(outputs, dim=1)
    outputs = outputs[range(batch_size), labels]
    return -torch.sum(outputs) / batch_size

#Does not work?
class ZeroOneLoss(nn.Module):
    def __init__(self):
        super(ZeroOneLoss, self).__init__()
    
    def forward(self, input, targets):
        return torch.tensor(sum([1 if x[y] > 0 else 0 for x,y in zip(input ,targets)])/input.size()[0], requires_grad=True)

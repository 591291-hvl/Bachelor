import torch
import torch.nn as nn
import torch.nn.functional as F


# #Cross entropy implementation
# def crossEntropyLoss(outputs, labels):
#     batch_size = outputs.size()[0]
#     outputs = F.log_softmax(outputs, dim=1)
#     outputs = outputs[range(batch_size), labels]
#     return -torch.sum(outputs) / batch_size

def logSoftmax(x):
        x_pow = x.sign() * x.abs().pow(1/3)
        x_exp = torch.exp(x_pow)
        x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
        return torch.log(x_exp/x_exp_sum)

#This does the same
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, input, targets):
        critertion = nn.CrossEntropyLoss()
        loss = critertion(input, targets)
        mask = targets == 0
        highCost = (loss * mask.float()).mean()
        return loss + highCost


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, input, targets):
        batch_size = input.size()[0]
        input = logSoftmax(input)
        input = input[range(batch_size), targets]
        return -torch.sum(input) / batch_size
    
    
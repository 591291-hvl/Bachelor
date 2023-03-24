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


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, input, targets):
        batch_size = input.size()[0]
        input = logSoftmax(input)
        input = input[range(batch_size), targets]
        return -torch.sum(input) / batch_size

class Hingeloss(nn.Module):
    def __init__(self):
        super(Hingeloss, self).__init__()
    
    def forward(self, input, targets):
        m = nn.Sigmoid()
        input = m(input)

        input = torch.max(m(input), dim=-1)[0]

        target = targets.bool()
        margin = torch.zeros_like(input)
        margin[target] = input[target]
        margin[~target] = -input[~target]

        measures = 1 - margin
        measures = torch.clamp(measures, 0)
        total = torch.tensor(target.shape[0], device=target.device)
        return (measures.sum(dim=0) / total)
    


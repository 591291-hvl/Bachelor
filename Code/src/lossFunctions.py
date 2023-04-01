import torch
import torch.nn as nn
import torch.nn.functional as F


# #Cross entropy implementation
# def crossEntropyLoss(outputs, labels):
#     batch_size = outputs.size()[0]
#     outputs = F.log_softmax(outputs, dim=1)
#     outputs = outputs[range(batch_size), labels]
#     return -torch.sum(outputs) / batch_size

def logSoftmax(x, exponent):
    x_pow = x.sign() * x.abs().pow(1/exponent)
    x_exp = torch.exp(x_pow)
    x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
    return torch.log(x_exp/x_exp_sum)


class CustomLoss(nn.Module):
    def __init__(self, exponent):
        super(CustomLoss, self).__init__()
        self.exponent = exponent

    def forward(self, input, targets):
        batch_size = input.size()[0]
        input = logSoftmax(input, self.exponent)
        input = input[range(batch_size), targets]
        return -torch.sum(input) / batch_size


def testSoftmax(x,exponent):
    x_pow = x.sign() * x.abs().pow(1/exponent)
    x_exp = torch.exp(x_pow)
    x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
    return x_exp/x_exp_sum


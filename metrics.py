import torch


def acc(pred, target):
    correct = torch.sum(pred == target.argmax(dim=1))
    return correct.type(torch.float64)/pred.size(0)

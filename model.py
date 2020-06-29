import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from resnet import resnet


def normalize_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


class GraphConv(nn.Module):
    def __init__(self, in_planes, out_planes, bias=False):
        super(GraphConv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_planes, out_planes), requires_grad=True)
        self.adj_weight = nn.Parameter(torch.ones(10, out_planes), requires_grad=True)
        if bias: self.bias = nn.Parameter(torch.Tensor(out_planes), requires_grad=True)
        else: self.register_buffer('bias', None)
        self.init_weight()

    def init_weight(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inp, adj):
        support = torch.mm(inp, self.weight)
        output = torch.mm(adj, self.adj_weight * support)
        # output = torch.mm(adj, support)
        if self.bias is not None:
            output += self.bias
        return output


class Classifier(nn.Module):
    ESP = 1.0e-5
    K = 0.99
    B = 0.01

    def __init__(self, opt):
        super(Classifier, self).__init__()
        self.enable_gcn = opt.enable_gcn
        self.backbone = resnet(opt)
        self.feat_dim = self.backbone.expansion * 512
        self.ac = nn.Sequential(
            nn.BatchNorm1d(10),
            nn.Softmax(dim=1)
        )

        self.adj = nn.Parameter(1 - torch.eye(10), requires_grad=True)
        self.inp = nn.Parameter(torch.ones(10, 1), requires_grad=True)
        self.gcn1 = GraphConv(1, 128)
        self.gcn2 = GraphConv(128, self.feat_dim)

    def pred(self, dist):
        return dist.argmax(dim=1)

    def criteria(self, x, y):
        pos_mask = y
        neg_mask = 1 - y

        pos_cnt = pos_mask.sum(dim=1) + self.ESP
        neg_cnt = neg_mask.sum(dim=1) + self.ESP

        pos_loss = -1 * (pos_mask * (x * self.K + self.B).log()).sum(dim=1) / pos_cnt
        neg_loss = -1 * (neg_mask * (1 - x * self.K).log()).sum(dim=1) / neg_cnt

        return (pos_loss + neg_loss).mean()

    def forward(self, x, y):
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        if self.enable_gcn:
            adj = normalize_adj(self.adj)
            inp = self.gcn1(self.inp, adj)
            inp = F.relu(inp)
            inp = self.gcn2(inp, adj)
            inp = F.relu(inp)
            x = torch.mm(x, inp.t())

        x = self.ac(x)
        loss = self.criteria(x, y)
        pred = self.pred(x)

        return loss, pred

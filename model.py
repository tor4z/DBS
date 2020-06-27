import torch
from torch import nn
from torch.nn import functional as F

from resnet import resnet


class Classifier(nn.Module):
    ESP = 1.0e-5
    K = 0.99
    B = 0.01

    def __init__(self, opt):
        super(Classifier, self).__init__()
        self.backbone = resnet(opt)
        self.feat_dim = self.backbone.expansion * 512
        anchor = torch.randn(opt.num_classes, self.feat_dim)
        self.anchor = nn.Parameter(anchor, requires_grad=True)

    @classmethod
    def fix_range(cls, x):
        return (x + 1) / 2

    def l2_norm(self, x):
        return x / (x.norm(dim=1).view(-1, 1) + self.ESP)

    def pred(self, dist):
        return dist.argmax(dim=1)

    def forward(self, x, y):
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        x = self.l2_norm(x)
        anchor = self.l2_norm(self.anchor)

        dist = torch.mm(x, anchor.t())
        dist = self.fix_range(dist)

        pos_mask = y
        neg_mask = 1 - y

        pos_cnt = pos_mask.sum(dim=1) + self.ESP
        neg_cnt = neg_mask.sum(dim=1) + self.ESP

        pos_loss = -1 * (pos_mask * (self.K * dist + self.B).log()).sum(dim=1) / pos_cnt
        neg_loss = -1 * (neg_mask * (1 - self.K * dist).log()).sum(dim=1) / neg_cnt
        loss = (pos_loss + neg_loss).mean()
        
        pred = self.pred(dist)

        return loss, pred

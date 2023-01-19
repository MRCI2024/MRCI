import torch
import torch.nn as nn
from torch.nn import functional as F

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.criterion = nn.BCELoss(reduction='mean')

    def forward(self, logits, target):
        loss0 = self.criterion(logits[0], target[0])
        loss1 = self.criterion(logits[1], target[1])
        loss2 = self.criterion(logits[2], target[2])
        loss = self.criterion(logits[3], target[2])
        return loss0,loss1,loss2,loss

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def sdloss(self, logit, target):
        num = target.size(0)
        # probs = torch.sigmoid(logits)
        probs = logit
        m1 = probs.view(num, -1)
        m2 = target.view(num, -1)
        intersection = m1 * m2

        score = (
            2.0
            * (intersection.sum(1) + self.smooth)
            / (m1.sum(1) + m2.sum(1) + self.smooth)
        )
        score = 1 - score.sum() / num
        return score
    def forward(self, logits, target):
        loss0 = self.sdloss(logits[0], target[0])
        loss1 = self.sdloss(logits[1], target[1])
        loss2 = self.sdloss(logits[2], target[2])
        loss = self.sdloss(logits[3], target[2])
        return loss0,loss1,loss2,loss

class SegLoss(nn.Module):
    def __init__(self, SoftDice,Crossentropy,w_ce=1,w_sd=1):
        super(SegLoss, self).__init__()
        self.crossentropy = Crossentropy
        self.softdice = SoftDice
        self.l1Loss_1 = nn.L1Loss(reduction='mean')
        self.l1Loss_2 = nn.L1Loss(reduction='mean')
        self.w_ce = w_ce
        self.w_sd = w_sd
    def forward(self, logits, targets):
        closs0, closs1, closs2, closs = self.crossentropy(logits, targets)
        perceptual128 = self.l1Loss_1(logits[5],logits[4]) # self_interaction loss
        perceptual96 = self.l1Loss_2(logits[6],logits[5]) # self_interaction loss
        return  closs0, closs1 , closs2, closs,perceptual128,perceptual96
import torch
import torch.nn as nn
import numpy as np
from sklearn.utils import class_weight

class multitask_CE(torch.nn.Module):
    def __init__(self, num_multitask=3, reduction="mean"):
        super(multitask_CE, self).__init__()
        self.reduction = reduction
        self.num_multitask = num_multitask
        self.criterion = nn.CrossEntropyLoss(reduction=self.reduction)

    def forward(self, input, target, reduction: str = "mean"):
        loss = 0
        loss_0 = self.criterion(input[0], target[0])
        loss_1 = self.criterion(input[1], target[1])
        loss_2 = self.criterion(input[2], target[2])
        #for i in range(self.num_multitask):
        #    loss += self.criterion(input[i], target[i])

        loss = loss_0 + loss_1 + loss_2
        return loss, loss_0 , loss_1 , loss_2

class multitask_WCE(torch.nn.Module):
    def __init__(self, device, weights, num_multitask=3, reduction="mean"):
        super(multitask_WCE, self).__init__()
        self.reduction = reduction
        self.num_multitask = num_multitask
        #class_weights_0 = torch.tensor([6., 1.41, 1.])
        #class_weights_1 = torch.tensor([1., 9.41, 4.40, 10.43])
        #class_weights_2 = torch.tensor([1., 4.74, 28.33, 27.17])
        #class_weights_0 = class_weights_0.to(device)
        #class_weights_1 = class_weights_1.to(device)
        #class_weights_2 = class_weights_2.to(device)

        self.criterion_0 = nn.CrossEntropyLoss(weight=torch.tensor(weights[0], dtype=torch.float).to(device), reduction=self.reduction)
        self.criterion_1 = nn.CrossEntropyLoss(weight=torch.tensor(weights[1], dtype=torch.float).to(device), reduction=self.reduction)
        self.criterion_2 = nn.CrossEntropyLoss(weight=torch.tensor(weights[2], dtype=torch.float).to(device), reduction=self.reduction)

    def forward(self, input, target, reduction: str = "mean"):
        loss = 0
        loss_0 = self.criterion_0(input[0], target[0])
        loss_1 = self.criterion_1(input[1], target[1])
        loss_2 = self.criterion_2(input[2], target[2])
        
        loss = loss_0 + loss_1 + loss_2
        return loss, loss_0 , loss_1 , loss_2
        #return loss

class multitask_focal(torch.nn.Module):
    def __init__(self, num_multitask=3, gamma=2, reduction="mean"):
        super(multitask_focal, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.num_multitask = num_multitask
        self.criterion = nn.CrossEntropyLoss(reduction=self.reduction)

    def forward(self, input, target, reduction: str = "mean"):
        loss = 0

        ce_loss = self.criterion(input[0], target[0])
        pt = torch.exp(-ce_loss)
        loss_0 = ((1 - pt) ** self.gamma * ce_loss).mean()

        ce_loss = self.criterion(input[1], target[1])
        pt = torch.exp(-ce_loss)
        loss_1 = ((1 - pt) ** self.gamma * ce_loss).mean()

        ce_loss = self.criterion(input[2], target[2])
        pt = torch.exp(-ce_loss)
        loss_2 = ((1 - pt) ** self.gamma * ce_loss).mean()

        loss = loss_0 + loss_1 + loss_2
        return loss, loss_0 , loss_1 , loss_2

class multitask_mixup_CE(torch.nn.Module):
    def __init__(self, num_multitask=3, reduction="mean"):
        super(multitask_mixup_CE, self).__init__()
        self.reduction = reduction
        self.num_multitask = num_multitask
        self.criterion = nn.CrossEntropyLoss(reduction=self.reduction)

    def forward(self, input, targets_a, targets_b, lam):
        loss = 0
        for i in range(self.num_multitask):
            loss += lam * self.criterion(input[i], targets_a[i]) + (1 - lam) * self.criterion(input[i], targets_b[i])

        return loss
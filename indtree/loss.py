import torch
from torch import nn
import wandb


class LossTree3Weights(nn.Module):
    def __init__(self, weight_feat=1, weight_class=1, weight_thresh=1, wandb_on=False):
        super().__init__()
        self.weight_feat = torch.Tensor([weight_feat])
        self.weight_class = torch.Tensor([weight_class])
        self.weight_thresh = torch.Tensor([weight_thresh])

        self.criterion_feat = nn.CrossEntropyLoss()  # For feature prediction
        self.criterion_class = nn.CrossEntropyLoss()  # For class prediction
        self.criterion_thresh = nn.MSELoss()  # For threshold prediction
        self.wandb_on = wandb_on

    def forward(self, pred, target, step=None):
        loss_feat = self.criterion_feat(pred[0], target[0])
        loss_class = self.criterion_class(pred[1], target[1])
        loss_thresh = self.criterion_thresh(pred[2], target[2])

        if self.wandb_on and iter is not None:
            wandb.log(
                data={
                    "loss_feat": loss_feat.item(),
                    "loss_class": loss_class.item(),
                    "loss_thresh": loss_thresh.item(),
                },
                step=step,
            )

        loss = (
            self.weight_feat * loss_feat
            + self.weight_class * loss_class
            + self.weight_thresh * loss_thresh
        )
        # Compute the loss
        return loss


class LossTree1Weight(nn.Module):
    def __init__(self, weight, wandb_on=False):
        super().__init__()
        self.weight_feat = torch.Tensor([weight])
        self.weight_class = torch.Tensor([1 - weight])
        self.weight_thresh = torch.Tensor([weight])

        self.criterion_feat = nn.CrossEntropyLoss()  # For feature prediction
        self.criterion_class = nn.CrossEntropyLoss()  # For class prediction
        self.criterion_thresh = nn.MSELoss()  # For threshold prediction
        self.wandb_on = wandb_on

    def forward(self, pred, target):
        loss_feat = self.criterion_feat(pred[0], target[0])
        loss_class = self.criterion_class(pred[1], target[1])
        loss_thresh = self.criterion_thresh(pred[2], target[2])

        if self.wandb_on:
            wandb.log(
                data={
                    "loss_feat": loss_feat.item(),
                    "loss_class": loss_class.item(),
                    "loss_thresh": loss_thresh.item(),
                }
            )

        loss = (
            self.weight_feat * loss_feat
            + self.weight_class * loss_class
            + self.weight_thresh * loss_thresh
        )
        # Compute the loss
        return loss

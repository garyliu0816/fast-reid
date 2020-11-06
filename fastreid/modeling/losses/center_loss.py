# encoding: utf-8
import torch
import torch.nn as nn
from torch.autograd.function import Function


class CenterLoss(nn.Module):
    def __init__(self, cfg):
        super(CenterLoss, self).__init__()
        self.num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        self.feat_dim = cfg.MODEL.BACKBONE.FEAT_DIM
        self.centers = nn.Parameter(
            torch.randn(self.num_classes, self.feat_dim))
        self.centerlossfunc = CenterlossFunc.apply
        self.size_average = cfg.MODEL.LOSSES.CENTER.SIZE_AVERAGE

    def forward(self, embedding, targets):
        batch_size = embedding.size(0)
        embedding = embedding.view(batch_size, -1)
        # To check the dim of centers and features
        if embedding.size(1) != self.feat_dim:
            raise ValueError(
                "Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim, embedding.size(1)))
        batch_size_tensor = embedding.new_empty(1).fill_(
            batch_size if self.size_average else 1)
        loss = self.centerlossfunc(embedding, targets, self.centers,
                                   batch_size_tensor)
        return loss


class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, targets, centers, batch_size):
        ctx.save_for_backward(feature, targets, centers, batch_size)
        centers_batch = centers.index_select(0, targets.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, targets, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, targets.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(targets.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, targets.long(), ones)
        grad_centers.scatter_add_(
            0,
            targets.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers / counts.view(-1, 1)
        return -grad_output * diff / batch_size, None, grad_centers / batch_size, None
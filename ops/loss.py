import torch.nn as nn
import torch


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x, axis):
        """
        x: (bs, ch, D, H, W)
        along axis
        """
        BS, _, D, H, W = x.shape
        if axis==0:
            loss = self.vector_l2(x[:,:,1:,:,:], x[:,:,:D-1,:,:])/(BS*H*W*(D-1))
        elif axis==1:
            loss = self.vector_l2(x[:,:,:,1:,:], x[:,:,:,:H-1,:])/(BS*D*W*(H-1))
        elif axis==2:
            loss = self.vector_l2(x[:,:,:,:,1:], x[:,:,:,:,:W-1])/(BS*D*H*(W-1))
        return self.TVLoss_weight*loss

    def vector_l2(self, x1, x2):
        return torch.pow(x1-x2, 2).sum()
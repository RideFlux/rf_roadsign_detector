import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, beta=1/9, cls_w=1.0, reg_w=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.cls_w = cls_w
        self.reg_w = reg_w
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none', beta=beta)
    
    def forward(self,
                bbox_cls_pred,
                bbox_pred,
                batched_labels, 
                num_cls_pos, 
                batched_bbox_reg):
        '''
        bbox_cls_pred: (n, 5)
        bbox_pred: (n, 7)
        bbox_dir_cls_pred: (n, 2)
        batched_labels: (n, )
        num_cls_pos: int
        batched_bbox_reg: (n, 7)
        batched_dir_labels: (n, )
        return: loss, float.
        '''
        # 1. bbox cls loss
        # focal loss: FL = - \alpha_t (1 - p_t)^\gamma * log(p_t)
        #             y == 1 -> p_t = p
        #             y == 0 -> p_t = 1 - p
        nclasses = bbox_cls_pred.size(1)

        batched_labels = F.one_hot(batched_labels, nclasses + 1)[:, :nclasses].float() # (n, 3)
        np.printoptions(threshold=np.inf, linewidth=np.inf, precision=4, suppress=True)
        bbox_cls_pred_sigmoid = torch.sigmoid(bbox_cls_pred) # (n, 3)
        weights = self.alpha * (1 - bbox_cls_pred_sigmoid).pow(self.gamma) * batched_labels + \
             (1 - self.alpha) * bbox_cls_pred_sigmoid.pow(self.gamma) * (1 - batched_labels) # (n, 3)

        cls_loss = F.binary_cross_entropy(bbox_cls_pred_sigmoid, batched_labels, reduction='none')
        cls_loss = cls_loss * weights
        cls_loss = cls_loss.sum() / num_cls_pos
        
        # 2. regression loss
        reg_loss = self.smooth_l1_loss(bbox_pred, batched_bbox_reg)
        reg_loss = reg_loss.sum() / reg_loss.size(0) if reg_loss.numel() > 0 else torch.tensor(0.0, device=reg_loss.device)
        
        # 3. total loss
        total_loss = self.cls_w * cls_loss + self.reg_w * reg_loss
        
        loss_dict={'cls_loss': cls_loss, 
                    'reg_loss': reg_loss,
                    'total_loss': total_loss}
        return loss_dict
    
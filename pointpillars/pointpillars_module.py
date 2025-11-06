from typing import Any, Dict
from hydra.utils import instantiate

import torch
from torch import Tensor
import torch.nn as nn
from lightning import LightningModule

from pointpillars.anchors import anchor_target, anchors2bboxes
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

class RoadSignDetectorModule(LightningModule):
    def __init__(
        self,
        training_model: bool,
        exporting_onnx: bool,
        num_classes: int = 5,  # Default value for backward compatibility
        pillar_layer: torch.nn.Module = None,
        pillar_vfe: torch.nn.Module = None,
        pillar_scatter: torch.nn.Module = None,
        backbone: torch.nn.Module = None,
        neck: torch.nn.Module = None,
        head: torch.nn.Module = None,
        anchors = None,
        loss_module: torch.nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.num_classes = num_classes
        
        self.pillar_layer = instantiate(pillar_layer)
        self.pillar_vfe = instantiate(pillar_vfe)
        self.pillar_scatter = instantiate(pillar_scatter)
        self.backbone = instantiate(backbone)
        self.neck = instantiate(neck)
        self.head = instantiate(head)
        self.anchors_generator = instantiate(anchors)
        self.loss_module = instantiate(loss_module)

        self.classfication_head = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),


            nn.Flatten(),
            nn.Linear(64 * 25 * 25, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )

        pass

    def lidar_to_image(self, lidar_points):  ###############TODO###################3
        return torch.rand(100, 100).float()

    def forward(self, batched_pts, mode='val', batched_gt_bboxes=None, batched_gt_labels=None):
        batch_size = len(batched_pts)  # Number of batches
        # batched_pts: list[tensor] -> pillars: (p1 + p2 + ... + pb, num_points, c), 
        #                              coors_batch: (p1 + p2 + ... + pb, 1 + 3), 
        #                              num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        
        # Convert tensor to {n,4} ndarray
        batch_dict = self.pillar_layer(batched_pts)

        # pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        # coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        # npoints_per_pillar: (p1 + p2 + ... + pb, )
        #                     -> pillar_features: (bs, out_channel, y_l, x_l)
        
        batch_dict = self.pillar_vfe(batch_dict)
        
        batch_dict = self.pillar_scatter(batch_dict)
        
        
        # xs:  [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        xs = self.backbone(batch_dict)

        # x: (bs, 384, 248, 216)
        x = self.neck(xs)
        
        bbox_cls_pred, bbox_pred = self.head(x)
        feature_map_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), device=bbox_cls_pred.device)
        anchors = self.anchors_generator.get_multi_anchors(feature_map_size)
        batched_anchors = [anchors for _ in range(batch_size)]
        
        results = self.get_predicted_bboxes(bbox_cls_pred=bbox_cls_pred, 
                                            bbox_pred=bbox_pred,
                                            batched_anchors=batched_anchors)
        

        bboxes = [bbox["final_bboxes"] for bbox in results]
        imgs = []
        for idx in range(batch_size):
            lidar_points = batched_pts[idx]
            cx, cy, cz, dx, dy, dz = bboxes[idx].T

            in_box_mask = (
                (lidar_points[:, 0] > cx - dx / 2) & (lidar_points[:, 0] < cx + dx / 2) &
                (lidar_points[:, 1] > cy - dy / 2) & (lidar_points[:, 1] < cy + dy / 2) &
                (lidar_points[:, 2] > cz - dz / 2) & (lidar_points[:, 2] < cz + dz / 2)
            )

            lidar_points = lidar_points[in_box_mask]

            ###############TODO###################3
            img = self.lidar_to_image(lidar_points)
            imgs.append(img)

        imgs = torch.stack(imgs).unsqueeze(1).to(bbox_cls_pred.device)
        img_cls_pred = self.classfication_head(imgs)

        if mode == 'train':
            anchor_target_dict = anchor_target(
                batched_anchors=batched_anchors, 
                batched_gt_bboxes=batched_gt_bboxes, 
                batched_gt_labels=batched_gt_labels,
                nclasses=self.num_classes
            )
            
            return bbox_cls_pred, bbox_pred, anchor_target_dict, img_cls_pred
        elif mode == 'val':

            results = self.get_predicted_bboxes(bbox_cls_pred=bbox_cls_pred, 
                                                bbox_pred=bbox_pred,
                                                batched_anchors=batched_anchors)
            return results

        elif mode == 'test':
            results = self.get_predicted_bboxes(bbox_cls_pred=bbox_cls_pred,        
                                                bbox_pred=bbox_pred,
                                                batched_anchors=batched_anchors)
            return results
        else:
            raise ValueError   

    def get_predicted_bboxes_single(self, bbox_cls_pred, bbox_pred, anchors):
        '''
        bbox_cls_pred: (n_anchors*3, 248, 216) 
        bbox_pred: (n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (n_anchors*2, 248, 216)
        anchors: (y_l, x_l, 3, 2, 7)
        return: 
            bboxes: (k, 7)
            labels: (k, )
            scores: (k, )
        '''
        # 1. reshape inputs
        bbox_cls_pred = bbox_cls_pred.permute(1, 2, 0).reshape(-1, self.num_classes)  # [H*W*A, C]
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 6)  # [H*W*A, 6]
        anchors = anchors.reshape(-1, 6)

        # 2. sigmoid classification
        bbox_cls_pred = torch.sigmoid(bbox_cls_pred)  # [N, C]

        # 3. get top 1 score and its index across all classes
        scores, labels = bbox_cls_pred.max(dim=1)  # [N], [N]
        topk_scores, topk_indices = torch.topk(scores, 1)  # [1]
        
        # 4. gather top prediction
        topk_bbox_pred = bbox_pred[topk_indices]      # [1, 6]
        topk_anchor = anchors[topk_indices]           # [1, 6]
        topk_label = labels[topk_indices]             # [1]
        topk_score = topk_scores                      # [1]

        # 5. decode bbox
        topk_bbox = anchors2bboxes(topk_anchor, topk_bbox_pred)  # [1, 7]

        # 6. truncate to max_num (==1 for now)
        final_bboxes = topk_bbox[:1]
        final_labels = topk_label[:1]
        final_scores = topk_score[:1]

        result = {
            'final_bboxes': final_bboxes,
            'final_labels': final_labels,
            'final_scores': final_scores
        }
        return result

    def get_predicted_bboxes(self, bbox_cls_pred, bbox_pred, batched_anchors):
        '''
        bbox_cls_pred: (bs, n_anchors*3, 248, 216) 
        bbox_pred: (bs, n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        batched_anchors: (bs, y_l, x_l, 3, 2, 7)
        return: 
            bboxes: [(k1, 7), (k2, 7), ... ]
            labels: [(k1, ), (k2, ), ... ]
            scores: [(k1, ), (k2, ), ... ] 
        '''
        results = []
        bs = bbox_cls_pred.size(0)
        for i in range(bs):
            result = self.get_predicted_bboxes_single(bbox_cls_pred=bbox_cls_pred[i],
                                                      bbox_pred=bbox_pred[i],
                                                      anchors=batched_anchors[i])
            results.append(result)
        return results

    def get_loss(self, data, idx, mode='train') -> Tensor:
        img, batched_gt_bboxes, batched_gt_labels, _ = data
        bbox_cls_pred, bbox_pred, anchor_target_dict, img_cls_pred = self.forward(
            img, mode='train', batched_gt_bboxes=batched_gt_bboxes, batched_gt_labels=batched_gt_labels
        )

        # (B, C, H, W) -> (B*H*W*num_anchors, C)
        bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 6)

        # anchor_target 기준 ground truth
        batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
        batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
        batched_bbox_reg = anchor_target_dict['batched_bbox_reg']
        
        batched_bbox_reg = batched_bbox_reg.reshape(-1, 6)
        pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < self.num_classes)
        
        bbox_pred = bbox_pred[pos_idx]
        batched_bbox_reg = batched_bbox_reg[pos_idx]

        num_cls_pos = (batched_bbox_labels < self.num_classes).sum() + 1
        bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
        batched_bbox_labels = torch.where(batched_bbox_labels < 0, self.num_classes, batched_bbox_labels)
        batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]
        
        img_cls_gt = torch.cat(batched_gt_labels)
        img_cls_gt = img_cls_gt.to(torch.long)
        img_cls_loss = nn.CrossEntropyLoss()(img_cls_pred, img_cls_gt)

        losses = self.loss_module.forward(
            bbox_cls_pred=bbox_cls_pred,
            bbox_pred=bbox_pred,
            batched_labels=batched_bbox_labels,
            num_cls_pos=num_cls_pos,
            batched_bbox_reg=batched_bbox_reg
        )

        losses["total_loss"] = losses["total_loss"] + img_cls_loss
        losses["img_cls_loss"] = img_cls_loss

        if(mode == 'train'):
            writer.add_scalar("Loss/train", losses['total_loss'], idx)
        if(mode == 'val'):
            writer.add_scalar("Loss/val", losses['total_loss'], idx)

        return losses

    def training_step(self, data, idx:int) -> Dict[str, Tensor]:
        losses = self.get_loss(data, idx=idx, mode='train')
        
        self.log_dict(
            losses,
            prog_bar=True, 
            sync_dist=True
        )

        return losses['total_loss']

    def validation_step(self, data, idx: int) -> Dict[str, Tensor]:
        losses = self.get_loss(data, idx=idx, mode='val')
        
        # Get batch size from input data
        batch_size = data['gt_bboxes_3d'].size(0) if 'gt_bboxes_3d' in data else 1
        
        self.log_dict(
            losses,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True
        )

        return losses['total_loss']

    def test_step(self, data, idx: int) -> Dict[str, Any]:
        losses = self.get_loss(data, idx=idx, mode='test')
        
        self.log_dict(
            losses,
            prog_bar=True,
            sync_dist=True
        )
        
        return losses['total_loss']
    
    def configure_optimizers(self) -> Dict[str, Any]:
        
        optimizer = instantiate(self.hparams.optimizer, params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            self.hparams.scheduler["total_steps"] = int(self.trainer.estimated_stepping_batches)
            
            scheduler = instantiate(self.hparams.scheduler, optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "total_loss",
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
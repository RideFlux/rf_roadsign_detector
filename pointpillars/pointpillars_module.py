from typing import Any, Dict
from hydra.utils import instantiate

import torch
from torch import Tensor
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
        pillar_encoder: torch.nn.Module = None,
        backbone: torch.nn.Module = None,
        neck: torch.nn.Module = None,
        head: torch.nn.Module = None,
        anchors = None,
        loss_module: torch.nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        output_pool_nms : bool = True,
        nms_pre: int = 256,
        nms_thr: float = 0.01,
        score_thr: float = 0.1,
        max_num: int = 500,
        nclasses: int = None,
        assigners = None
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.num_classes = num_classes
        self.nclasses = nclasses if nclasses is not None else num_classes
        self.nms_pre = nms_pre
        self.nms_thr = nms_thr
        self.score_thr = score_thr
        self.max_num = max_num
        self.assigners = assigners
        
        self.pillar_layer = instantiate(pillar_layer)
        self.pillar_encoder = instantiate(pillar_encoder)
        self.backbone = instantiate(backbone)
        self.neck = instantiate(neck)
        self.head = instantiate(head)
        self.anchors_generator = instantiate(anchors)
        self.loss_module = instantiate(loss_module)
        pass

    def forward(self, batched_pts, mode='val', batched_gt_bboxes=None, batched_gt_labels=None):
        batch_size = len(batched_pts)
        # batched_pts: list[tensor] -> pillars: (p1 + p2 + ... + pb, num_points, c), 
        #                              coors_batch: (p1 + p2 + ... + pb, 1 + 3), 
        #                              num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        
        # Convert tensor to {n,4} ndarray
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)

        # pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        # coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        # npoints_per_pillar: (p1 + p2 + ... + pb, )
        #                     -> pillar_features: (bs, out_channel, y_l, x_l)
        
        pillar_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)
        
        # xs:  [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        xs = self.backbone(pillar_features)

        # x: (bs, 384, 248, 216)
        x = self.neck(xs)
        
        bbox_cls_pred, bbox_pred = self.head(x)
        device = bbox_cls_pred.device
        feature_map_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), device=device)
        anchors = self.anchors_generator.get_multi_anchors(feature_map_size)
        batched_anchors = [anchors for _ in range(batch_size)]
        
        if mode == 'train':
            anchor_target_dict = anchor_target(
                batched_anchors=batched_anchors, 
                batched_gt_bboxes=batched_gt_bboxes, 
                batched_gt_labels=batched_gt_labels,
                nclasses=self.nclasses
            )
            
            return bbox_cls_pred, bbox_pred, anchor_target_dict
        elif mode == 'val':

            results = self.get_predicted_bboxes(pts=batched_pts,
                                                bbox_cls_pred=bbox_cls_pred, 
                                                bbox_pred=bbox_pred,
                                                batched_anchors=batched_anchors)
            return results

        elif mode == 'test':
            results = self.get_predicted_bboxes(pts=batched_pts,
                                                bbox_cls_pred=bbox_cls_pred,        
                                                bbox_pred=bbox_pred,
                                                batched_anchors=batched_anchors)
            return results
        else:
            raise ValueError   

    def get_predicted_bboxes_single(self, pts, bbox_cls_pred, bbox_pred, anchors):
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
        # 0. pre-process 
        bbox_cls_pred = bbox_cls_pred.permute(1, 2, 0).reshape(-1, self.nclasses)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 6)
        anchors = anchors.reshape(-1, 6)
        
        bbox_cls_pred = torch.sigmoid(bbox_cls_pred)

        # 1. obtain self.nms_pre bboxes based on scores
        inds = bbox_cls_pred.max(1)[0].topk(self.nms_pre)[1]
        bbox_cls_pred = bbox_cls_pred[inds]
        bbox_pred = bbox_pred[inds]
        anchors = anchors[inds]

        # 2. decode predicted offsets to bboxes
        bbox_pred = anchors2bboxes(anchors, bbox_pred)
        
        ret_bboxes, ret_labels, ret_scores = [], [], []
        for i in range(self.nclasses):
            # 3.1 filter bboxes with scores below self.score_thr
            cur_bbox_cls_pred = bbox_cls_pred[:, i]
            score_inds = cur_bbox_cls_pred > 0 # self.score_thr
            if score_inds.sum() == 0:
                continue

            cur_bbox_cls_pred = cur_bbox_cls_pred[score_inds]
            cur_bbox_pred = bbox_pred[score_inds]
            
            # replace nms. pick top 1 score bbox
            order = cur_bbox_cls_pred.sort(0, descending=True)[1]
            keep_inds = order[:1]

            cur_bbox_cls_pred = cur_bbox_cls_pred[keep_inds]
            cur_bbox_pred = cur_bbox_pred[keep_inds]

            ret_bboxes.append(cur_bbox_pred)
            ret_labels.append(torch.zeros_like(cur_bbox_pred[:, 0], dtype=torch.long) + i)
            ret_scores.append(cur_bbox_cls_pred)

        # 4. filter some bboxes if bboxes number is above self.max_num
        if len(ret_bboxes) == 0:
            return {
                'lidar_bboxes': [],
                'labels': [],
                'scores': []
            }
        ret_bboxes = torch.cat(ret_bboxes, 0)
        ret_labels = torch.cat(ret_labels, 0)
        ret_scores = torch.cat(ret_scores, 0)
        if ret_bboxes.size(0) > self.max_num:
            final_inds = ret_scores.topk(self.max_num)[1]
            ret_bboxes = ret_bboxes[final_inds]
            ret_labels = ret_labels[final_inds]
            ret_scores = ret_scores[final_inds]
        result = {
            'lidar_bboxes': ret_bboxes.detach().cpu().numpy(),
            'labels': ret_labels.detach().cpu().numpy(),
            'scores': ret_scores.detach().cpu().numpy()
        }
        return result

    def get_predicted_bboxes(self, pts, bbox_cls_pred, bbox_pred, batched_anchors):
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
            result = self.get_predicted_bboxes_single(pts=pts,
                                                        bbox_cls_pred=bbox_cls_pred[i],
                                                        bbox_pred=bbox_pred[i],
                                                        anchors=batched_anchors[i])
            results.append(result)
        return results

    def get_loss(self, data, idx, mode='train') -> Tensor:
        img, batched_gt_bboxes, batched_gt_labels = data
        bbox_cls_pred, bbox_pred, anchor_target_dict = self.forward(
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
        pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < self.nclasses)
        
        bbox_pred = bbox_pred[pos_idx]
        batched_bbox_reg = batched_bbox_reg[pos_idx]

        num_cls_pos = (batched_bbox_labels < self.nclasses).sum() + 1
        bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
        batched_bbox_labels = torch.where(batched_bbox_labels < 0, self.nclasses, batched_bbox_labels)
        batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]
        
        losses = self.loss_module.forward(
            bbox_cls_pred=bbox_cls_pred,
            bbox_pred=bbox_pred,
            batched_labels=batched_bbox_labels,
            num_cls_pos=num_cls_pos,
            batched_bbox_reg=batched_bbox_reg
        )

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
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
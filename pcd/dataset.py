import torch
from torch.utils.data import Dataset

class RoadSignDataset(Dataset):
    def __init__(
        self, 
        down_stride: int,
    ):
        super().__init__()
        self.data_list = []
        
        self.down_stride = down_stride
        pass

    
    def setDataset(self, data_list):
        self.data_list = data_list
    
    def getDataset(self):
        return self.data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]
    
    def collate(self, data):
        """
        Collate function to batch data
        data: list of tuples (img, gt_bboxes, gt_cls)
        """
        batched_pts_list = []
        batched_gt_bboxes_list = []
        batched_gt_labels_list = []
        
        for item in data:
            img, gt_bboxes, gt_cls = item
            
            batched_pts_list.append(torch.from_numpy(img))
            batched_gt_bboxes_list.append(torch.from_numpy(gt_bboxes))
            batched_gt_labels_list.append(torch.from_numpy(gt_cls))
        
        return (batched_pts_list, batched_gt_bboxes_list, batched_gt_labels_list)
        
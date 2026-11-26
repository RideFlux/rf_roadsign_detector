import torch
from torch.utils.data import Dataset
from pcd.utils import read_pcd

class RoadSignDataset(Dataset):
    def __init__(
        self
    ):
        super().__init__()
        self.data_list = []
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
        batched_pcd_path_list = []
        
        for idx, item in enumerate(data):
            # img, gt_bboxes, gt_cls = item
            ## ram이 터져서 pcd_path만 미리 읽도록 하고 여기서 pcd파일 읽도록 함
            pcd_path, gt_bboxes, gt_cls, reversed_flag = item
            # print(idx, pcd_path)

            img = read_pcd(pcd_path)
            if len(gt_bboxes) > 0 and gt_bboxes[0][0] > 0:
                # Rotate point cloud: (x, y) -> (y, -x)
                img_rotated = img.copy()
                img_rotated[:, 0] = img[:, 1]  # new x = old y
                img_rotated[:, 1] = -img[:, 0]  # new y = -old x
                img = img_rotated
            if reversed_flag:
                img[:, 1] = -img[:, 1]
            
            batched_pts_list.append(torch.from_numpy(img))
            batched_gt_bboxes_list.append(torch.from_numpy(gt_bboxes))
            batched_gt_labels_list.append(torch.from_numpy(gt_cls))
            batched_pcd_path_list.append(pcd_path)
        
        return (batched_pts_list, batched_gt_bboxes_list, batched_gt_labels_list, batched_pcd_path_list)
        
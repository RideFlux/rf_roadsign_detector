import os
import random
from typing import Dict
import sys
import numpy as np
import warnings
from tqdm import tqdm
import pickle

from pcd.utils import read_pcd

class DataProcessor:
    def __init__(self, pickle_files: Dict, stride):
        self.pickle_files = pickle_files  # {'train': 'path/to/train.pkl', 'val': 'path/to/val.pkl', ...}
        self.stride = stride
        self.valid_split_names = pickle_files.keys()
        self.num_classes = 1  # 고정값으로 설정
        pass

    def get_data_list(self, split_name: str):
        if split_name not in self.valid_split_names:
            print(f"invalid split name: {split_name}. please select split name in [{', '.join(self.valid_split_names)}]")
            sys.exit()
            return
        
        # Load pickle file
        pickle_file_path = self.pickle_files[split_name]
        with open(pickle_file_path, 'rb') as f:
            dataset = pickle.load(f)
        
        # Process data with progress bar
        processed_data = []
        for item in tqdm(dataset[::self.stride], desc=f"Processing {split_name} data", unit="files"):
            result = self._read_data_from_pickle(item)
            if result is not None:
                processed_data.append(result)
        return processed_data

    def _read_data_from_pickle(self, data_item):
        # Extract data from pickle item
        pcd_path = data_item['pcd_path']
        label_data = data_item['label']
        reversed_flag = data_item['reversed']
        
        # Skip problematic files
        if 'C8_construct(e)' in pcd_path:
            return None
            
        # Read PCD file
        img = read_pcd(pcd_path)
        
        # Initialize empty arrays for the case when no labels exist
        gt_bboxes = np.zeros([0, 6], dtype=np.float32)
        gt_cls = np.zeros([0], dtype=np.int32)
        
        
        
        # Process label if exists
        if label_data is not None:
            try:
                class_id = label_data['class']
                bbox_coords = label_data['bbox']  # [8][3] 형태
                
                # Convert class (1-based to 0-based)
                gt_cls = np.array([class_id - 1], dtype=np.int32)
                
                # Convert [8][3] bbox to [x, y, z, w, l, h] format
                # bbox_coords는 8개 꼭짓점의 좌표 [8][3]
                # 기존 로직과 동일하게 변환
                coords_flat = bbox_coords.flatten()  # [24] 형태로 변환
                
                # 기존 로직 적용 (label[:, 1:25]를 coords_flat[0:24]로 대체)
                dim = np.array([coords_flat[3] - coords_flat[0],    # x_max - x_min
                               coords_flat[7] - coords_flat[4],     # y_max - y_min  
                               coords_flat[14] - coords_flat[11]], dtype=np.float32)   # z_max - z_min
                
                loc = np.array([coords_flat[0] + (coords_flat[3] - coords_flat[0])/2,   # x_center
                               coords_flat[4] + (coords_flat[7] - coords_flat[4])/2,    # y_center
                               coords_flat[11]], dtype=np.float32)                      # z_min
                
                gt_bboxes = np.concatenate([loc, dim]).reshape(1, 6).astype(np.float32)
                
            except Exception as e:
                print(f"Error processing label data {pcd_path}: {e}")
                return (img, gt_bboxes, gt_cls)
        
        # Check if we need to rotate data based on reversed flag
        if len(gt_bboxes) > 0 and gt_bboxes[0][0] > 0:
            # Rotate point cloud: (x, y) -> (y, -x)
            img_rotated = img.copy()
            img_rotated[:, 0] = img[:, 1]  # new x = old y
            img_rotated[:, 1] = -img[:, 0]  # new y = -old x
            img = img_rotated
            
            # Rotate bounding boxes if they exist
            if len(gt_bboxes) > 0:
                gt_bboxes_rotated = gt_bboxes.copy()
                gt_bboxes_rotated[:, 0] = gt_bboxes[:, 1]   # new x = old y
                gt_bboxes_rotated[:, 1] = -gt_bboxes[:, 0]  # new y = -old x
                # Swap w and l (width and length)
                gt_bboxes_rotated[:, 3] = gt_bboxes[:, 4]   # new w = old l
                gt_bboxes_rotated[:, 4] = gt_bboxes[:, 3]   # new l = old w
                gt_bboxes = gt_bboxes_rotated
                
        if reversed_flag:
            img[:, 1] = -img[:, 1]
            

        return (img, gt_bboxes, gt_cls)
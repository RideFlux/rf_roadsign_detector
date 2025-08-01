import os
from typing import Dict
import sys
import numpy as np
import warnings
from tqdm import tqdm

from pcd.utils import read_pcd

class DataProcessor:
    def __init__(self, dataset_info: Dict, stride, down_stride):
        self.dataset_info = dataset_info
        self.stride = stride
        self.down_stride = down_stride
        self.valid_split_names = self.dataset_info['imageset_files'].keys()

        self.root = dataset_info['dataset_root_dir']
        self.image_folder = os.path.join(self.root, dataset_info['rimg_dir_name'])
        self.annotation_folder = os.path.join(self.root, dataset_info['label_dir_name'])
        self.num_classes = dataset_info['num_classes']
        pass

    def get_data_list(self, split_name: str):
        if split_name not in self.valid_split_names:
            print(f"invalid split name: {split_name}. please select split name in [{', '.join(self.valid_split_names)}]")
            sys.exit()
            return
        
        data_root_dir_path = self.dataset_info['dataset_root_dir']
        imageset_files = self.dataset_info['imageset_files'][split_name]
        
        # Handle both single file (string) and multiple files (list)
        if isinstance(imageset_files, str):
            imageset_files = [imageset_files]
        
        current_imagesets = []
        for imageset_file in imageset_files:
            imageset_file_path = os.path.join(data_root_dir_path, imageset_file)
            with open(imageset_file_path) as f:
                file_imagesets = f.read().split()[::self.stride]
                current_imagesets.extend(file_imagesets)

        # Process data with progress bar
        processed_data = []
        for item in tqdm(current_imagesets, desc=f"Processing {split_name} data", unit="files"):
            processed_item, fake_item = self._read_data_path(item)
            if processed_item is not None:
                processed_data.append(processed_item)
            if fake_item is not None:
                processed_data.append(fake_item)
        return processed_data

    def _read_data_path(self, file_name):
        rimg_path = os.path.join(self.image_folder, file_name+'.pcd')
        label_path = os.path.join(self.annotation_folder, file_name+'.txt')
        # print(f"Processing {rimg_path} and {label_path}")
        if('C8_construct(e)' in rimg_path):
            return None, None
        img = read_pcd(rimg_path)
        
        # Initialize empty arrays for the case when no labels exist
        gt_bboxes = np.zeros([0, 6], dtype=np.float32)
        gt_cls = np.zeros([0], dtype=np.int32)
        
        if os.path.exists(label_path):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    # Read as string first, then process
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    
                    if not lines:
                        return img, gt_bboxes, gt_cls

                    # Process each line
                    labels = []
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(',')
                        if len(parts) >= 25:
                            # Convert to float
                            label_row = [float(x) for x in parts[:25]]
                            labels.append(label_row)
                    
                    if not labels:
                        return img, gt_bboxes, gt_cls

                    # Convert to numpy array
                    label = np.array(labels, dtype=np.float32)
                    
                    # Extract class and bounding box info
                    gt_cls = label[:, 0].astype(np.int32) - 1
                    
                    dim = np.stack([label[:, 4] - label[:, 1], # np.ones_like(label[:, 4])
                                label[:, 8] - label[:, 5], # np.ones_like(label[:, 8])
                                label[:, 15] - label[:, 12]], axis=1).astype(np.float32)  # dimensions
                    loc = np.stack([label[:, 1] + (label[:, 4] - label[:, 1])/2,
                                label[:, 5] + (label[:, 8] - label[:, 5])/2, 
                                label[:, 12]], axis=1).astype(np.float32)  # locations
                    
                    gt_bboxes = np.concatenate([loc, dim], axis=1).astype(np.float32)
                    
            except Exception as e:
                print(f"Error processing label file {label_path}: {e}")
                return img, gt_bboxes, gt_cls
            
        # Check if we need to rotate data (when x > 0)
        # 작년 데이터를 함께 학습시키기 위한 코드
        if len(gt_bboxes) > 0 and gt_bboxes[0, 0] > 0:
            # Rotate point cloud: (x, y) -> (y, -x)
            img_rotated = img.copy()
            img_rotated[:, 0] = img[:, 1]  # new x = old y
            img_rotated[:, 1] = -img[:, 0]  # new y = -old x
            img = img_rotated
            
            # Rotate bounding boxes: (x, y) -> (y, -x)
            gt_bboxes_rotated = gt_bboxes.copy()
            gt_bboxes_rotated[:, 0] = gt_bboxes[:, 1]   # new x = old y
            gt_bboxes_rotated[:, 1] = -gt_bboxes[:, 0]  # new y = -old x
            # Swap w and l (width and length)
            gt_bboxes_rotated[:, 3] = gt_bboxes[:, 4]   # new w = old l
            gt_bboxes_rotated[:, 4] = gt_bboxes[:, 3]   # new l = old w
            gt_bboxes = gt_bboxes_rotated

        # ego 뒷부분 point들을 이용해서 semi-supervised 학습
        img_back = img.copy()
        img_back[:, 1] = -img[:, 1]  # y -> -y

        return (img, gt_bboxes, gt_cls), (img_back, np.zeros([0, 6], dtype=np.float32), np.zeros([0], dtype=np.int32))

import numpy as np
import pypcd
import pypcd.pypcd

def read_pcd(pcd_file_path: str) -> np.ndarray:
    points_pcd = pypcd.pypcd.PointCloud.from_path(pcd_file_path)
    pcd_x = points_pcd.pc_data['x'].copy()
    pcd_y = points_pcd.pc_data['y'].copy()
    pcd_z = points_pcd.pc_data['z'].copy()
    pcd_intensity = points_pcd.pc_data['intensity'].copy().astype(np.float32)
    
    pcd = np.array([pcd_x, pcd_y, pcd_z, pcd_intensity])
    return pcd.T
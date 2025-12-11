import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from evaluation.predict import  inference_test_imgs_qtt
from utils.sum_resolver import SumResolver
import cv2
import numpy as np
import os
from tqdm import tqdm
import torch.nn.functional as F

pandarGeneral_elev_angle_map = np.array(
    [
        14.882, 11.032, 8.059, 5.057, 3.04, 2.028, 1.86, 1.688, 1.522, 1.351, 1.184, 1.013,
        0.846, 0.675, 0.508, 0.337, 0.169, 0.0, -0.169, -0.337, -0.508, -0.675, -0.845, -1.013,
        -1.184, -1.351, -1.522, -1.688, -1.86, -2.028, -2.198, -2.365, -2.536, -2.7, -2.873,
        -3.04, -3.21, -3.375,-3.548, -3.712, -3.884, -4.05, -4.221, -4.385, -4.558, -4.72, -4.892,
        -5.057, -5.229, -5.391, -5.565,-5.726, -5.898, -6.061, -7.063, -8.059, -9.06, -9.885,
        -11.032, -12.006, -12.974, -13.93, -18.889, -24.897
    ]
)

vert_tan = np.tan(np.radians(pandarGeneral_elev_angle_map))

def zoom_tensor(img_tensor, zoom_factor):
    img_tensor = img_tensor.unsqueeze(0)
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    N, C, H, W = img_tensor.shape
    # Resize
    new_H, new_W = int(H * zoom_factor), int(W * zoom_factor)
    zoomed = F.interpolate(img_tensor, size=(new_H, new_W), mode='bilinear', align_corners=False)

    # Crop or pad to original size
    if zoom_factor < 1:
        pad_h = (H - new_H) // 2
        pad_w = (W - new_W) // 2
        result = F.pad(zoomed, (pad_w, W - new_W - pad_w, pad_h, H - new_H - pad_h))
    else:  # enlarge -> crop
        start_h = (new_H - H) // 2
        start_w = (new_W - W) // 2
        result = zoomed[:, :, start_h:start_h + H, start_w:start_w + W]

    # Remove batch dim if original had none
    if img_tensor.shape[0] == 1 and img_tensor.shape[1] != 1:
        result = result.squeeze(0)
        
    result = result.squeeze(0).squeeze(0)
    return result

def find_nearest_altitude_idx(points):
    # Extract x, y, z
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Compute tangent of altitude angle = z / hypot(x,y)
    r = np.hypot(x, y)          # SIMD + fast C implementation
    u = z / r                   # tan(theta)

    # Find nearest beam by tangent difference
    diff = np.abs(u[:, None] - vert_tan[None, :])
    beam_idx = np.argmin(diff, axis=1)

    return beam_idx

def get_image(bboxes, points, size_u = 64, size_v = 32):
    cx, cy, cz, dx, dy, dz = bboxes[0]

    dx = dx * 1.2
    dz = dz * 1.2

    min_x = cx - dx / 2
    max_x = cx + dx / 2
    min_z = cz - dz / 2
    max_z = cz + dz / 2
    min_y = cy - dy / 2
    max_y = cy + dy / 2

    min_z = min_z + dz / 2
    max_z = max_z + dz / 2

    min_y = cy - 1
    max_y = cy + 1

    # cz = (min_z + max_z) / 2 + np.random.rand() - 0.5
    # min_z = cz - dz / 2
    # max_z = cz + dz / 2

    mask = (
        (points[:, 0] >= min_x) & (points[:, 0] <= max_x) &
        (points[:, 1] >= min_y) & (points[:, 1] <= max_y) ## &
        # (points[:, 2] >= min_z) & (points[:, 2] <= max_z)
    )
    points = points[mask]
    if len(points) < 50 or len(points) > 50000:
        return torch.zeros((3, size_u, size_v))
        # return torch.zeros((2, size_u, size_v))
    x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    
    # 64x32 grid로 매핑할 인덱스 계산
    # x축은 x좌표 정보를 이용
    # z축은 lidar의 ring index를 찾아서 배치
    grid_v = ((max_x - x) / (max_x - min_x) * size_v).long().clamp(0, size_v-1)
    grid_u = find_nearest_altitude_idx(points)
    device = points.device
    grid = torch.zeros((size_u, size_v), dtype=torch.float32, device=device)
    count = torch.zeros((size_u, size_v), dtype=torch.float32, device=device)
    ydist = torch.zeros((size_u, size_v), dtype=torch.float32, device=device)
    for gu, gv, yval, inten in zip(grid_u, grid_v, y, intensity):
        grid[gu, gv] += inten
        ydist[gu, gv] += yval
        count[gu, gv] += 1

    grid = torch.where(count > 0, grid / count, torch.zeros_like(grid))/255
    meany = -torch.where(count > 0, ydist / count, torch.zeros_like(grid))/40   ## 40으로 나눈건 데이터 정규화를 위해서. 포인트 범위는 0~-48까지이긴 하지만 40m 밖으로는 거의 없어서 이렇게 사용
    img = torch.stack([grid, meany, count], dim=0)                              ## shape : 3, size_u(=64), size_v(=32)
    return img.cuda()

OmegaConf.register_new_resolver("sum", SumResolver, replace=True)

@hydra.main(version_base=None, config_path="configs", config_name="classifier.yaml")
def main(cfg: DictConfig):
    ckpt_path = cfg.regression_checkpoint

    num_classes = cfg.data.datasets.num_classes
    
    model = hydra.utils.instantiate(
        cfg.model.roadsign_detector.module,
        training_model=False,
        exporting_onnx=False,
        num_classes=num_classes
    )
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    print(f"Loading checkpoint from {ckpt_path}...")
    map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location=map_location)
    loaded_state_dict = checkpoint['state_dict']
    model.load_state_dict(loaded_state_dict, strict=False)
    print("Checkpoint loaded successfully.")

    dataset = hydra.utils.instantiate(cfg.data.datamodule)
    test_dataloader = dataset.test_dataloader("test1")
    train_dataloader = dataset.train_dataloader()
    mode = "train"
    creating_dataloader = train_dataloader if "train" in mode else test_dataloader

    version_description = "classification"
    label_info_file_path = f"Dataset/{mode}_{version_description}.txt"
    if os.path.exists(label_info_file_path):
        print(f"{label_info_file_path} already exists. Do you want to erase this file? (y/n)")
        a = input()
        if a == 'y':
            os.remove(label_info_file_path)
    
    database_dir = f"Dataset/{mode}/{version_description}"
    os.makedirs(database_dir, exist_ok=True)

    none_count = 0

    for batch in tqdm(creating_dataloader, desc="Batches", unit="batch"):
        point_clouds, boxes, labels, pcd_paths = batch
        for (point_cloud, box, label, pcd_path) in tqdm(
            zip(point_clouds, boxes, labels, pcd_paths),
            total=len(point_clouds),
            desc="Processing samples",
            leave=False
        ):  
            if len(box) == 0 or len(box[0]) != 6:
                gt_img = None
                label_number = 8
                none_count += 1
                if none_count != 5:
                    continue
                else:
                    none_count = 0
            else:
                gt_img = get_image(box, point_cloud, size_u=64)
                label_number = label.item()

            box = model([point_cloud.cuda()], mode='test')[0]["final_bboxes"]
            inference_img = get_image(box.cpu(), point_cloud)

            folder = os.path.basename(os.path.dirname(pcd_path))        # '20_binary'
            filename = os.path.splitext(os.path.basename(pcd_path))[0]  # '144208_test_10605'

            inference_img_path = f"{database_dir}/inference_{folder}_{filename}.npy"
            inference_img = (inference_img.cpu().numpy()).astype(np.float32)
            np.save(inference_img_path, inference_img)
            with open(label_info_file_path, "a") as f:
                f.write(f"{inference_img_path} {label_number}\n")

            if gt_img is not None:
                gt_img_path = f"{database_dir}/gt_{folder}_{filename}.npy"
                gt_img = (gt_img.cpu().numpy()).astype(np.float32)
                np.save(gt_img_path, gt_img)
                with open(label_info_file_path, "a") as f:
                    f.write(f"{gt_img_path} {label_number}\n")

    mode = "test"
    creating_dataloader = train_dataloader if "train" in mode else test_dataloader

    version_description = "classification"
    label_info_file_path = f"Dataset/{mode}_{version_description}.txt"
    if os.path.exists(label_info_file_path):
        print(f"{label_info_file_path} already exists. Do you want to erase this file? (y/n)")
        a = input()
        if a == 'y':
            os.remove(label_info_file_path)
    
    database_dir = f"Dataset/{mode}/{version_description}"
    os.makedirs(database_dir, exist_ok=True)

    none_count = 0

    for batch in tqdm(creating_dataloader, desc="Batches", unit="batch"):
        point_clouds, boxes, labels, pcd_paths = batch
        for (point_cloud, box, label, pcd_path) in tqdm(
            zip(point_clouds, boxes, labels, pcd_paths),
            total=len(point_clouds),
            desc="Processing samples",
            leave=False
        ):  
            if len(box) == 0 or len(box[0]) != 6:
                gt_img = None
                label_number = 8
                none_count += 1
                if none_count != 5:
                    continue
                else:
                    none_count = 0
            else:
                gt_img = get_image(box, point_cloud, size_u=64)
                label_number = label.item()

            box = model([point_cloud.cuda()], mode='test')[0]["final_bboxes"]
            inference_img = get_image(box.cpu(), point_cloud)

            folder = os.path.basename(os.path.dirname(pcd_path))        # '20_binary'
            filename = os.path.splitext(os.path.basename(pcd_path))[0]  # '144208_test_10605'

            inference_img_path = f"{database_dir}/inference_{folder}_{filename}.npy"
            inference_img = (inference_img.cpu().numpy()).astype(np.float32)
            np.save(inference_img_path, inference_img)
            with open(label_info_file_path, "a") as f:
                f.write(f"{inference_img_path} {label_number}\n")

            if gt_img is not None:
                gt_img_path = f"{database_dir}/gt_{folder}_{filename}.npy"
                gt_img = (gt_img.cpu().numpy()).astype(np.float32)
                np.save(gt_img_path, gt_img)
                with open(label_info_file_path, "a") as f:
                    f.write(f"{gt_img_path} {label_number}\n")

    print("DONE")

if __name__ == '__main__':
    main()
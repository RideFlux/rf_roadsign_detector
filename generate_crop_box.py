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

def get_image(bboxes, points, size_u = 17, size_v = 17):
    cx, cy, cz, dx, dy, dz = bboxes[0]

    min_x = cx - dx / 2
    max_x = cx + dx / 2
    min_z = cz - dz / 2
    max_z = cz + dz / 2
    min_y = cy - dy / 2
    max_y = cy + dy / 2

    min_z = min_z + dz / 2
    max_z = max_z + dz / 2

    min_y = cy - 3
    max_y = cy + 3

    cz = (min_z + max_z) / 2 + np.random.rand() - 0.5
    min_z = cz - dz / 2
    max_z = cz + dz / 2

    # new_dx = dx * 1
    # new_dz = dz * 2
    # min_x = cx - new_dx / 2
    # max_x = cx + new_dx / 2
    # min_z = cz - new_dz / 2
    # max_z = cz + new_dz / 2 + new_dz / 2

    
    # max_z = cz + new_dz / 2
    mask = (
        (points[:, 0] >= min_x) & (points[:, 0] <= max_x) &
        (points[:, 1] >= min_y) & (points[:, 1] <= max_y) &
        (points[:, 2] >= min_z) & (points[:, 2] <= max_z)
    )
    points = points[mask]
    if len(points) < 50 or len(points) > 1000:
        return torch.zeros((size_u, size_v))
    x, z, intensity = points[:, 0], points[:, 2], points[:, 3]
    # 17x17 grid로 매핑할 인덱스 계산
    grid_u = ((max_z - z) / (max_z - min_z) * size_u).long().clamp(0, size_u-1)
    grid_v = ((max_x - x) / (max_x - min_x) * size_v).long().clamp(0, size_v-1)
    device = points.device
    grid = torch.zeros((size_u, size_v), dtype=torch.float32, device=device)
    count = torch.zeros((size_u, size_v), dtype=torch.float32, device=device)
    for gu, gv, inten in zip(grid_u, grid_v, intensity):
        grid[gu, gv] += inten
        count[gu, gv] += 1
    grid = torch.where(count > 0, grid / count, torch.zeros_like(grid))
    return grid.cuda()


OmegaConf.register_new_resolver("sum", SumResolver, replace=True)

@hydra.main(version_base=None, config_path="configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    ckpt_path = cfg.eval_checkpoint

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

    for batch in tqdm(train_dataloader, desc="Batches", unit="batch"):
        mode = "train"
        point_clouds, boxes, labels, pcd_paths = batch
        for (point_cloud, box, label, pcd_path) in tqdm(
            zip(point_clouds, boxes, labels, pcd_paths),
            total=len(point_clouds),
            desc="Processing samples",
            leave=False
        ):
            gt_img = get_image(box, point_cloud)
            box = model([point_cloud.cuda()], mode='test')[0]["final_bboxes"]
            inference_img = get_image(box.cpu(), point_cloud)
            inference_img = inference_img / 255
            gt_img = gt_img / 255

            inference_img = zoom_tensor(inference_img, np.random.uniform(0.5, 2))
            gt_img = zoom_tensor(gt_img, np.random.uniform(0.5, 2))

            folder = os.path.basename(os.path.dirname(pcd_path))        # '20_binary'
            filename = os.path.splitext(os.path.basename(pcd_path))[0]  # '144208_test_10605'

            inference_img_path = f"Dataset/{mode}/inference_{folder}_{filename}.jpg"
            inference_img = (inference_img.cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(inference_img_path, inference_img)
            with open("Dataset/label.txt", "a") as f:
                f.write(f"{inference_img_path} {label.item()}\n")

            gt_img_path = f"Dataset/{mode}/gt_{folder}_{filename}.jpg"
            gt_img = (gt_img.cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(gt_img_path, gt_img)
            with open("Dataset/label.txt", "a") as f:
                f.write(f"{gt_img_path} {label.item()}\n")

            # # img = get_image(box, point_cloud)
            # img = img / 255
            # # # img = img - img.min()  # Normalize to [0, 1]
            # # # img = img / img.max()  # Scale to [0, 1]
            # # # img = img * 255  # Scale to [0, 255]

            # folder = os.path.basename(os.path.dirname(pcd_path))        # '20_binary'
            # filename = os.path.splitext(os.path.basename(pcd_path))[0]  # '144208_test_10605'
            # img_path = f"Dataset_inference/{mode}/{folder}_{filename}.jpg"
            # img = (img.cpu().numpy() * 255).astype(np.uint8)
            # # img = (img.cpu().numpy()).astype(np.uint8)
            
            # cv2.imwrite(img_path, img)

            # with open("Dataset_inference/label.txt", "a") as f:
            #     f.write(f"{img_path} {label.item()}\n")

    print("DONE")
        # for i in range(len(point_clouds)):
        #     point_clouds[i] = point_clouds[i].cuda()
        # with torch.no_grad():
        #     detects = model(point_clouds, mode='test')




if __name__ == '__main__':
    main()
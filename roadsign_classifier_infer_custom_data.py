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
from pcd.utils import read_pcd
from roadsign_classifier import ImageClassificationDataset, Classifier

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

    mask = (
        (points[:, 0] >= min_x) & (points[:, 0] <= max_x) &
        (points[:, 1] >= min_y) & (points[:, 1] <= max_y) ## &
        # (points[:, 2] >= min_z) & (points[:, 2] <= max_z)
    )
    points = points[mask]
    if len(points) < 50 or len(points) > 50000:
        return torch.zeros((3, size_u, size_v)).cuda()
    
    x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    # 64x32 grid로 매핑할 인덱스 계산
    grid_u = find_nearest_altitude_idx(points)
    grid_v = ((max_x - x) / (max_x - min_x) * size_v).long().clamp(0, size_v-1)

    device = points.device
    grid = torch.zeros((size_u, size_v), dtype=torch.float32, device=device)
    count = torch.zeros((size_u, size_v), dtype=torch.float32, device=device)
    ydist = torch.zeros((size_u, size_v), dtype=torch.float32, device=device)
    for gu, gv, yval, inten in zip(grid_u, grid_v, y, intensity):
        grid[gu, gv] += inten
        ydist[gu, gv] += yval
        count[gu, gv] += 1

    grid = torch.where(count > 0, grid / count, torch.zeros_like(grid))/255
    meany = -torch.where(count > 0, ydist / count, torch.zeros_like(grid))/40   ## 정규화를 위해 40으로 나눔. 포인트 범위는 0~-48까지이긴 하지만 40m 밖으로 잡히는 경우는 거의 없음
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
    model.eval()
    print("Checkpoint loaded successfully.")

    classification_model = Classifier(in_channels=3, num_classes=9).to('cuda')
    classification_checkpoint = torch.load('classfier_checkpoints/best_model.pth', map_location='cuda')
    classification_model.load_state_dict(classification_checkpoint['model_state_dict'])
    classification_model.eval()

    #### check here
    #### 여기를 pcd들 있는 폴더로 넣어주면 됨
    '''
    - pcd_dir_root
        - 0.pcd
        - 1.pcd
        ...
    '''
    pcd_dir_root = '/path/to/pcd/dir'
    if not os.path.exists(pcd_dir_root):
        print(f'cannot find directory {pcd_dir_root}. please check "pcd_dir_root"')
    pcd_file_list = sorted([x for x in os.listdir(pcd_dir_root) if x.endswith('.pcd')], key=lambda x:int(x.split('.')[0]))

    for pcd_file_name in pcd_file_list:
        pcd_file_path = os.path.join(pcd_dir_root, pcd_file_name)
        
        img = read_pcd(pcd_file_path)

        point_cloud = torch.from_numpy(img)
        output = model([point_cloud.cuda()], mode='test')[0]
        bbox = output['final_bboxes'].cpu().detach().numpy()
        score = output['final_scores'].cpu().detach().numpy()[0]

        print(pcd_file_path)
        # print(output)
        # print(bbox, score)
        img = get_image(bbox, point_cloud)

        cv2.imwrite('test.png', (img.cpu().numpy()[0]*255).astype(np.uint8))

        class_infer = classification_model(img[None, ...])[0]
        # print(class_infer, torch.argmax(class_infer))
        print("detected class : ", torch.argmax(class_infer).detach().cpu().numpy())

        a = input()
        if a == 'c':
            exit()

if __name__ == '__main__':
    main()
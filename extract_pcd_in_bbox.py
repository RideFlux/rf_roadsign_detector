import numpy as np
import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from utils.sum_resolver import SumResolver
import yaml
from pypcd import pypcd
from tqdm import tqdm

OmegaConf.register_new_resolver("sum", SumResolver, replace=True)

class FlowListDumper(yaml.Dumper):
    pass

def str_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

def represent_list_flow(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

FlowListDumper.add_representer(list, represent_list_flow)

def crop_pcd_in_bbox(pcd, bbox):
    cx, cy, cz, dx, dy, dz = bbox

    minx = cx - dx / 2
    maxx = cx + dx / 2
    miny = cy - dy / 2
    maxy = cy + dy / 2
    minz = cz - dz / 2
    maxz = cz + dz / 2

    filter = torch.logical_and(
        torch.logical_and (torch.logical_and(pcd[:, 0] > minx, pcd[:, 0] < maxx),
                           torch.logical_and(pcd[:, 1] > miny, pcd[:, 1] < maxy)),
                           torch.logical_and(pcd[:, 2] > minz, pcd[:, 2] < maxz)
    )
    
    cropped_pcd = pcd[filter]

    return cropped_pcd

def save_array_as_pcd(xyzi, pcd_path, metadata=None, intensity_type = np.uint8):
    """ Make a pointcloud object from xyzi array.
    xyzi array is cast to float32.
    shape of xyzi is (n, 4)
    """
    md = {'version': .7,
          'width': len(xyzi),
          'height': 1,
          'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
          'points': len(xyzi),
          'data': 'binary',
          'fields': ['x', 'y', 'z', 'intensity']}

    if intensity_type == np.uint8:
        md['size'] = [4, 4, 4, 1]
        md['type'] = ['F', 'F', 'F', 'U']
    else:
        md['size'] = [4, 4, 4, 4]
        md['type'] = ['F', 'F', 'F', 'F']

    md['count'] = [1, 1, 1, 1]

    if isinstance(xyzi, torch.Tensor):
        xyzi = xyzi.detach().cpu().numpy()

    xyz = xyzi[:, :3].astype(np.float32)
    intensity = xyzi[:, 3].astype(intensity_type)

    dt = np.dtype([('x', np.float32),
                    ('y', np.float32),
                    ('z', np.float32),
                    ('intensity', intensity_type)])
    pc_data = np.rec.fromarrays([xyz[:, 0], xyz[:, 1], xyz[:, 2], intensity], dtype=dt)

    if metadata is not None:
        md.update(metadata)
    
    # print(md, len(pc_data))
    pcd = pypcd.PointCloud(md, pc_data)
    pcd.save_pcd(pcd_path, compression = 'binary')
    
    return pc_data

def infer_and_crop_pcd(model, cfg, device):
    threshold = cfg.detection_threshold
    testset_name = cfg.selected_dataset

    dataset_module = hydra.utils.instantiate(cfg.data.datamodule)
    test_dataloader = dataset_module.test_dataloader(testset_name)
    
    filter_loc_sign = cfg.filter_loc_sign_only

    cnt = 0

    save_dir_root = cfg.cropped_pcd_save_dir_path
    if not os.path.exists(save_dir_root):
        os.makedirs(save_dir_root)
    
    original_pcd_dir = os.path.join(save_dir_root, 'original_pcd')
    gt_crop_pcd_dir = os.path.join(save_dir_root, 'gt_crop_pcd')
    model_crop_pcd_dir = os.path.join(save_dir_root, 'infer_crop_pcd')
    info_dir = os.path.join(save_dir_root, 'file_info')
    os.makedirs(original_pcd_dir, exist_ok=True)
    os.makedirs(gt_crop_pcd_dir, exist_ok=True)
    os.makedirs(model_crop_pcd_dir, exist_ok=True)
    os.makedirs(info_dir, exist_ok=True)

    for idx, batch in enumerate(tqdm(test_dataloader)):
        # if idx < 12: continue
        batch_pcds, batch_boxes, batch_labels = batch
        # print(batch)
        batch_len = len(batch_pcds)
        for i in range(batch_len):
            batch_pcds[i] = batch_pcds[i].to(device)
        with torch.no_grad():
            batch_detects = model(batch_pcds, mode='test')

        for i in range(batch_len):
            pcd = batch_pcds[i]
            true_bbox = batch_boxes[i]
            true_cls = batch_labels[i]
            detect_info = batch_detects[i]

            if len(true_cls) == 0 or len(true_bbox) == 0:
                continue

            ## filter_loc_sign_only 이 True일 경우 측위보정 표지판 결과만을 저장하도록
            if filter_loc_sign and true_cls != 2: continue

            ## extract point based on label
            gt_crop = crop_pcd_in_bbox(pcd, true_bbox[0])
            is_detect_valid = (detect_info['final_scores'].cpu()[0] > threshold)
            if is_detect_valid:
            # if False:
                model_crop = crop_pcd_in_bbox(pcd, detect_info['final_bboxes'][0].cpu())
            else:
                model_crop = torch.zeros((0, 4), dtype=torch.float32, device=device)

            # print(model_crop, len(model_crop))

            info_dict = {
                'gt_class' : true_cls.tolist()[0],
                'gt_bbox' : true_bbox.tolist()[0],
                'infer_score' : detect_info['final_scores'][0].cpu().tolist(),
                'infer_class' : detect_info['final_labels'][0].cpu().tolist(),
                'infer_bbox' : detect_info['final_bboxes'][0].cpu().tolist(),
            }

            # print(info_dict)

            original_pcd_file_path = os.path.join(original_pcd_dir, f"{cnt:05d}.pcd")
            gt_crop_pcd_file_path = os.path.join(gt_crop_pcd_dir, f"{cnt:05d}.pcd")
            model_crop_pcd_file_path = os.path.join(model_crop_pcd_dir, f"{cnt:05d}.pcd")
            info_file_path = os.path.join(info_dir, f"{cnt:05d}.txt")
            
            with open(info_file_path, 'w') as f:
                yaml.dump(info_dict, f, Dumper=FlowListDumper, indent=4, sort_keys=False, width=1000)
            ## 읽을 때는 yaml.safe_load 로 하면 됨

            save_array_as_pcd(pcd, original_pcd_file_path)
            save_array_as_pcd(gt_crop, gt_crop_pcd_file_path)
            ## TODO y축 두께가 워낙 작아서 (대부분 10cm 정도) 약간의 오차만으로도 포인트가 거의 없어지는 현상이 발생 중... 
            # 특히 y축이 살짝 멀게 detect 되는 경우 표지판 바로 뒤쪽이다 보니 point가 거의 찍히지 않음
            # 이거에 대한 해결 필요할 것 같음
            if len(model_crop) > 0:
                save_array_as_pcd(model_crop, model_crop_pcd_file_path)
            
            cnt += 1

            # a = input()
            # if a == 'c':
            #     exit()

### CHECK
### configs/crop_pcd.yaml에 있는 cropped_pcd_save_dir_path 경로를 설정하면 해당 경로에 추출된 pcd를 저장합니다.
### selected_dataset에서 어떤 데이터셋에 대한 pcd 추출을 진행할지 결정합니다. 이를 위해서 eval.py가 돌아갈 수 있는 상태여야 합니다 
### (README 참고. collect_pcd_data가 선행으로 실행되어서 데이터셋 구성이 완료된 상태여아 합니다)
@hydra.main(version_base=None, config_path="configs", config_name="crop_pcd.yaml")
def main(cfg: DictConfig):
    ckpt_path = cfg.checkpoint

    num_classes = cfg.data.datasets.num_classes

    print(ckpt_path)
    print(num_classes)

    model = hydra.utils.instantiate(
        cfg.model.roadsign_detector.module,
        training_model=False,
        exporting_onnx=False,
        num_classes=num_classes
    )
    model.eval()
    map_location = 'cpu'
    
    if torch.cuda.is_available():
        model = model.cuda()
        map_location = 'cuda'
    
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location=map_location)
    loaded_state_dict = checkpoint['state_dict']
    model.load_state_dict(loaded_state_dict, strict=False)

    infer_and_crop_pcd(model, cfg, map_location)

if __name__ == '__main__':
    main()
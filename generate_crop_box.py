"""Generate classifier training crops from GT and regression-predicted boxes."""

from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from evaluation.roadsign_crop import build_classifier_crop
from utils.sum_resolver import SumResolver


OmegaConf.register_new_resolver('sum', SumResolver, replace=True)


@torch.inference_mode()
def save_split_crops(model, dataloader, mode, version, device,
                     dataset_dir='Dataset', negative_stride=5):
    """Save both crop types for objects and every fifth empty-label frame."""
    root = Path(dataset_dir)
    manifest = root / f'{mode}_{version}.txt'
    if manifest.exists():
        answer = input(f'{manifest} already exists. Replace its contents? (y/n): ')
        if answer.strip().lower() != 'y':
            raise FileExistsError(f'Keeping existing manifest: {manifest}')
    output_dir = root / mode / version
    output_dir.mkdir(parents=True, exist_ok=True)
    negative_count = 0
    saved = 0
    with manifest.open('w') as handle:
        for points_list, boxes, labels, paths in tqdm(dataloader, desc=f'{mode} crops'):
            for points, gt_boxes, gt_labels, path in zip(points_list, boxes, labels, paths):
                has_gt = len(gt_boxes) > 0
                if not has_gt:
                    negative_count += 1
                    if negative_count % negative_stride:
                        continue
                class_id = int(gt_labels.item()) if has_gt else 8
                prediction = model([points.to(device)], mode='test')[0]
                bbox = prediction['final_bboxes'][0]
                source = Path(path)
                stem = f'{source.parent.name}_{source.stem}'
                crops = [('inference', build_classifier_crop(points, bbox))]
                if has_gt:
                    crops.append(('gt', build_classifier_crop(points, gt_boxes[0])))
                for kind, crop in crops:
                    output_path = output_dir / f'{kind}_{stem}.npy'
                    np.save(output_path, crop.numpy().astype(np.float32))
                    handle.write(f'{output_path} {class_id}\n')
                    saved += 1
    print(f'{manifest}: {saved} crops')
    return saved


@hydra.main(version_base=None, config_path='configs', config_name='classifier.yaml')
def main(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = hydra.utils.instantiate(
        cfg.model.roadsign_detector.module,
        training_model=False, exporting_onnx=False, num_classes=1,
    ).to(device)
    checkpoint = torch.load(cfg.regression_checkpoint, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    print(f'Regression checkpoint: {cfg.regression_checkpoint}, device: {device}')
    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    for mode, dataloader in (
            ('train', datamodule.train_dataloader()),
            ('test', datamodule.test_dataloader('test1'))):
        save_split_crops(model, dataloader, mode, cfg.crop_version, device)


if __name__ == '__main__':
    main()

"""Evaluate both roadsign models directly from PCDs, without saved crop files."""

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from evaluation.two_stage import evaluate_two_stage, save_evaluation
from evaluation.two_stage_visualization import InteractiveVisualizer, select_eval_mode
from utils.sum_resolver import SumResolver


OmegaConf.register_new_resolver('sum', SumResolver, replace=True)


@hydra.main(version_base=None, config_path='configs', config_name='eval_two_stage')
def main(cfg: DictConfig):
    eval_mode = select_eval_mode(cfg.eval_mode)
    device_name = cfg.device
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use device=cpu or device=auto.')
    for path in (cfg.regression_checkpoint, cfg.classification_checkpoint):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if cfg.data.datamodule.data_processor._target_ != 'pcd.data_processor.DataProcessor':
        raise ValueError('Use pcd.data_processor.DataProcessor to retain the original GT classes.')

    from roadsign_classifier import Classifier

    regressor = hydra.utils.instantiate(
        cfg.model.roadsign_detector.module,
        training_model=False, exporting_onnx=False, num_classes=1,
    ).to(device)
    checkpoint = torch.load(cfg.regression_checkpoint, map_location=device, weights_only=False)
    regressor.load_state_dict(checkpoint['state_dict'], strict=True)
    classifier = Classifier(in_channels=3, num_classes=9).to(device)
    checkpoint = torch.load(cfg.classification_checkpoint, map_location=device, weights_only=False)
    classifier.load_state_dict(checkpoint['model_state_dict'], strict=True)

    print(f'Regression: {cfg.regression_checkpoint}')
    print(f'Classifier: {cfg.classification_checkpoint}')
    print(f'Device: {device}')
    print(f'Dataset: {cfg.testset_name}, detection threshold: {cfg.detection_threshold}')
    print(f'Evaluation mode: {eval_mode}')
    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    if cfg.testset_name == 'val':
        dataloader = datamodule.val_dataloader()
    else:
        dataloader = datamodule.test_dataloader(cfg.testset_name)
    visualizer = (InteractiveVisualizer(
        cfg.inference_img_path, cfg.model.roadsign_detector.point_cloud_range,
    ) if eval_mode == 1 else None)
    records, summary = evaluate_two_stage(
        regressor, classifier, dataloader, device,
        detection_range=cfg.model.roadsign_detector.point_cloud_range,
        detection_threshold=cfg.detection_threshold,
        distance_threshold=cfg.distance_threshold,
        max_samples=cfg.max_samples,
        show_progress=eval_mode == 2,
        on_sample=visualizer,
    )
    summary['configuration'] = dict(
        regression_checkpoint=str(cfg.regression_checkpoint),
        classification_checkpoint=str(cfg.classification_checkpoint),
        testset_name=str(cfg.testset_name), detection_threshold=float(cfg.detection_threshold),
        dataset_root=str(cfg.data.datasets.dataset_root_dir),
        pickle_files=OmegaConf.to_container(cfg.data.datamodule.data_processor.pickle_files),
        stride=int(cfg.data.datamodule.data_processor.stride),
        max_samples=cfg.max_samples,
        device=str(device),
        eval_mode=eval_mode,
    )
    save_evaluation(records, summary, cfg.results_dir)
    if summary['stopped_early']:
        print('시각화를 종료했습니다. 지금까지 평가한 프레임만 결과에 포함됩니다.')
    regression, classification = summary['regression'], summary['classification']
    print(f"\nSamples: {summary['samples']} (objects: {summary['gt_objects']}, no object: {summary['gt_no_object']})")
    print(f"Stage 1 detection: precision={regression['precision']:.4%}, recall={regression['recall']:.4%}")
    print(f"Stage 1 localization incl. no object: {regression['localization_accuracy_including_no_object']:.4%}")
    print(f"Final class accuracy: {classification['accuracy']:.4%}, macro F1: {classification['macro_f1']:.4%}")
    print(f"Class + center error <= {cfg.distance_threshold} m: {summary['end_to_end_accuracy']:.4%}")
    print('Confusion matrix (rows=true, columns=predicted):')
    for row in classification['confusion_matrix']:
        print(row)
    for row in classification['per_class']:
        print(f"{row['name']}: precision={row['precision']:.4f}, recall={row['recall']:.4f}, F1={row['f1']:.4f}, n={row['support']}")
    print(f'Results: {Path(cfg.results_dir).resolve()}')
    from evaluation.utils import create_hyperlink
    create_hyperlink(str(Path(cfg.results_dir) / 'confusion_matrix.png'), 'Confusion matrix image')
    print(f'Confusion matrix: {(Path(cfg.results_dir) / "confusion_matrix.png").resolve()}')


if __name__ == '__main__':
    main()

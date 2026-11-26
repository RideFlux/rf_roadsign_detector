"""Online regression -> predicted-box crop -> classification evaluation."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from evaluation.roadsign_crop import build_classifier_crop
from evaluation.plots import save_confusion_matrix


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def summarize_predictions(records, no_object_class=8, distance_threshold=0.1):
    if not records:
        raise ValueError("The evaluation dataset is empty")
    n_classes = no_object_class + 1
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    detection = dict(tp=0, fp=0, fn=0, tn=0)
    errors = []
    localized = 0
    joint_correct = 0
    for row in records:
        gt_object = row['gt_class'] != no_object_class
        detected = row['detected']
        detection['tp' if gt_object and detected else
                  'fn' if gt_object else 'fp' if detected else 'tn'] += 1
        confusion[row['gt_class'], row['pred_class']] += 1
        close = (row['center_xy_error_m'] is not None
                 and row['center_xy_error_m'] <= distance_threshold)
        if gt_object and detected:
            errors.append(row['center_xy_error_m'])
            localized += int(close)
        joint_correct += int(
            (not gt_object and row['pred_class'] == no_object_class)
            or (gt_object and detected and close and row['pred_class'] == row['gt_class'])
        )

    per_class = []
    for label in range(n_classes):
        tp = int(confusion[label, label])
        fp = int(confusion[:, label].sum()) - tp
        fn = int(confusion[label].sum()) - tp
        per_class.append(dict(
            class_id=label, name='No object' if label == no_object_class else f'Class_{label + 1}',
            support=tp + fn, precision=_ratio(tp, tp + fp), recall=_ratio(tp, tp + fn),
            f1=_ratio(2 * tp, 2 * tp + fp + fn),
        ))
    tp, fp, fn, tn = (detection[key] for key in ('tp', 'fp', 'fn', 'tn'))
    active_classes = [i for i in range(n_classes)
                      if confusion[i].sum() + confusion[:, i].sum() > 0]
    return dict(
        samples=len(records), gt_objects=tp + fn, gt_no_object=tn + fp,
        no_object_class=no_object_class, distance_threshold_m=distance_threshold,
        regression=dict(
            **detection, precision=_ratio(tp, tp + fp), recall=_ratio(tp, tp + fn),
            detection_accuracy=_ratio(tp + tn, len(records)),
            localization_correct=localized,
            localization_recall=_ratio(localized, tp + fn),
            localization_accuracy_including_no_object=_ratio(localized + tn, len(records)),
            center_xy_error_mean_m=float(np.mean(errors)) if errors else None,
            center_xy_error_median_m=float(np.median(errors)) if errors else None,
        ),
        classification=dict(
            accuracy=_ratio(int(np.trace(confusion)), len(records)),
            macro_f1=float(np.mean([per_class[i]['f1'] for i in active_classes])),
            macro_f1_classes=active_classes,
            macro_f1_all_classes=float(np.mean([row['f1'] for row in per_class])),
            per_class=per_class, confusion_matrix=confusion.tolist(),
        ),
        end_to_end_correct=joint_correct,
        end_to_end_accuracy=_ratio(joint_correct, len(records)),
    )


@torch.inference_mode()
def evaluate_two_stage(regressor, classifier, dataloader, device, detection_range,
                       detection_threshold=0.5, distance_threshold=0.1,
                       no_object_class=8, max_samples=None, show_progress=True,
                       on_sample=None):
    """Evaluate every PCD once, including empty labels and the final batch.

    GT classes must come from pcd.data_processor.DataProcessor, which preserves
    object classes. Out-of-range GT is counted as No object, like eval.py.
    Distance is Euclidean distance between the actual box centers (cx, cy).
    on_sample(points, record) may return False to stop and report the evaluated
    prefix, including the last displayed frame.
    """
    if max_samples is not None and max_samples <= 0:
        raise ValueError('max_samples must be positive or null')
    if distance_threshold < 0 or not 0 <= detection_threshold <= 1:
        raise ValueError('Invalid distance/detection threshold')
    regressor.eval()
    classifier.eval()
    records = []
    stopped_early = False
    for point_clouds, boxes, labels, pcd_paths in tqdm(
            dataloader, desc='Two-stage evaluation', disable=not show_progress):
        if max_samples is not None:
            remaining = max_samples - len(records)
            point_clouds, boxes, labels, pcd_paths = (
                values[:remaining] for values in (point_clouds, boxes, labels, pcd_paths))
        predictions = regressor([points.to(device) for points in point_clouds], mode='test')
        if len(predictions) != len(point_clouds):
            raise ValueError('Regression output count does not match the batch size')
        batch_records, crops, crop_indices = [], [], []
        for points, gt_boxes, gt_labels, pcd_path, prediction in zip(
                point_clouds, boxes, labels, pcd_paths, predictions):
            gt_boxes = torch.as_tensor(gt_boxes).cpu()
            gt_labels = torch.as_tensor(gt_labels).cpu()
            if len(gt_boxes) > 1 or len(gt_boxes) != len(gt_labels):
                raise ValueError(f'Expected zero or one GT object: {pcd_path}')
            gt_bbox = gt_boxes[0] if len(gt_boxes) else None
            raw_gt_class = int(gt_labels[0]) if gt_bbox is not None else no_object_class
            if gt_bbox is not None and not 0 <= raw_gt_class < no_object_class:
                raise ValueError(f'Invalid GT class {raw_gt_class}: {pcd_path}')
            in_range = bool(gt_bbox is not None
                            and detection_range[0] <= gt_bbox[0] <= detection_range[3]
                            and detection_range[1] <= gt_bbox[1] <= detection_range[4])
            gt_class = raw_gt_class if in_range else no_object_class

            pred_boxes = prediction['final_bboxes'].detach().cpu()
            scores = prediction['final_scores'].detach().cpu().reshape(-1)
            if len(pred_boxes) != len(scores):
                raise ValueError(f'Prediction box/score count mismatch: {pcd_path}')
            pred_bbox, score = None, None
            if len(scores):
                best = int(scores.argmax())
                pred_bbox, score = pred_boxes[best], float(scores[best])
                if (pred_bbox.shape != (6,) or not torch.isfinite(pred_bbox).all()
                        or (pred_bbox[3:] <= 0).any() or not np.isfinite(score)):
                    raise ValueError(f'Invalid regression output: {pcd_path}')
            detected = score is not None and score > detection_threshold
            center_error = (float(torch.linalg.vector_norm(pred_bbox[:2] - gt_bbox[:2]))
                            if in_range and pred_bbox is not None else None)
            row = dict(
                pcd_path=str(pcd_path), raw_gt_class=raw_gt_class,
                gt_in_range=in_range, gt_class=gt_class, pred_class=no_object_class,
                detected=detected, regression_score=score,
                classifier_ran=False, classifier_score=None,
                center_xy_error_m=center_error,
                gt_bbox=gt_bbox.tolist() if gt_bbox is not None else None,
                pred_bbox=pred_bbox.tolist() if pred_bbox is not None else None,
            )
            if detected:
                crop_indices.append(len(batch_records))
                crops.append(build_classifier_crop(points, pred_bbox))
            batch_records.append(row)
        if crops:
            logits = classifier(torch.stack(crops).to(device))
            if logits.shape != (len(crops), no_object_class + 1) or not torch.isfinite(logits).all():
                raise ValueError('Invalid classifier output shape or non-finite logits')
            probabilities = logits.softmax(dim=1)
            for index, probabilities_row in zip(crop_indices, probabilities):
                row = batch_records[index]
                row['pred_class'] = int(probabilities_row.argmax())
                row['classifier_score'] = float(probabilities_row.max())
                row['classifier_ran'] = True
        for points, row in zip(point_clouds, batch_records):
            records.append(row)
            if on_sample is not None and on_sample(points, row) is False:
                stopped_early = True
                break
        if stopped_early:
            break
        if max_samples is not None and len(records) >= max_samples:
            break
    summary = summarize_predictions(records, no_object_class, distance_threshold)
    summary['stopped_early'] = stopped_early
    return records, summary


def save_evaluation(records, summary, output_dir):
    """Save machine-readable metrics, per-PCD predictions, and a confusion plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'metrics.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    with (output_dir / 'predictions.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        for record in records:
            row = dict(record)
            for key in ('gt_bbox', 'pred_bbox'):
                row[key] = json.dumps(row[key])
            writer.writerow(row)
    classification = summary['classification']
    names = [row['name'] for row in classification['per_class']]
    save_confusion_matrix(classification['confusion_matrix'], names,
                          output_dir / 'confusion_matrix.png', 'Two-stage confusion matrix')

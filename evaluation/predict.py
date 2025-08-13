import cv2
import hydra
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import torch
from rich.progress import track
from evaluation.utils import compute_distance, create_hyperlink
from pointpillars.utils.vis_o3d import draw_bev, vis_pc

def inference_test_imgs_qtt(model, cfg, mode):

    score_list = []
    result_list = []
    
    threshold = cfg.detection_threshold
    detection_range = cfg.model.roadsign_detector.point_cloud_range
    num_classes = cfg.data.datasets.num_classes
    
    dataset = hydra.utils.instantiate(cfg.data.datamodule)
    test_dataloader = dataset.eval_dataloader()
    data_iter = iter(test_dataloader)
    data = next(data_iter)
    if mode == 1:
        create_hyperlink(cfg.inference_img_path, 'Inference Image')
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_len = len(data_iter)
    
    seq = track(range(batch_len), description="Processing batches") if mode == 3 else range(batch_len)

    for _ in seq:
        point_clouds, boxes, labels = data
        
        for i in range(len(point_clouds)):
            point_clouds[i] = point_clouds[i].to(device)
        with torch.no_grad():
            detects = model(point_clouds, mode='test')
        
        for i in range(len(point_clouds)):
            point_cloud = point_clouds[i].detach().cpu().numpy()
            gt_bboxes = boxes[i].detach().cpu().numpy()
            gt_labels = labels[i].detach().cpu().numpy()
            pred_bbox = detects[i]["final_bboxes"][0].detach().cpu().numpy()
            pred_label = detects[i]["final_labels"][0].detach().cpu().numpy()
            pred_score = detects[i]["final_scores"][0].detach().cpu().numpy()

            score_list.append(pred_score)
            if len(gt_bboxes) > 0:
                gt_bbox = gt_bboxes[0]
                gt_label = gt_labels[0]
                distance = compute_distance(pred_bbox, gt_bbox)
                
                gt_in_range = (gt_bbox[0] >= detection_range[0] and gt_bbox[0] <= detection_range[3] and
                                gt_bbox[1] >= detection_range[1] and gt_bbox[1] <= detection_range[4])
                
                if not gt_in_range:
                    gt_label = num_classes
                if pred_score <= threshold:
                    pred_label = num_classes
                    
                result_list.append((pred_label, gt_label, distance)) 
            else:
                if pred_score <= threshold:
                    pred_label = num_classes
                result_list.append((pred_label, num_classes, 1000.0))
            
            if mode == 1:
                bev = draw_bev(point_cloud,
                               gt_bbox,
                               pred_bbox)
                cv2.imwrite(cfg.inference_img_path, bev)
                input()

            if mode == 2:
                vis_pc(point_cloud,
                               pred_bbox,
                               pred_label)
                t = input("Press Enter to continue, or 'q' to quit")
                if(t.strip().lower() == 'q'):
                    return

        try:
            data = next(data_iter)
        except StopIteration:
            data = None
            break
    
    print("\n[ EVALUATION RESULTS ]\n")
    
    result_list = np.array(result_list)
    score_list = np.array(score_list)
    
    distance_array = ((result_list[:, 2] <= 0.1) & (score_list > threshold)) | ((score_list <= threshold) & (result_list[:, 1] == num_classes))
    distance_true_count = np.sum(distance_array)
    distance_rate = distance_true_count / len(distance_array) if len(distance_array) > 0 else 0.0
    print(f"중심간 거리 10cm 이하: {distance_rate:.4f} ({distance_true_count}/{len(distance_array)})")

    class_equal_array = result_list[:, 0] == result_list[:, 1]
    class_equal_count = np.sum(class_equal_array)
    class_equal_rate = class_equal_count / len(class_equal_array) if len(class_equal_array) > 0 else 0.0
    print(f"클래스 정확도: {class_equal_rate:.4f} ({class_equal_count}/{len(class_equal_array)})")

    all_correct_array = (result_list[:, 0] == result_list[:, 1]) & ((result_list[:, 2] <= 0.1) & (score_list > threshold))  | ((score_list <= threshold) & (result_list[:, 1] == num_classes))
    all_correct_count = np.sum(all_correct_array)
    all_correct_rate = all_correct_count / len(all_correct_array) if len(all_correct_array) > 0 else 0.0
    print(f"10cm 이하 ∧ 클래스 정답: {all_correct_rate:.4f} ({all_correct_count}/{len(all_correct_array)})")
        
    # Create histogram for distance
    distances = [item[2] for item in result_list if item[2] <= 1.0]
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Distance (m)')
    plt.ylabel('Frequency')
    plt.title('Distance Histogram between Predicted and Ground Truth Centers')
    plt.grid(True, alpha=0.3)
    plt.savefig('distance_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()
    create_hyperlink('distance_histogram.png', 'Distance Histogram')

    
    # Create confusion matrix for classes
    pred_classes = [item[0] for item in result_list]
    true_classes = [item[1] for item in result_list]
    
    # Get unique classes
    all_classes = sorted(list(set(pred_classes + true_classes)))
    n_classes = len(all_classes)
    
    # Create confusion matrix
    confusion_mat = np.zeros((n_classes, n_classes), dtype=np.int32)
    for pred, true in zip(pred_classes, true_classes):
        pred_idx = all_classes.index(pred)
        true_idx = all_classes.index(true)
        confusion_mat[true_idx, pred_idx] += 1
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues, norm=matplotlib.colors.LogNorm())
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add class labels
    class_names = [f'Class_{int(i)+1}' if i < num_classes else 'No object' for i in all_classes]
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = confusion_mat.max() / 20.
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, format(confusion_mat[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if confusion_mat[i, j] > thresh else "black")
    
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    create_hyperlink('confusion_matrix.png', 'Confusion Matrix')
    
    # Print confusion matrix statistics
    print(f"\nConfusion Matrix:")
    print(f"Classes: {class_names}")
    print(confusion_mat)
    
    # Calculate per-class metrics
    print(f"\nPer-class metrics:")
    for i, class_name in enumerate(class_names):
        tp = confusion_mat[i, i]
        fp = np.sum(confusion_mat[:, i]) - tp
        fn = np.sum(confusion_mat[i, :]) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"  {class_name}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    # Create histogram for scores
    plt.figure(figsize=(10, 6))
    plt.hist(score_list, bins=30, alpha=0.7, edgecolor='black', color='orange')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Confidence Score Histogram')
    plt.grid(True, alpha=0.3)
    plt.savefig('score_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()
    create_hyperlink('score_histogram.png', 'Score Histogram')
    
    # Print score statistics
    scores_array = np.array(score_list)
    print(f"\nScore Statistics:")
    print(f"  Mean: {np.mean(scores_array):.4f}")
    print(f"  Std: {np.std(scores_array):.4f}")
    print(f"  Min: {np.min(scores_array):.4f}")
    print(f"  Max: {np.max(scores_array):.4f}")

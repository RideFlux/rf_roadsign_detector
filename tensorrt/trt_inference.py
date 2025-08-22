# TensorRT 8.5.3 환경에서 실행되는 코드입니다.

import os
import pickle
import sys
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from pypcd.pypcd import PointCloud
from rich.progress import track

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.utils import compute_distance, create_hyperlink

pycuda.autoinit # 생략 가능
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
stream = cuda.Stream()

def pcd_to_input(pcd_file_path: str) -> np.ndarray:
    pcd = PointCloud.from_path(pcd_file_path)
    x = pcd.pc_data['x'].astype(np.float32)
    y = pcd.pc_data['y'].astype(np.float32)
    z = pcd.pc_data['z'].astype(np.float32)
    i = pcd.pc_data['intensity'].astype(np.float32)
    pts = np.stack([x, y, z, i], axis=1)
    num_points = pts.shape[0]
    
    pts = pts[None, ...]  # (1, N, 4) : 배치 차원을 추가합니다.
    
    # Zero-padding: VoxelGeneratorPlugin이 static input shape을 요구하고 있으므로, 이를 맞춰줍니다. 
    target_points = 200000
    if num_points < target_points:
        pad = np.zeros((1, target_points - num_points, 4), dtype=np.float32)
        pts = np.concatenate([pts, pad], axis=1)
    elif num_points > target_points:
        pts = pts[:, :target_points, :]
    
    return pts, num_points  

def _read_data_from_pickle(data_item):
    # Extract data from pickle item
    pcd_path = data_item['pcd_path']
    label_data = data_item['label']
    reversed_flag = data_item['reversed']
    
    input = pcd_to_input(pcd_path)
    gt_bboxes = np.zeros([0, 6], dtype=np.float32)
    gt_cls = np.zeros([0], dtype=np.int32)
    
    if label_data is not None:
        try:
            class_id = label_data['class']
            bbox_coords = label_data['bbox']
            
            gt_cls = np.array([class_id-1], dtype=np.int32)
            
            coords_flat = bbox_coords.flatten()
            
            dim = np.array([coords_flat[3] - coords_flat[0],    # x_max - x_min
                            coords_flat[7] - coords_flat[4],     # y_max - y_min  
                            coords_flat[14] - coords_flat[11]], dtype=np.float32)   # z_max - z_min
            
            loc = np.array([coords_flat[0] + (coords_flat[3] - coords_flat[0])/2,   # x_center
                            coords_flat[4] + (coords_flat[7] - coords_flat[4])/2,    # y_center
                            coords_flat[11]], dtype=np.float32)                      # z_min
            
            gt_bboxes = np.concatenate([loc, dim]).reshape(1, 6).astype(np.float32)
        except Exception as e:
            return input, gt_bboxes, gt_cls, reversed_flag
    
    if reversed_flag:
        input[0][:, :, 1] = -input[0][:, :, 1]  # new x = old y
        gt_bboxes = np.zeros([0, 6], dtype=np.float32)
        gt_cls = np.zeros([0], dtype=np.int32)
        
    return input, gt_bboxes, gt_cls, reversed_flag

def main(trt_path: str):
    with open('/home/yongwooklee/LIDAR_DATA/test1_set.pkl', 'rb') as f:
        dataset = pickle.load(f)
    
    
    # ----- 1. 엔진 로드 -----
    with open(trt_path, "rb") as f:
        trt.init_libnvinfer_plugins(None, "") # TensorRT 플러그인을 사용하기 위해 필요합니다.
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

    # ----- 5. 출력 버퍼 크기 확인 -----
    boxes_shape  = tuple(context.get_tensor_shape("final_boxes"))
    scores_shape = tuple(context.get_tensor_shape("final_scores"))
    labels_shape = tuple(context.get_tensor_shape("final_labels"))

    final_boxes  = np.empty(boxes_shape,  dtype=trt.nptype(engine.get_tensor_dtype("final_boxes")))
    final_scores = np.empty(scores_shape, dtype=trt.nptype(engine.get_tensor_dtype("final_scores")))
    final_labels = np.empty(labels_shape, dtype=trt.nptype(engine.get_tensor_dtype("final_labels")))

    d_points     = cuda.mem_alloc(np.empty((1, 200000, 4), dtype=np.float32).nbytes)
    d_num_points = cuda.mem_alloc(np.empty((1,), dtype=np.int32).nbytes)
    d_boxes      = cuda.mem_alloc(final_boxes.nbytes)
    d_scores     = cuda.mem_alloc(final_scores.nbytes)
    d_labels     = cuda.mem_alloc(final_labels.nbytes)

    bindings = [int(d_points), int(d_num_points), int(d_scores), int(d_labels), int(d_boxes)]

    score_list = []
    result_list = []
    detection_range = [-8, -48, -3, 0, 0, 1]
    num_classes = 8
    threshold = 0.5

    for item in track(dataset, description=f"Processing test data", finished_style="rgb(249,38,114)"):
        processed = _read_data_from_pickle(item)
        input, gt_bboxes, gt_labels, reversed_flag = processed
        points, num_points = input
        
        # ----- 7. 추론 -----
        # Host(CPU) to Device(GPU)
        cuda.memcpy_htod_async(d_points, points, stream)
        cuda.memcpy_htod_async(d_num_points, np.array([num_points], dtype=np.int32), stream)

        # 실행
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        # Device(GPU) to Host(CPU)
        cuda.memcpy_dtoh_async(final_boxes,  d_boxes,  stream)
        cuda.memcpy_dtoh_async(final_scores, d_scores, stream)
        cuda.memcpy_dtoh_async(final_labels, d_labels, stream)

        stream.synchronize()
        cuda.Context.synchronize()

        pred_bbox = final_boxes[0]
        pred_label = final_labels[0]
        pred_score = final_scores[0]
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
                
            # if(gt_label != num_classes and pred_label == num_classes):
            #     print(gt_bbox, gt_label, pred_bbox, pred_label, pred_score, reversed_flag)
    
            result_list.append((pred_label, gt_label, distance)) 
        else:
            
            gt_label = num_classes
            if pred_score <= threshold:
                pred_label = num_classes
            result_list.append((pred_label, num_classes, 1000.0))
        
        if gt_label != 8 and pred_label == 8:
            print(item, gt_bbox, gt_label, pred_bbox, pred_label, pred_score, reversed_flag)
            print()
        
        
        
        score_list.append(pred_score)

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

    
if __name__ == "__main__":
    import argparse
    import pathlib
    
    parser = argparse.ArgumentParser(description="TensorRT Inference for Point Cloud Data")
    # parser.add_argument("--pcd", type=pathlib.Path, required=True, help="Path to the PCD file")
    parser.add_argument("--trt", type=pathlib.Path, default="./final.trt", help="Path to the TensorRT engine file")
    args = parser.parse_args()

    main(args.trt)
import os
import time
import numpy as np
import argparse
import pathlib

import pandas as pd

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

from rich.progress import track

pycuda.autoinit # 생략 가능
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
stream = cuda.Stream()
    
def pcd_to_input(pcd_file_path: str) -> np.ndarray:
    from pypcd.pypcd import PointCloud
    
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
    
    return pts[None, ...], num_points  

with open("./final.trt", "rb") as f:
    trt.init_libnvinfer_plugins(None, "") # TensorRT 플러그인을 사용하기 위해 필요합니다.
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    
    
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
    
def measure_time(pcd_path: str) -> tuple:
    total_start_event = time.perf_counter()
    
    points, num_points = pcd_to_input(pcd_path)

    # Host(CPU) to Device(GPU)    
    memcpy_h2d_start_event = time.perf_counter()
    cuda.memcpy_htod_async(d_points, points, stream)
    cuda.memcpy_htod_async(d_num_points, np.array([num_points], dtype=np.int32), stream)
    stream.synchronize()
    memcpy_h2d_end_event = time.perf_counter()

    # 실행
    inference_start_event = time.perf_counter()
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    stream.synchronize()
    inference_end_event = time.perf_counter()

    # Device(GPU) to Host(CPU)
    memcpy_d2h_start_event = time.perf_counter()
    cuda.memcpy_dtoh_async(final_boxes,  d_boxes,  stream)
    cuda.memcpy_dtoh_async(final_scores, d_scores, stream)
    cuda.memcpy_dtoh_async(final_labels, d_labels, stream)
    stream.synchronize()
    memcpy_d2h_end_event = time.perf_counter()
    
    cuda.Context.synchronize()
    total_end_event = time.perf_counter()
    
    memcpy_h2d_time = memcpy_h2d_end_event - memcpy_h2d_start_event
    inference_time = inference_end_event - inference_start_event
    memcpy_d2h_time = memcpy_d2h_end_event - memcpy_d2h_start_event
    total_time = total_end_event - total_start_event
    
    return total_time, inference_time, memcpy_h2d_time, memcpy_d2h_time
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="TensorRT Inference for Point Cloud Data")
    parser.add_argument("--dir", type=pathlib.Path, default='/home/yongwooklee/LIDAR_DATA', help="Path to the directory containing PCD files")
    parser.add_argument("--trt", type=pathlib.Path, default="./final.trt", help="Path to the TensorRT engine file")
    args = parser.parse_args()

    root_dir = '/home/yongwooklee/LIDAR_DATA'
    
    total_time_list = []
    inference_time_list = []
    memcpy_h2d_time_list = []
    memcpy_d2h_time_list = []
    
    pcd_file_list = []
    
    for dirpath, dirnames, filenames in os.walk(args.dir):
        for filename in filenames:
            if filename.endswith('.pcd'):
                full_path = os.path.join(dirpath, filename)
                pcd_file_list.append(full_path)

    for full_path in track(pcd_file_list):
        total_time, inference_time, memcpy_h2d_time, memcpy_d2h_time = measure_time(full_path)
        total_time_list.append(total_time)
        inference_time_list.append(inference_time)
        memcpy_h2d_time_list.append(memcpy_h2d_time)
        memcpy_d2h_time_list.append(memcpy_d2h_time)
    
    # Create DataFrame with timing data
    df = pd.DataFrame({
        'total_time': total_time_list[1:],
        'inference_time': inference_time_list[1:],
        'memcpy_h2d_time': memcpy_h2d_time_list[1:],
        'memcpy_d2h_time': memcpy_d2h_time_list[1:],
    })

    # Save to CSV
    df.to_csv('timing_results.csv', index=False)
    print(f"Results saved to timing_results.csv ({len(df)} samples)")

    print(f"Total Time: {np.mean(total_time_list[1:]):.4f} ± {np.std(total_time_list[1:]):.4f} seconds, WCET: {max(total_time_list[1:]):.4f} seconds")
    print(f"Inference Time: {np.mean(inference_time_list[1:]):.4f} ± {np.std(inference_time_list[1:]):.4f} seconds, WCET: {max(inference_time_list[1:]):.4f} seconds")
    print(f"Memcpy H2D Time: {np.mean(memcpy_h2d_time_list[1:]):.4f} ± {np.std(memcpy_h2d_time_list[1:]):.4f} seconds, WCET: {max(memcpy_h2d_time_list[1:]):.4f} seconds")
    print(f"Memcpy D2H Time: {np.mean(memcpy_d2h_time_list[1:]):.4f} ± {np.std(memcpy_d2h_time_list[1:]):.4f} seconds, WCET: {max(memcpy_d2h_time_list[1:]):.4f} seconds")
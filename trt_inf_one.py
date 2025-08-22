import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from pypcd.pypcd import PointCloud

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
    
    return pts[None, ...], num_points  

def main(pcd_path: str, trt_path: str):
    # ----- 1. 엔진 로드 -----
    with open(trt_path, "rb") as f:
        trt.init_libnvinfer_plugins(None, "") # TensorRT 플러그인을 사용하기 위해 필요합니다.
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
    
    # ----- 2. 입력 데이터 준비 -----
    points, num_points = pcd_to_input(pcd_path)

    # ----- 4. shape 세팅 -----
    context.set_input_shape("points", points.shape)
    context.set_input_shape("num_points", (1,))

    # ----- 5. 출력 버퍼 크기 확인 -----
    boxes_shape  = tuple(context.get_tensor_shape("final_boxes"))
    scores_shape = tuple(context.get_tensor_shape("final_scores"))
    labels_shape = tuple(context.get_tensor_shape("final_labels"))

    final_boxes  = np.empty(boxes_shape,  dtype=trt.nptype(engine.get_tensor_dtype("final_boxes")))
    final_scores = np.empty(scores_shape, dtype=trt.nptype(engine.get_tensor_dtype("final_scores")))
    final_labels = np.empty(labels_shape, dtype=trt.nptype(engine.get_tensor_dtype("final_labels")))

    # ----- 6. GPU 메모리 할당 -----
    d_points     = cuda.mem_alloc(points.nbytes)
    d_num_points = cuda.mem_alloc(np.array([num_points], dtype=np.int32).nbytes)
    d_boxes      = cuda.mem_alloc(final_boxes.nbytes)
    d_scores     = cuda.mem_alloc(final_scores.nbytes)
    d_labels     = cuda.mem_alloc(final_labels.nbytes)

    bindings = [int(d_points), int(d_num_points), int(d_scores), int(d_labels), int(d_boxes)]
    
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
    pred = (final_boxes, final_scores, final_labels)

    print("Boxes:", pred[0][0])
    print("Scores:", pred[1][0])
    print("Labels:", pred[2][0]+1) # 실제 클래스는 1-based index이므로 +1
    
if __name__ == "__main__":
    import argparse
    import pathlib
    
    parser = argparse.ArgumentParser(description="TensorRT Inference for Point Cloud Data")
    parser.add_argument("--pcd", type=pathlib.Path, required=True, help="Path to the PCD file")
    parser.add_argument("--trt", type=pathlib.Path, default="./final.trt", help="Path to the TensorRT engine file")
    args = parser.parse_args()

    main(args.pcd, args.trt)
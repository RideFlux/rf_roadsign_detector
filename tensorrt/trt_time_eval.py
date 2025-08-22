# TensorRT 8.5.3 환경에서 실행되는 코드입니다.

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import time
import pandas as pd

# =========================
# 1. Engine 로드
# =========================
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
with open("./final.trt", "rb") as f:
    trt.init_libnvinfer_plugins(None, "")
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

# =========================
# 2. 버퍼 준비
# =========================
input_shape = (1, 200000, 4)   # 고정
d_points = cuda.mem_alloc(np.empty(input_shape, dtype=np.float32).nbytes)
d_num_points = cuda.mem_alloc(np.empty((1,), dtype=np.int32).nbytes)

# 출력 버퍼
final_boxes  = np.empty(tuple(context.get_tensor_shape("final_boxes")),  dtype=trt.nptype(engine.get_tensor_dtype("final_boxes")))
final_scores = np.empty(tuple(context.get_tensor_shape("final_scores")), dtype=trt.nptype(engine.get_tensor_dtype("final_scores")))
final_labels = np.empty(tuple(context.get_tensor_shape("final_labels")), dtype=trt.nptype(engine.get_tensor_dtype("final_labels")))

d_boxes  = cuda.mem_alloc(final_boxes.nbytes)
d_scores = cuda.mem_alloc(final_scores.nbytes)
d_labels = cuda.mem_alloc(final_labels.nbytes)

bindings = [int(d_points), int(d_num_points), int(d_scores), int(d_labels), int(d_boxes)]
stream = cuda.Stream()

# =========================
# 3. 랜덤 입력 생성
# =========================
def make_random_input():
    pts = np.random.rand(*input_shape).astype(np.float32)
    num_points = np.array([200000], dtype=np.int32)
    return pts, num_points

# =========================
# 4. 측정 루프
# =========================
def measure(num_iters=1000, warmup=50):
    e2e_times = []
    inf_times = []

    for i in range(num_iters + warmup):
        pts, num_points = make_random_input()

        # ---------------- E2E 측정 ----------------
        start_e2e = time.perf_counter()

        cuda.memcpy_htod_async(d_points, pts, stream)
        cuda.memcpy_htod_async(d_num_points, num_points, stream)

        start_event = cuda.Event()
        end_event = cuda.Event()

        start_event.record(stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        end_event.record(stream)

        cuda.memcpy_dtoh_async(final_boxes,  d_boxes,  stream)
        cuda.memcpy_dtoh_async(final_scores, d_scores, stream)
        cuda.memcpy_dtoh_async(final_labels, d_labels, stream)

        stream.synchronize()
        end_e2e = time.perf_counter()

        # inference-only (CUDA 이벤트)
        inf_ms = start_event.time_till(end_event)
        e2e_ms = (end_e2e - start_e2e) * 1000.0

        if i >= warmup:  # warm-up 제외
            inf_times.append(inf_ms)
            e2e_times.append(e2e_ms)

    return inf_times, e2e_times


if __name__ == "__main__":
    inf_times, e2e_times = measure(num_iters=1000, warmup=50)

    df = pd.DataFrame({
        "e2e_ms": e2e_times
    })
    df.to_csv("timing_random.csv", index=False, header=False)

    print(f"Saved {len(inf_times)} samples to timing_random.csv")
    print(f"[Inference-only] mean={np.mean(inf_times):.3f} ms, std={np.std(inf_times):.3f} ms, max={np.max(inf_times):.3f} ms")
    print(f"[E2E]           mean={np.mean(e2e_times):.3f} ms, std={np.std(e2e_times):.3f} ms, max={np.max(e2e_times):.3f} ms")

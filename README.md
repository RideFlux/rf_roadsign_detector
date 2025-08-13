# Roadsign Detector

> 모델 논문: [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lang_PointPillars_Fast_Encoders_for_Object_Detection_From_Point_Clouds_CVPR_2019_paper.pdf)

## 개발 환경

아래 환경은 필수적인 환경은 아니며, 모델 학습 시에는 적당한 버전의 Python과 CUDA 코드를 돌릴 수 있는 GPU, 모델 평가 시에는 Python만 있어도 됩니다.

- Ubuntu 20.04.6 LTS (64-bit)
- 12th Gen Intel® Core™ i7-12700 × 20
- NVIDIA GeForce RTX 3070/PCIe/SSE2
- Python 3.8.20
- CUDA 12.8

## 학습/평가 전 준비 과정

### 1. 필수 패키지 설치

> [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install)와 같은 가상환경 도구를 사용하는 것을 추천합니다.

```bash
pip install -r requirements.txt
```

아래 코드를 가상환경 터미널에 붙여넣기하여 설치 여부를 확인하실 수 있습니다.

```bash
python -c 'import pytorch_lightning as pl; print(f"PyTorch Lightning version: {pl.__version__}"); import torch; print(f"PyTorch version: {torch.__version__}"); print(f"CUDA available: {torch.cuda.is_available()}"); import hydra; print(f"Hydra version: {hydra.__version__}"); '
```

아래처럼 각 모듈의 버전이 정상적으로 나온다면 성공입니다. (버전이 조금씩 다르더라도 괜찮습니다.)

```
PyTorch Lightning version: 2.4.0
PyTorch version: 2.4.1+cu121
CUDA available: True
Hydra version: 1.3.2
```

> CUDA available이 False라면 모델 평가만 진행할 수 있습니다.

### 2. CUDA 코드 빌드

학습 및 평가 시에 사용하는 코드 중 일부분이 성능상의 이유로 `.cu(.cpp)` 파일로 작성되어있습니다. 이를 파이썬 코드에서 사용할 수 있도록 빌드하는 과정이 필요합니다.

```bash
python setup.py build_ext --inplace
```

실행 후 `pointpillars/ops` 아래에 `voxel_op.~~~.so` 파일이 생성되었다면 성공입니다.

## 모델 학습하기

> 모델 학습을 새로하지 않고, 보유한 체크포인트 파일이 있다면 바로 평가하기를 진행할 수 있습니다. `checkpoints/best.ckpt`에 가장 결과가 잘 나왔던 체크포인트를 저장해두었습니다.

### 1. 학습데이터 만들기

아래처럼 dataset이 있다고 가정합니다. `.pcd` 파일과, 라벨 데이터 파일 (`.txt`)이 같은 폴더에 들어있는 환경을 가정하고 있습니다.

```
/home/rideflux/LIDAR_DATA
├── 10_binary
│   ├── 123_1.1m.pcd
│   ├── 123_1.1m.txt
│   └── ...
├── 11_binary
│   ├── 123_1.1m.pcd
│   ├── 123_1.1m.txt
│   └── ... 
├── 12_binary
│   ├── 123_1.1m.pcd
│   ├── 123_1.1m.txt
│   └── ... 
```

라벨 데이터의 구성은 다음 그림을 따릅니다.

![alt text](figures/label_format.png)


[collect_pcd_data.py](./collect_pcd_data.py)를 수정하여 데이터셋이 있는 폴더를 지정합니다. 본 스크립트는 해당 폴더 하위(중첩 폴더 포함)의 모든 `.pcd` 파일을 수집하므로, 최상단의 폴더 이름으로 수정하면 됩니다. 

위 예시 기준으로는 아래와 같이 수정하시면 됩니다.
```py
# collect_pcd_data.py
root_dir = '/home/rideflux/LIDAR_DATA'
...
```

> `~`(Home 디렉토리)는 사용할 수 없으므로, 해당 데이터 폴더의 전체 절대경로를 입력해주셔야 합니다.

수정 후 다음을 실행합니다.

```bash
python collect_pcd_data.py
```

실행 후 지정한 폴더 아래에 

- train_set.pkl
- val_set.pkl
- test1_set.pkl
- test2_set.pkl
- test3_set.pkl

와 같은 파일들이 생겼다면 성공입니다.

```
/home/rideflux/LIDAR_DATA
├── 10_binary
│   ├── 123_1.1m.pcd
│   └── ... 
├── 11_binary
│   ├── 123_1.1m.pcd
│   └── ... 
├── 12_binary
│   ├── 123_1.1m.pcd
│   └── ... 
├── train_set.pkl
├── val_set.pkl
├── test1_set.pkl
├── test2_set.pkl
└── test3_set.pkl
```

### 2. config 수정하기

다음으로, [pcd.yaml](./configs/data/datasets/pcd.yaml)을 수정합니다. 

학습데이터 만들기에서 사용한 폴더명을 그대로 사용하시면 됩니다.

```yaml
dataset_root_dir: /home/rideflux/LIDAR_DATA
```

### 3. 학습하기

아래 명령을 입력하여 학습을 시작합니다.

```bash
python train.py
```

기본적으로 epoch은 총 500번 돌게 되어있으며, 최소 20번의 epoch 실행 후 더이상 성능 개선이 이뤄지지 않는다면 멈추게 되어있습니다. (제공해주신 데이터 기준으로 약 22번 실행 후 종료되었습니다.)

> Processing train data <span style="color:#f92672;">━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</span> 100% 0:00:10  
> train dataset size : 9078  
> Processing val data <span style="color:#f92672;">━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸</span>━━━━━━━━━━━  72% 0:00:01


위 과정이 종료된 후, 

> Epoch 0/499 <span style="color:#6022DB;">━━</span>╺━━━━━━━━━━━━━━━━━━━━━ 48/568 0:00:21 • 0:03:55 2.21it/s

이런 모양이 뜨면 성공이며, 학습 완료 후 체크포인트 파일은 `outputs/yyyy-MM-dd/HH-mm-ss/checkpoints/`에 저장되어있습니다.

## 모델 평가하기

만약 새로 학습시킨 체크포인트를 평가하고 싶다면 [eval.yaml](./configs/eval.yaml)에서 아래와 같이 `outputs/yyyy-MM-dd/HH-mm-ss/checkpoints/epoch_***.ckpt`로 수정하면 됩니다.

```yaml
eval_checkpoint: /home/rideflux/roadsign-detector/outputs/2025-07-31/19-42-26/checkpoints/epoch_017.ckpt
```

수정한 뒤 `eval.py`를 실행합니다.

```py
python eval.py
```

아래와 같은 글씨가 떴다면 원하는 모드를 선택하여 실행합니다.

```
평가 모드를 선택해 주세요
1. Bird's eye view(조감도)로 시각화하기
2. 3D로 시각화하기
3. 정량 평가만 진행하기

1/2/3 중 하나 입력: 
```

### 1. Bird's eye view(조감도)로 시각화하기

`Inference Image (ctrl + click)`

평가 모드 선택 후 등장하는 위 텍스트에 마우스를 대고 컨트롤 + 클릭을 하면 아래와 같은 이미지를 볼 수 있습니다. (직접 [test_infer.png](test_infer.png)를 켜고 보셔도 무방합니다)

![alt text](figures/test_infer.png)

여기에서 빨간 사각형이 실제 박스의 위치이고, 노란 사각형이 추론을 통해 얻어낸 박스의 위치입니다. 엔터를 누를 때마다 다음 프레임으로 넘어가게 됩니다.

Ctrl+C 를 눌러 프로그램을 중단할 수 있습니다.

### 2. 3D로 시각화하기

2번을 선택하면 테스트데이터 로딩 후 자동으로 3d 화면이 뜨게 됩니다. 

편의상 일정 반사율 이상의 점만 표시하였고, 마우스 좌클릭-드래그와 휠클릭-드래그로 이동하며 보실 수 있습니다.

**q**를 눌러 화면을 닫을 수 있고, 엔터를 눌러 다음 프레임으로 넘어갈 수 있습니다.

Ctrl+C 를 눌러 프로그램을 중단할 수 있습니다.

### 3. 정량 평가만 진행하기

3번을 선택하면 사용자와의 상호작용없이 테스트셋을 모두 평가합니다.

그 결과로 다음 세 개의 그래프가 생성되게 됩니다.

#### Confusion Matrix(혼동 행렬)

![alt text](figures/confusion_matrix.png)

x축이 추론한 클래스이고, y축이 실제 클래스입니다.

각 숫자는 실제 클래스가 Class_y일 때 Class_x로 추론한 개수를 나타내며, 대각선에 밀집해 분포할 수록 정확도가 높은 모델입니다.

#### Distance Histogram

![alt text](figures/distance_histogram.png)

이 그래프는 실제 박스의 중심 위치와, 추론을 통해 얻어낸 박스의 중심위치가 얼마나 떨어져있는지를 의미합니다.

더 왼쪽으로 치우쳐 있을수록, x축의 단위가 작을수록 좋은 모델입니다.

#### Score Histogram

![alt text](figures/score_histogram.png)

본 모델은 일정 이하의 score라면 화면상에 객체가 없다고 판단합니다.

따라서, 실제로 객체가 없는 데이터가 테스트셋에 있을 때, score를 최대한 작게, 실제 객체가 있다면 최대한 크게하는 것이 유리합니다. 

Score Histogram은 이 분포를 나타내주며, 양쪽에 치우쳐 있을수록, 두 봉우리의 경계가 명확할 수록 좋은 모델입니다.

## Troubleshooting

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 454.00 MiB. GPU 0 has a total capacity of 7.66 GiB of which 269.25 MiB is free. Process 2157851 has 5.39 GiB memory in use. Including non-PyTorch memory, this process has 1.13 GiB memory in use. Of the allocated memory 955.86 MiB is allocated by PyTorch, and 14.14 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

학습 코드를 돌리는 중에 평가 코드를 돌리거나 등의 작업을 하면 GPU의 메모리가 부족하여 멈추게 됩니다. 하나를 중지시키면 정상적으로 작동합니다.


## ONNX, TensorRT 빌드

### 0. 환경 세팅

TensorRT의 버전을 반드시 8.5 (8.5.0.3)으로 맞추는 것이 제일 중요합니다. NVIDIA의 공식 도커 컨테이너를 사용하여 설정하는 것을 추천합니다.

개발 시에는 [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt/tags)의 `23.03-py3`(nvcr.io/nvidia/tensorrt:23.03-py3)을 사용하였습니다.

### 1. ONNX 생성

```bash
python export_pointpillars.py
```

### 2. TensorRT Engine 추출

```bash
trtexec --onnx=final.onnx --saveEngine=final.trt
```

### 3. trt 파일 테스트

```bash
python trt_inference.py --trt=final.trt --pcd=sample/111105_test_05234.pcd
```


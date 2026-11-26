# Roadsign Detector

LiDAR PCD에서 표지판의 박스와 클래스를 추론합니다. 현재 모델은 회귀 모델과 분류 모델을 순서대로 실행하는 2단계 구조입니다.

```text
PCD (x, y, z, intensity)
  → PointPillars: 박스와 검출 점수
  → 예측 박스로 3 × 64 × 32 crop 생성
  → Classifier: 클래스 0~7 또는 8 (No object)
```

PointPillars 논문: [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lang_PointPillars_Fast_Encoders_for_Object_Detection_From_Point_Clouds_CVPR_2019_paper.pdf)

## 최종 모델

| 모델 | 체크포인트 | 용도 |
|---|---|---|
| Regression ver219 | `checkpoints/best_regression_ver219.ckpt` | 박스 회귀와 표지판 검출 |
| Classification 261112 | `classifier_261112_checkpoints/best_model.pth` | 8개 객체 클래스와 No object 분류 |

회귀 체크포인트는 `outputs/2026-11-19/16-00-58/checkpoints/epoch_012.ckpt`의 복사본입니다.
분류 체크포인트는 `261112` 학습의 best 모델입니다. 기본 crop 생성, 회귀 평가, 통합 평가, export 설정은 위 체크포인트를 사용합니다.

이번 버전에 추가하는 모델 파일은 위 두 개입니다. 과거 버전의 체크포인트와 ONNX/TensorRT 파일을 로컬에서 생성해도 새로 Git에 포함되지 않도록 `.gitignore`에 설정했습니다. 기존 커밋에 포함된 과거 모델은 유지합니다.

## 개발 환경과 설치

개발에 사용한 주요 환경은 다음과 같습니다. 전체 패키지 목록은 [requirements.txt](requirements.txt)에 있습니다.

- Ubuntu 20.04, Python 3.8.20
- PyTorch 2.4.1, Lightning 2.4.0, Hydra 1.3.2
- spconv 2.3.6, OpenCV, Open3D, Matplotlib

```bash
pip install -r requirements.txt
python -c "import torch, hydra, lightning, spconv; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

학습에는 GPU를 사용합니다. 통합 평가와 crop 생성, 분류 평가는 GPU가 없으면 CPU로 실행할 수 있습니다. spconv 등 모델 의존 패키지는 CPU 평가에서도 필요합니다.
모든 명령은 저장소 루트에서 실행합니다.

## 주요 파일

| 파일 | 역할 |
|---|---|
| `generate_roadsign_label_from_rf_bbox_label.py` | RF 박스 라벨 변환 및 PCD 복사 |
| `collect_pcd_data.py` | PCD와 라벨을 train/val/test PKL로 수집 |
| `train_regression.py` | 회귀 모델 학습 |
| `generate_crop_box.py` | 분류 학습용 GT crop과 예측 crop 생성 |
| `roadsign_classifier.py` | 분류 모델 정의 및 학습 |
| `eval_two_stage.py` | PCD에서 두 모델을 순서대로 실행해 통합 평가 |
| `roadsign_classifier_eval.py` | 저장된 crop으로 분류 모델만 평가 |
| `eval.py` | 회귀 모델만 평가 |
| `evaluation/roadsign_crop.py` | crop 생성과 통합 평가가 공유하는 투영 함수 |
| `evaluation/plots.py` | 혼동행렬 이미지 저장 |

## 데이터 준비

### 데이터 루트

PCD와 같은 이름의 TXT 라벨이 같은 폴더에 있어야 합니다. 데이터와 생성된 PKL, crop은 Git에 포함하지 않습니다.

```text
LIDAR_DATA2/
├── data261112/
│   └── scene_name/
│       ├── 0.pcd
│       ├── 0.txt
│       └── ...
├── train_set_regression_261112.pkl
├── val_set_regression_261112.pkl
└── test1_set_regression_test1_261112.pkl
```

데이터 루트는 환경변수로 지정할 수 있습니다. `collect_pcd_data.py`와 회귀·분류용 Hydra 데이터 설정에 함께 적용됩니다.

```bash
export ROADSIGN_DATA_ROOT=/absolute/path/to/LIDAR_DATA2
```

환경변수를 지정하지 않으면 기존 개발 경로 `/media/jungirf/nia_0516/LIDAR_DATA2`를 사용합니다.
Hydra를 사용하는 명령에서는 `data.datasets.dataset_root_dir=/absolute/path` 옵션으로도 변경할 수 있습니다.

### RF 라벨 변환

RF TXT의 `class cx cy cz dx dy dz ...`를 클래스와 8개 꼭짓점 좌표로 변환합니다.
원본 PCD는 `<data-root>/<scene>/<sensor>/pcd/<번호>.pcd`에서 찾습니다.

```bash
python generate_roadsign_label_from_rf_bbox_label.py \
  --label-root /absolute/path/to/rf_labels/pandar64_0 \
  --data-root /absolute/path/to/roadsign_261112 \
  --output-root "$ROADSIGN_DATA_ROOT/data261112" \
  --scene-date 2026-11-12
```

변환된 라벨은 한 줄에 `class,x1,y1,z1,...,x8,y8,z8`의 25개 값을 저장하며, 라벨이 없으면 빈 TXT로 남습니다. 현재 데이터 처리 코드는 프레임당 첫 번째 라벨을 사용합니다.

### PKL 생성

```bash
python collect_pcd_data.py
```

| PKL | 구성 |
|---|---|
| `train_set_regression_261112.pkl` | 학습 프레임 |
| `val_set_regression_261112.pkl` | 검증 프레임 |
| `test1_set_regression_test1_261112.pkl` | val과 같은 프레임 |
| `test2_set_regression_test2_261112.pkl` | 현재 추가 샘플링이 없어 val과 같은 프레임 |
| `test3_set_regression_test3_261112.pkl` | `data261112`에 속한 검증 프레임 |

`data261112`는 장면별로 TXT를 숫자순으로 정렬한 뒤 빈 라벨도 포함해 매 5번째 프레임을 val에, 나머지를 train에 넣습니다. 빈 라벨은 `label=None`으로 저장하며 `fp_paths`의 무작위 샘플링에 넣지 않습니다.
다른 날짜 데이터의 분할은 기존 규칙을 유지합니다. `fp_paths`는 전체 후보의 50%를 무작위 선택한 뒤 train/val에 8:2로 나눕니다. 따라서 PKL을 다시 만들면 이 후보들의 구성이 달라질 수 있습니다.

분류용 데이터 처리기는 클래스 번호를 보존합니다. `data261112`의 클래스는 PKL 저장 시 +1, 로딩 시 -1하여 최종적으로 TXT의 0~7 번호를 유지합니다. 회귀용 처리기는 모든 객체를 클래스 0으로 읽습니다.

## 모델 학습

### 1. 회귀 모델

```bash
python train_regression.py
```

설정은 [configs/train_regression.yaml](configs/train_regression.yaml)과 [configs/data/roadsign_data_regression.yaml](configs/data/roadsign_data_regression.yaml)을 사용합니다.
기본 batch size는 16이며, 최대 500 epoch, 최소 20 epoch 이후 early stopping을 적용합니다.
학습 체크포인트는 `outputs/<날짜>/<시간>/checkpoints/`에 저장됩니다.

### 2. 분류용 crop 생성

```bash
python generate_crop_box.py
```

[configs/classifier.yaml](configs/classifier.yaml)의 `regression_checkpoint`로 예측 박스를 생성합니다. 새 회귀 모델로 다시 학습할 경우 이 경로를 해당 체크포인트로 지정합니다.

```bash
python generate_crop_box.py regression_checkpoint=outputs/<날짜>/<시간>/checkpoints/epoch_012.ckpt
```

객체가 있는 프레임은 GT 박스와 예측 박스로 만든 crop을 모두 저장합니다. 빈 라벨은 5개 중 1개의 예측 crop을 클래스 8로 저장합니다. 이 crop 샘플링은 PKL 분할과 별개입니다.
기존 라벨 목록이 있으면 교체 여부를 확인하며, `y`를 입력하지 않으면 기존 목록을 유지하고 종료합니다.

```text
Dataset/
├── train/classification_261112/*.npy
├── test/classification_261112/*.npy
├── train_classification_261112.txt
└── test_classification_261112.txt
```

crop은 `float32[3, 64, 32]`이며 채널은 평균 intensity/255, 평균 -y/40, 포인트 개수입니다.
예측 박스의 x 폭은 1.2배, y 범위는 중심 ±1 m를 사용합니다. z로 포인트를 자르지 않습니다.
선택한 포인트가 50개 미만 또는 50,000개 초과이면 0으로 채운 crop을 사용합니다.

### 3. 분류 모델

```bash
python roadsign_classifier.py
```

`Dataset/train_classification_261112.txt`와 `Dataset/test_classification_261112.txt`를 읽습니다.
기본 batch size는 32, 최대 200 epoch이며 검증 정확도가 가장 높은 모델을 `classifier_261112_checkpoints/best_model.pth`에 저장합니다.

## 통합 평가

```bash
python eval_two_stage.py
```

실행 시 다음 모드 중 하나를 선택합니다.

```text
1. 전방 + BEV로 시각화하기
2. 정량 평가만 진행하기
```

선택 메뉴를 생략하려면 `eval_mode`를 지정합니다.

```bash
python eval_two_stage.py eval_mode=1 testset_name=test3
python eval_two_stage.py eval_mode=2 testset_name=test1
python eval_two_stage.py eval_mode=2 testset_name=val max_samples=32
```

설정 파일은 [configs/eval_two_stage.yaml](configs/eval_two_stage.yaml)입니다.

| 설정 | 기본값 / 의미 |
|---|---|
| `testset_name` | `test1`; `val`, `test2`, `test3`, `custom`도 선택 가능 |
| `regression_checkpoint` | `checkpoints/best_regression_ver219.ckpt` |
| `classification_checkpoint` | `classifier_261112_checkpoints/best_model.pth` |
| `detection_threshold` | 0.5; 점수가 이 값 이하면 No object |
| `distance_threshold` | 0.1 m; 박스 중심 오차 허용값 |
| `device` | `auto`; `cpu`, `cuda`, `cuda:0`도 지정 가능 |
| `max_samples` | `null`; 지정하면 해당 개수까지만 평가 |
| `data.datamodule.num_workers` | 16; 단일 프로세스 로딩은 0 |

통합 평가는 PCD마다 회귀 모델의 **예측 박스**로 crop을 만들어 분류합니다. GT crop이나 빈 라벨 샘플링을 사용하지 않으며, 선택한 PKL 항목을 모두 평가합니다.
회귀 점수가 기준 이하이면 분류 모델을 실행하지 않고 최종 클래스를 8로 정합니다. 분류 모델도 클래스 8을 예측할 수 있습니다.

### 시각화

1번 모드에서는 터미널의 이미지 링크로 `test_infer.png`를 열어 전방·BEV를 확인합니다.
GT 박스는 빨강, 회귀 후보 박스는 청록으로 표시하며, 최종 클래스와 두 모델의 점수도 표시합니다.

- Enter: 다음 프레임
- 숫자 N: 다음 N개 프레임의 시각화만 생략하고 평가 계속
- `c` 또는 `q`: 현재 프레임까지의 결과를 저장하고 종료

기본 이미지 경로는 실행마다 생성되는 결과 폴더 안에 있습니다. 같은 VS Code 탭을 계속 사용하려면 고정 경로를 지정합니다.

```bash
python eval_two_stage.py eval_mode=1 inference_img_path=test_infer.png
```

### 저장되는 결과와 지표

두 모드 모두 `evaluation_results/two_stage/<날짜_시간>/`에 다음 파일을 저장합니다.

| 파일 | 내용 |
|---|---|
| `metrics.json` | 검출·위치·분류·통합 지표, 클래스별 지표와 실행 설정 |
| `predictions.csv` | PCD별 GT/예측 클래스, 박스, 점수와 중심 오차 |
| `confusion_matrix.png` | 9클래스 혼동행렬; 행=GT, 열=예측 |
| `test_infer.png` | 시각화 모드에서 마지막으로 확인한 프레임 |

회귀 검출 precision/recall은 범위 안 GT 객체의 존재 여부와 점수 기준으로 계산하며, 위치 정확도는 별도로 집계합니다.
중심 오차는 `(cx, cy)` 사이의 2차원 거리입니다. 박스 중심에 반크기를 더하는 기존 `evaluation.utils.compute_distance`와 계산 방식이 다릅니다.
GT 중심이 설정된 x/y 평가 범위 밖에 있으면 No object로 취급합니다.
통합 정답은 객체의 검출·클래스·위치를 모두 맞힌 경우, 또는 No object를 최종적으로 올바르게 예측한 경우입니다.
Macro F1은 GT 또는 예측에 등장한 클래스의 평균이며, 전체 9클래스 평균도 `macro_f1_all_classes`로 저장합니다.
시각화를 중간에 종료한 경우 지표는 해당 시점까지 처리한 프레임에 대한 값입니다.

## 개별 모델 평가

회귀 모델만 평가하려면 다음을 실행합니다. 데이터 선택과 체크포인트는 [configs/eval_regression.yaml](configs/eval_regression.yaml)에서 변경할 수 있습니다.

```bash
python eval.py eval_mode=5 testset_name=test1
```

저장된 crop으로 분류 모델만 평가하려면 다음을 실행합니다.

```bash
python roadsign_classifier_eval.py
python roadsign_classifier_eval.py --device cpu --output evaluation_results/classifier/confusion_matrix.png
```

`--checkpoint`, `--labels`, `--batch-size`도 지정할 수 있습니다.
이 평가는 라벨 목록에 저장된 GT crop과 예측 crop을 사용하므로, PCD 전체를 처리하는 통합 평가와 평가 대상이 다릅니다.

## ONNX / TensorRT

회귀 모델과 분류 모델은 각각 export합니다.

```bash
python export_pointpillars.py
python roadsign_classifier_onnx.py
```

회귀 모델은 [configs/export_regression.yaml](configs/export_regression.yaml)의 최종 체크포인트를 읽어 `final_regression_ver219.onnx`를 생성합니다. GPU와 ONNX graph plugin 구성에 필요한 패키지가 필요합니다.
분류 모델은 `roadsign_classifier_ver219.onnx`를 생성하며, `--checkpoint`와 `--output`으로 경로를 변경할 수 있습니다.
TensorRT plugin을 포함한 배포 환경에서 회귀 ONNX를 빌드해야 합니다. 기존 TensorRT 8.5 개발 환경과 plugin 코드는 `tensorrt/`를 참고합니다.
생성한 ONNX, TensorRT engine은 이번 체크포인트 업로드 대상에 포함하지 않습니다.

## 테스트

```bash
python -m unittest discover -s tests -v
```

테스트는 GPU나 실제 데이터셋 없이 crop 투영, 분할 규칙, 최종 클래스 전달, 점수 기준 처리, 마지막 배치, 시각화 종료와 결과 파일 생성을 확인합니다.
실제 데이터로 짧게 확인하려면 다음을 실행합니다.

```bash
python eval_two_stage.py eval_mode=2 testset_name=test3 max_samples=16 device=cpu data.datamodule.num_workers=0
```

기존 단일 모델 학습(`train.py`)과 과거 실험용 추론 스크립트는 유지하며, 현재 2단계 모델의 기본 진입점은 위 표의 파일을 사용합니다.

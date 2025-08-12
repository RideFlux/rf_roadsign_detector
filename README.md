# PER-roadsign_detector

## Environments

- Ubuntu 20.04.6 LTS (64-bit)
- 12th Gen Intel® Core™ i7-12700 × 20
- NVIDIA GeForce RTX 3070/PCIe/SSE2
- Python 3.8.20
- CUDA 12.8

## How to run

### 1. Install dependencies

> Virtual environment recommended (e.g. conda)

```bash
pip install -r requirements.txt
```

### 2. Setup

```bash
python setup.py build_ext --inplace
```

### 3. Create train data

Modify [collect_pcd_data.py](./collect_pcd_data.py) to your `.pcd` files' path and run it. This script collects pcd files **recursively**, so you can just enter the root directory of all pcd files.

Also you can modify the ratio of train and testset. (Default: 0.2)


```bash
python collect_pcd_data.py
```

#### Example

```py
# collect_pcd_data.py
root_dir = '/home/o-bard-o/LIDAR_DATA'
...
```

### 4. Modify config files

Modify [pcd.yaml](./configs/data/datasets/pcd.yaml)

The recommended structure's like below.

```
/home/o-bard-o/LIDAR_DATA
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

`~`(Home directory) doesn't work. You should enter the full absolute path.

```yaml
dataset_root_dir: /home/o-bard-o/LIDAR_DATA
```

### 5. Train

```py
python train.py
```

### 6. Evaluate

If you had trained the model recently, you should change the checkpoint file's path in [eval.yaml](./configs/eval.yaml)

```yaml
eval_checkpoint: /home/o-bard-o/PER-roadsign_detector/outputs/2025-07-31/19-42-26/checkpoints/epoch_017.ckpt
```

Then, run the `eval.py`

```py
python eval.py
```

## How to deploy

### 0. Environment Setup

The most important thing is `TensorRT 8.5.0.3` and using official nvidia container is recommended.

I used `23.03-py3`(nvcr.io/nvidia/tensorrt:23.03-py3) from [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt/tags).

- NVIDIA-SMI 570.133.07 (Should be installed at host device)
- CUDA 12.1
- cuDNN 8.8
- TensorRT 8.5.0.3
- Python 3.8.10

### 1. Create onnx file

```bash
python export_pointpillars.py
```

### 2. Build TensorRT Engine from onnx

```bash
trtexec --onnx=final.onnx --saveEngine=final.trt
```

### 3. Test inference

```bash
python trt_inference.py --trt=final.trt --pcd=sample/111105_test_05234.pcd
```


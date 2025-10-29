# 🛰️ PointCloud Matcher

> `rf_roadsign_detector`로부터 얻은 측위 보정시설물의 PointCloud와 사전 구축된 맵 데이터를 정합(matching)하는 프로젝트입니다.



## ✨ 주요 기능 (Features)

* **PCD 파일 정합**: 소스(Source)와 타겟(Target) PointCloud를 정밀하게 정합합니다.
* **시각화**: 정합 과정과 최종 결과를 PCL Viewer를 통해 시각적으로 확인할 수 있습니다.
* **결과 저장**: 정합이 완료된 PointCloud를 새로운 `.pcd` 파일로 저장합니다.

---

## 🚀 시작하기 (Getting Started)

### 1. 빌드 (Build)

```bash
# git clone https://your-repository-url/pc_matcher.git
cd pc_matcher
mkdir build
cd build
cmake ..
make
```

### 2. 실행 (Run)

빌드가 완료된 후, `build` 폴더 내에서 아래 명령어로 프로그램을 실행합니다.

```bash
./main
```

---

## ⚙️ 파라미터 설정 (Configuration)

모든 파라미터는 `configs/config.toml` 파일에서 수정할 수 있습니다.

### `[Paths]`

파일 및 디렉토리 경로를 설정합니다. 모든 경로는 `.pcd` 확장자를 포함해야 합니다.

* **`target_cloud_path`**: (입력) 정합의 기준이 될 측위 보정시설물 맵 파일의 절대 경로
    * **`pcd/roadsign_map.pcd`** 에 맵 파일이 저장되어 있습니다.
* **`source_cloud_path`**: (입력) 정합을 시도할 측위 보정시설물 인지 결과 파일의 절대 경로
* **`result_pcd_path`**: (출력) 정합된 결과를 저장할 파일의 경로

```toml
[Paths]
target_cloud_path = "path/to/your/map.pcd" 
source_cloud_path = "path/to/your/source.pcd"
result_pcd_path = "results/matched_cloud.pcd"
```

### `[Settings]`

프로그램의 동작을 제어하는 옵션입니다 (`true` 또는 `false`).

* **`visualize_process`**: 정합이 진행되는 중간 과정을 시각화할지 여부
* **`visualize_result`**: 최종 정합 결과를 시각화할지 여부
* **`save_result_pcd`**: `result_pcd_path`에 지정된 경로로 정합 결과를 저장할지 여부

```toml
[Settings]
visualize_process = true
visualize_result = true
save_result_pcd = true
```
---

## 📦 필수 요구사항 (Prerequisites)

빌드에 실패한다면, 아래 라이브러리가 설치되어 있는지 확인해주세요.

* **C++17** 이상을 지원하는 컴파일러 (g++)
    ```bash
    sudo apt-get update && sudo apt-get install -y build-essential g++
    ```
* **PCL (Point Cloud Library)**
    ```bash
    sudo apt-get install -y libpcl-dev
    ```
* **OpenCV**
    ```bash
    sudo apt-get install -y libopencv-dev
    ```



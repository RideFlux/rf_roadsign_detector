import os
import random
import pickle
import numpy as np
from collections import defaultdict

# 설정
root_dir = os.environ.get('ROADSIGN_DATA_ROOT', '/media/jungirf/nia_0516/LIDAR_DATA2')

# 결과 파일
## 뒤에 261112는 데이터 셋 최종 수집 일자
train_file  = 'train_set_regression_261112.pkl'
val_file    = 'val_set_regression_261112.pkl'
test1_file  = 'test1_set_regression_test1_261112.pkl'
test2_file  = 'test2_set_regression_test2_261112.pkl'
test3_file  = 'test3_set_regression_test3_261112.pkl'

def parse_label_file(txt_path):
    """txt 파일에서 라벨을 파싱하여 클래스와 bounding box 좌표를 반환"""
    if not os.path.exists(txt_path):
        return None

    with open(txt_path, 'r') as f:
        line = f.readline().strip()
        if not line:
            return None

        values = list(map(float, line.split(',')))
        if len(values) != 25:  # 클래스(1) + 좌표(24)
            return None

        class_id = int(values[0])
        coordinates = np.array(values[1:]).reshape(8, 3)  # [8][3] 배열

        if 'label_1002' in txt_path or 'labels_0604' in txt_path or 'data260731' in txt_path or 'data260820' in txt_path or 'data260918' in txt_path or 'data261008' in txt_path or 'data261112' in txt_path:
            class_id  += 1

        return {
            'class': class_id,
            'bbox': coordinates
        }

def split_dataset_by_class(root_dir):
    """
    Args:
        root_dir: pcd 파일들이 있는 최상위 디렉토리
    Returns:
        d_train: train용 pcd 경로 리스트
        d_val: val용 pcd 경로 리스트
        class_dict: 클래스별 전체 경로 딕셔너리 (optional)
    """
    fp_paths = []
    d_train, d_val = [], []
    total_class_dict = defaultdict(list)
    # 1. sh 붙지 않은 모든 pcd 파일 탐색
    for dirpath, _, filenames in os.walk(root_dir):
        if 'sh_' in dirpath or 'data260731' in dirpath or 'data260820' in dirpath or 'data260918' in dirpath or 'data261008' in dirpath or 'data261112' in dirpath:
            continue
        class_dict = defaultdict(list)
        for fn in filenames:
            if fn.endswith(".pcd"):
                pcd_path = os.path.join(dirpath, fn)

                # txt 파일 경로로 변환
                no_ext_path = os.path.splitext(pcd_path)[0]  # .pcd 제거
                txt_path = no_ext_path + ".txt"
                relative_path = os.path.relpath(no_ext_path, root_dir)

                if(relative_path.startswith('fp')):
                    fp_paths.append(relative_path)
                else:
                    if os.path.exists(txt_path):
                        label = parse_label_file(txt_path)
                        if label is None:
                            cls = "none"
                        else:
                            cls = label['class']
                    else:
                        # continue
                        cls = "none"  # txt 없는 경우 별도 클래스

                    class_dict[cls].append(relative_path)
                    total_class_dict[cls].append(relative_path)

    # 2. 클래스별 8:2 split
    ## 폴더마다 class 분리하도록 설정
        for cls, paths in class_dict.items():
            paths.sort()  # 순서대로 split
            if cls == 'none':
                paths = paths[::10]

            num_val = int(len(paths) * 0.2)
            if num_val > 0:
                # print('class :', cls, 'data lengths :', len(paths[-num_val:]), len(paths[:-num_val]))
                d_val.extend(paths[-num_val:])
                d_train.extend(paths[:-num_val])
            else:
                d_train.extend(paths)  # 샘플 적으면 모두 train

    # sh 붙은 파일들에 대한 data split
    for dirpath, _, filenames in os.walk(root_dir):
        if 'sh_' not in dirpath:
            continue
        for fn in filenames:
            if not fn.endswith(".pcd"):
                continue
            pcd_path = os.path.join(dirpath, fn)
            no_ext_path = os.path.splitext(pcd_path)[0]  # .pcd 제거
            relative_path = os.path.relpath(no_ext_path, root_dir)

            if 'test' in dirpath:
                d_val.append(relative_path)
            else:
                d_train.append(relative_path)

    # 260731 data and 260820 split
    dir_idx = 0
    for dirpath, _, filenames in os.walk(root_dir):
        if 'data260731' not in dirpath and 'data260820' not in dirpath and 'data260918' not in dirpath:
            continue
        is_val = (dir_idx%5 == 0)
        dir_idx += 1
        for fn in filenames:
            if not fn.endswith('.txt'):
                continue
            txt_path = os.path.join(dirpath, fn)
            no_ext_path = os.path.splitext(txt_path)[0]
            relative_path = os.path.relpath(no_ext_path, root_dir)

            if os.path.getsize(txt_path) == 0:
                fp_paths.append(relative_path)
                continue

            if is_val:
                d_val.append(relative_path)
            else:
                d_train.append(relative_path)

    for dirpath, _, filenames in os.walk(root_dir):
        if 'data261008' not in dirpath and 'data261112' not in dirpath:
            continue

        file_idx = 0

        txt_files = (fn for fn in filenames if fn.endswith('.txt'))
        for fn in sorted(txt_files, key=lambda x: int(x.split('.')[0])):
            if not fn.endswith('.txt'):
                continue
            txt_path = os.path.join(dirpath, fn)
            no_ext_path = os.path.splitext(txt_path)[0]
            relative_path = os.path.relpath(no_ext_path, root_dir)

            # data261112는 빈 라벨도 포함해 전체 프레임을 5개 주기로 분할한다.
            if 'data261112' not in dirpath and os.path.getsize(txt_path) == 0:
                fp_paths.append(relative_path)
                continue

            file_idx += 1
            if file_idx % 5 == 0:
                d_val.append(relative_path)
            else:
                d_train.append(relative_path)


    return d_train, d_val, fp_paths, total_class_dict

# 데이터셋 구성
def create_dataset(d_paths, f_paths, d_sample_paths, start_index=0):
    dataset = []
    index = start_index

    # D 데이터 처리 (원본, reversed=False)
    for path in d_paths:
        pcd_path = os.path.join(root_dir, path + '.pcd')
        txt_path = os.path.join(root_dir, path + '.txt')
        label = parse_label_file(txt_path)

        dataset.append({
            'index': index,
            'pcd_path': pcd_path,
            'label': label,
            'reversed': False
        })
        index += 1

    # F 데이터 처리 (라벨 없음, reversed=False)
    for path in f_paths:
        pcd_path = os.path.join(root_dir, path + '.pcd')

        dataset.append({
            'index': index,
            'pcd_path': pcd_path,
            'label': None,
            'reversed': False
        })
        index += 1

    # D 데이터 샘플링 (라벨 없음, reversed=True)
    for path in d_sample_paths:
        pcd_path = os.path.join(root_dir, path + '.pcd')

        dataset.append({
            'index': index,
            'pcd_path': pcd_path,
            'label': None,
            'reversed': True
        })
        index += 1

    return dataset, index

def main():
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(root_dir)
    d_train, d_val, fp_paths, class_dict = split_dataset_by_class(root_dir)
    for k, v in class_dict.items():
        print(k, ":", len(v))

    fp_sample = random.sample(fp_paths, int(len(fp_paths) * 0.5))
    num_f_val = int(len(fp_sample) * 0.2)
    f_train = fp_sample[num_f_val:]
    f_val = fp_sample[:num_f_val]
    d_train_sample, d_val_sample = [], []
    d_val_sample_20 = []


    # Train: 객체 프레임 + 샘플링한 negative의 80%
    train_dataset, next_index = create_dataset(d_train, f_train, d_train_sample, 0)

    # Val: 객체 프레임 + 샘플링한 negative의 20%
    val_dataset, test_start_index = create_dataset(d_val, f_val, d_val_sample, next_index)

    # Test1: val과 동일
    test1_dataset, test2_start_index = create_dataset(d_val, f_val, d_val_sample, test_start_index)

    # Test2: 현재 추가 샘플링이 없으므로 val과 같은 프레임을 사용
    test2_dataset, test3_start_index = create_dataset(d_val, f_val, d_val_sample_20, test2_start_index)

    # Test3: 기존 Test3 구성 중 경로에 data261112가 포함된 데이터만 사용
    test3_d_paths = [path for path in d_val if 'data261112' in path]
    test3_f_paths = [path for path in fp_paths if 'data261112' in path]
    test3_sample_paths = [path for path in d_val_sample_20 if 'data261112' in path]
    test3_dataset, _ = create_dataset(
        test3_d_paths, test3_f_paths, test3_sample_paths, test3_start_index
    )

    # pickle 파일로 저장
    with open(os.path.join(root_dir, train_file), 'wb') as f:
        pickle.dump(train_dataset, f)

    print(f"Train set: D 데이터 {len(d_train)}개 + F 데이터 {len(f_train)}개 + D 샘플링 {len(d_train_sample)}개 = 총 {len(train_dataset)}개")

    with open(os.path.join(root_dir, val_file), 'wb') as f:
        pickle.dump(val_dataset, f)

    print(f"Val set: D 데이터 {len(d_val)}개 + F 데이터 {len(f_val)}개 + D 샘플링 {len(d_val_sample)}개 = 총 {len(val_dataset)}개")

    with open(os.path.join(root_dir, test1_file), 'wb') as f:
        pickle.dump(test1_dataset, f)

    print(f"Test1 set: D 데이터 {len(d_val)}개 + F 데이터 {len(f_val)}개 + D 샘플링 {len(d_val_sample)}개 = 총 {len(test1_dataset)}개")

    with open(os.path.join(root_dir, test2_file), 'wb') as f:
        pickle.dump(test2_dataset, f)

    print(f"Test2 set: D 데이터 {len(d_val)}개 + F 데이터 {len(f_val)}개 + D 샘플링 {len(d_val_sample_20)}개 = 총 {len(test2_dataset)}개")

    with open(os.path.join(root_dir, test3_file), 'wb') as f:
        pickle.dump(test3_dataset, f)

    print(f"Test3 set (data261112): D 데이터 {len(test3_d_paths)}개 + F 데이터 {len(test3_f_paths)}개 + D 샘플링 {len(test3_sample_paths)}개 = 총 {len(test3_dataset)}개")

    print(f"모든 데이터셋 파일이 저장되었습니다.")


if __name__ == '__main__':
    main()

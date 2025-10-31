import os
import random
import pickle
import numpy as np
from collections import defaultdict

# 설정
# root_dir = '/media/jungirf/NIA1/roadsign_data_pcd/pointpillar_data/all_roadsign_with_label'
root_dir = '/media/jungirf/nia_0516/LIDAR_DATA'

# 결과 파일
train_file = 'train_set_straight_ver1.pkl'
val_file = 'val_set_straight_ver1.pkl'
test1_file = 'test1_set_straight_ver1.pkl'
test2_file = 'test2_set_straight_ver1.pkl'
test3_file = 'test3_set_straight_ver1.pkl'

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
        
        if 'label_1002' in txt_path:
            class_id  += 1

        return {
            'class': class_id,
            'bbox': coordinates
        }

# ############################################################################
# # .pcd 경로 수집
# all_paths = []
# fp_paths = []
# for dirpath, dirnames, filenames in os.walk(root_dir):
#     for filename in filenames:
#         if filename.endswith('.pcd'):
#             full_path = os.path.join(dirpath, filename)
#             no_ext_path = os.path.splitext(full_path)[0]  # .pcd 제거
#             # 앞부분 경로 제거
#             relative_path = os.path.relpath(no_ext_path, root_dir)
#             if(relative_path.startswith('fp')):
#                 fp_paths.append(relative_path)
#             else:
#                 all_paths.append(relative_path)

# # 셔플 및 분할
# random.shuffle(all_paths)
# random.shuffle(fp_paths)

# # D 데이터 분할 (80% train, 20% val)
# num_d_val = int(len(all_paths) * 0.2)
# d_train = all_paths[num_d_val:]
# d_val = all_paths[:num_d_val]

# # F 데이터 10% 샘플링 후 8:2 분할
# fp_sample = random.sample(fp_paths, int(len(fp_paths) * 0.1))
# num_f_val = int(len(fp_sample) * 0.2)
# f_train = fp_sample[num_f_val:]
# f_val = fp_sample[:num_f_val]

# # 10% 샘플링을 위한 추가 데이터
# d_train_sample = random.sample(d_train, int(len(d_train) * 0.1))
# d_val_sample = random.sample(d_val, int(len(d_val) * 0.1))

# # 20% 샘플링을 위한 추가 데이터 (test2용)
# d_val_sample_20 = random.sample(d_val, int(len(d_val) * 0.2))
# ########################################################################
# # .pcd 경로 수집
# all_paths = []
# fp_paths = []

# for dirpath, dirnames, filenames in os.walk(root_dir):
#     filenames = sorted(filenames)  # 파일 이름 정렬 (순서를 일정하게)
#     for filename in filenames:
#         if filename.endswith('.pcd'):
#             full_path = os.path.join(dirpath, filename)
#             no_ext_path = os.path.splitext(full_path)[0]  # .pcd 제거
#             relative_path = os.path.relpath(no_ext_path, root_dir)
#             if relative_path.startswith('fp'):
#                 fp_paths.append(relative_path)
#             else:
#                 all_paths.append(relative_path)

# # 폴더 단위로 분할
# d_train, d_val = [], []
# folder_dict = {}

# for path in all_paths:
#     folder = os.path.dirname(path)
#     folder_dict.setdefault(folder, []).append(path)

# for folder, paths in folder_dict.items():
#     paths.sort()  # 파일 이름 순서 기준
#     num_val = int(len(paths) * 0.2)
#     val_paths = paths[-num_val:] if num_val > 0 else []
#     train_paths = paths[:-num_val] if num_val > 0 else paths
#     d_train.extend(train_paths)
#     d_val.extend(val_paths)

# # F 데이터 10% 샘플링 후 8:2 분할
# random.shuffle(fp_paths)
# fp_sample = random.sample(fp_paths, int(len(fp_paths) * 0.1))
# num_f_val = int(len(fp_sample) * 0.2)
# f_train = fp_sample[num_f_val:]
# f_val = fp_sample[:num_f_val]

# # 10% 샘플링을 위한 추가 데이터
# d_train_sample = random.sample(d_train, int(len(d_train) * 0.1))
# d_val_sample = random.sample(d_val, int(len(d_val) * 0.1))

# # 20% 샘플링을 위한 추가 데이터 (test2용)
# d_val_sample_20 = random.sample(d_val, int(len(d_val) * 0.2))
# ##############################################################################

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
    # 1. 모든 pcd 파일 탐색
    for dirpath, _, filenames in os.walk(root_dir):
        # print(dirpath)
        # print(filenames)

        # a = input()
        # if a == 'c':
        #     exit()
        # else:
        #     continue
        # if 'label_1002' not in dirpath:
        #     continue
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

d_train, d_val, fp_paths, class_dict = split_dataset_by_class(root_dir)
for k, v in class_dict.items():
    print(k, ":", len(v))

fp_sample = random.sample(fp_paths, int(len(fp_paths) * 0.05))
num_f_val = int(len(fp_sample) * 0.2)
f_train = fp_sample[num_f_val:]
f_val = fp_sample[:num_f_val]
# f_train, f_val = [], []
d_train_sample, d_val_sample = [], []
d_val_sample_20 = []

# exit()

# 트레인셋 생성 (D 80% + F 80% (10% 샘플링의 80%) + D 10% 샘플링)
train_dataset, next_index = create_dataset(d_train, f_train, d_train_sample, 0)

# 검증셋 생성 (D 20% + F 20% (10% 샘플링의 20%) + D 10% 샘플링)
val_dataset, test_start_index = create_dataset(d_val, f_val, d_val_sample, next_index)

# Test1: val과 동일
test1_dataset, test2_start_index = create_dataset(d_val, f_val, d_val_sample, test_start_index)

# Test2: D 20% + F 20% (10% 샘플링의 20%) + D 20% 샘플링
test2_dataset, test3_start_index = create_dataset(d_val, f_val, d_val_sample_20, test2_start_index)

# Test3: D 20% + 모든 F 데이터 + D 20% 샘플링
test3_dataset, _ = create_dataset(d_val, fp_paths, d_val_sample_20, test3_start_index)

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

print(f"Test3 set: D 데이터 {len(d_val)}개 + F 데이터 {len(fp_paths)}개 + D 샘플링 {len(d_val_sample_20)}개 = 총 {len(test3_dataset)}개")

print(f"모든 데이터셋 파일이 저장되었습니다.")

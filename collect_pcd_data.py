import os
import random
import pickle
import numpy as np

# 설정
root_dir = '/home/yongwooklee/LIDAR_DATA'

# 결과 파일
train_file = 'train_set.pkl'
val_file = 'val_set.pkl'
test1_file = 'test1_set.pkl'
test2_file = 'test2_set.pkl'
test3_file = 'test3_set.pkl'

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
        
        return {
            'class': class_id,
            'bbox': coordinates
        }

# .pcd 경로 수집
all_paths = []
fp_paths = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith('.pcd'):
            full_path = os.path.join(dirpath, filename)
            no_ext_path = os.path.splitext(full_path)[0]  # .pcd 제거
            # 앞부분 경로 제거
            relative_path = os.path.relpath(no_ext_path, root_dir)
            if(relative_path.startswith('fp')):
                fp_paths.append(relative_path)
            else:
                all_paths.append(relative_path)

# 셔플 및 분할
random.shuffle(all_paths)
random.shuffle(fp_paths)

# D 데이터 분할 (80% train, 20% val)
num_d_val = int(len(all_paths) * 0.2)
d_train = all_paths[num_d_val:]
d_val = all_paths[:num_d_val]

# F 데이터 10% 샘플링 후 8:2 분할
fp_sample = random.sample(fp_paths, int(len(fp_paths) * 0.1))
num_f_val = int(len(fp_sample) * 0.2)
f_train = fp_sample[num_f_val:]
f_val = fp_sample[:num_f_val]

# 10% 샘플링을 위한 추가 데이터
d_train_sample = random.sample(d_train, int(len(d_train) * 0.1))
d_val_sample = random.sample(d_val, int(len(d_val) * 0.1))

# 20% 샘플링을 위한 추가 데이터 (test2용)
d_val_sample_20 = random.sample(d_val, int(len(d_val) * 0.2))

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

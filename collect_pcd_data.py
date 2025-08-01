import os
import random

# 설정
root_dir = '/home/yongwooklee/LIDAR_DATA'
test_ratio = 0.2

# 결과 파일
train_file = '/home/yongwooklee/LIDAR_DATA/train_set.txt'
test_file = '/home/yongwooklee/LIDAR_DATA/test_set.txt'

# .pcd 경로 수집
all_paths = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith('.pcd'):
            full_path = os.path.join(dirpath, filename)
            no_ext_path = os.path.splitext(full_path)[0]  # .pcd 제거
            # 앞부분 경로 제거
            relative_path = os.path.relpath(no_ext_path, root_dir)
            all_paths.append(relative_path)

# 셔플 및 분할
random.shuffle(all_paths)
num_test = int(len(all_paths) * test_ratio)
test_paths = all_paths[:num_test]
train_paths = all_paths[num_test:]

# 저장
with open(train_file, 'w') as f_train:
    for path in train_paths:
        f_train.write(path + '\n')

with open(test_file, 'w') as f_test:
    for path in test_paths:
        f_test.write(path + '\n')

print(f"총 {len(all_paths)}개 중 {len(test_paths)}개를 test_set.txt에 저장하고, {len(train_paths)}개를 train_set.txt에 저장했습니다.")

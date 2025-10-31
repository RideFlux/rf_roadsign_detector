import os
import random
import pickle
import numpy as np

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

def create_test_data(dir_list, pkl_path):
    dataset = []
    idx = 0
    
    for dir_path in dir_list:
        # file_list = sorted(os.listdir(dir_path), key=lambda x : int(x.split('.')[0]))       ## 일당은 0.pcd 이런 형식이라고 가정 (decoding 하자마자 나오는 형태임)
        file_list = sorted([x for x in os.listdir(dir_path) if x.endswith('.pcd')])
        # print(file_list)
        # exit()

        for file_name in file_list:
            file_path = os.path.join(dir_path, file_name)
            txt_path = file_path[:-4] + '.txt'
            label = None
            if os.path.exists(txt_path):
                label = parse_label_file(txt_path)
            
            dataset.append({
                'index': idx,      ## 이거 필요 없는 것 같음?
                'pcd_path': file_path,
                'label': label,
                'reversed': False,
            })

            idx += 1

    with open(pkl_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    return dataset, idx

if __name__ == '__main__':
    dir_list = [
        # '/media/jungirf/NIA_jeju9_2TB/replay/roadsign_data/test_day2/success_case/2024-11-12-17-25-03_torres_v7_101/pandar64_0/pcd',

    ]
    result_dir = '/media/jungirf/NIA1/roadsign_data_pcd/pointpillar_data/Result'
    class_dir_list = sorted(os.listdir(result_dir))
    for class_dir_name in class_dir_list:
        class_dir_path = os.path.join(result_dir, class_dir_name)
        scene_list = sorted(os.listdir(class_dir_path))
        for scene_name in scene_list:
            scene_dir_path = os.path.join(class_dir_path, scene_name)
            dir_list.append(scene_dir_path)


    pkl_path = '/media/jungirf/NIA1/roadsign_data_pcd/pointpillar_data/all_roadsign_with_label/custom_testset.pkl'
    create_test_data(dir_list, pkl_path)
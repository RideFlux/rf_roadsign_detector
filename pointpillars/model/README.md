# PointPillars

## 1. PillarLayer

### 1.1. 개요

`PillarLayer`는 Point cloud data를 BEV 상에서 일정 간격으로 나누어 각각을 기둥(pillar)들로 만들고, 반환하는 역할을 합니다. 

본 레이어는 onnx 변환시 TensorRT의 내장 플러그인 `VoxelGeneratorPlugin`으로 대체됩니다.

```py
from pointpillars.model.voxel_generator import VoxelGeneratorWrapper

class PillarLayer(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super().__init__()
        
        self.voxel_layer = VoxelGeneratorWrapper(
            vsize_xyz=voxel_size,
            coors_range_xyz=point_cloud_range,
            max_num_points_per_voxel=max_num_points,
            max_num_voxels=max_voxels,
        )
```

### 1.2. 입력

입력은 `batched_pts`로, `list[tensor]` 형태의 리스트입니다. list의 길이는 배치의 크기이며, 텐서의 모양은 다음과 같습니다.

> $(N_i, 4)$ ($N_i$: $i$번째 프레임 점의 개수, 4: 점의 x, y, z, r 값)

```py
    @torch.no_grad()
    def forward(self, batched_pts):
        pillars, coors, npoints_per_pillar = [], [], []
        for i, pts in enumerate(batched_pts):
            pts = pts.T
            voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(pts) 
            
            pillars.append(voxels_out)
            coors.append(coors_out.long())
            npoints_per_pillar.append(num_points_per_voxel_out)
        
        pillars = torch.cat(pillars, dim=0) # (p1 + p2 + ... + pb, num_points, c)
        npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0) # (p1 + p2 + ... + pb, )
        coors_batch = []
        for i, cur_coors in enumerate(coors):
            coors_batch.append(F.pad(cur_coors, (1, 0), value=i))
        coors_batch = torch.cat(coors_batch, dim=0) # (p1 + p2 + ... + pb, 1 + 3)
        
        batch_dict = {}
        
        batch_dict['voxels'] = pillars
        batch_dict['voxel_coords'] = coors_batch
        batch_dict['voxel_num_points'] = npoints_per_pillar
        
        return batch_dict
```

### 1.3. 출력

출력은 `voxels`, `voxel_coords`, `voxel_num_points` 세 개의 텐서로, 각각 다음을 의미합니다.

1. `voxels`  
  차원: $(\sum_{i=1}^{bs} P_i, n_{P_i}, C)$ ($P_i$: 프레임 당 Pillar의 개수, $n_{P_i}$: Pillar 별 점의 개수, $C$: 각 점 별 채널 수)  
  각 프레임마다 생성된 pillar에 대한 정보를 갖고 있는 실수 텐서입니다. 

2. `voxel_coords`  
  차원: $(\sum_{i=1}^{bs} P_i, 4)$ 
  각 Pillar들의 BEV상 좌표로, batch index, x좌표, y좌표, z좌표를 갖고 있는 정수 텐서입니다.
  본 모델의 설정에서는 z좌표를 고려하지 않는 pillar를 사용하기 때문에 z좌표는 항상 0입니다.

3. `voxel_num_points`
  차원: $(\sum_{i=1}^{bs} P_i)$
  각 Pillar마다 점이 몇개씩 들어있는지를 저장하는 정수 텐서입니다.
  최대 32개까지만 저장됩니다.

## 2. PillarVFE

### 2.1. 개요

PillarLayer를 통해 얻은 `voxels`, `voxel_coords`, `voxel_num_points`를 입력받아 특징값을 추출하는 레이어입니다.

> [Reference: OpenMMLab - OpenPCDet](https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/models/backbones_3d/vfe/pillar_vfe.py)

### 2.2. Feature Extraction

본 레이어는 배포 시 모델 그래프에는 포함되지 않고, `VoxelGeneratorPlugin`으로 대체됩니다.

가장 먼저 `voxels`를 가공해서 10차원의 특징값을 추출합니다.  
각각은 원래 point에 있는 `[x,y,z,r]` 값과, Pillar의 중심좌표에서 점들의 평균좌표가 얼마나 떨어져있는지를 나타내는 `f_center`(3차원), 각 점들이 점들의 평균좌표로부터 얼마나 떨어져있는지를 나타내는 `f_cluster`(3차원)로 이루어져 있습니다.

### 2.3. PFNLayer

위 단계에서 얻은 10차원 데이터 `[x,y,z,r, ...f_cluster, ...f_center]`를 이용하여 Pillar별로 64개의 특징값을 추출합니다.

보통 2개의 MLP로 이루어져 있으며, 이 부분을 수정할 경우 export_pointpillars.py를 수정해야 합니다. 

1. 첫번째 MLP 이후 voxel안에 있는 point에 대해 maxpooling + tile해서 32채널, MLP output 32 channel 이렇게 두개를 concat해서 64채널을 만듭니다.
2. 두번째 MLP에서는 바로 64채널을 내보냅니다.

이 feature는 `pillar_features`라는 텐서로 다음 단계에 전달됩니다.

## 3. PillarScatter

### 3.1. 개요

위에서 얻은 Feature를 다시 원래 물리적 좌표 위치에 배치합니다. 
따라서 `voxel_coord`와 `pillar_features` 텐서를 입력으로 받습니다.

이 또한 배포시에는 모델 그래프에 포함되지 않고, `PillarScatterPlugin`으로 대체됩니다. 

그 이유는 아래와 같습니다.

> 1. onnx 단에서 variable인 `유효한 voxel수`를 나타내는 `num_pillar`를 처리할 수 없음. 그렇다면 MAX 사이즈에 대해서 다 scatter해 주어야 하는데,  
>그렇게 되면 `(0,0,0,0)` coordinate값이 오염됨. 이게 큰 문제는 아닌 케이스가 많을것 같지만 어찌됐든 비효율이긴 함 또한 MAX 사이즈에 대해 전체 진행하면 약간의 비효율 발생
> 2. pytorch에서 만들어 낸 onnx를 그대로 사용하면 node가 다소 지저분하기도 하고 이 부분 plugin으로 쓰지 않을때 대비 onnx 전체 용량이 3배이상임 (20mb -> 68mb).  
> 용량이 절대적으로 중요한 것은 아니지만 문제의 소지가 있음

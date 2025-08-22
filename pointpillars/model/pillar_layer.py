import torch
import torch.nn as nn
import torch.nn.functional as F

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

    @torch.no_grad()
    def forward(self, batched_pts):
        '''
        batched_pts: list[tensor], len(batched_pts) = bs
        return: 
            pillars: (p1 + p2 + ... + pb, num_points, c), 
            coors_batch: (p1 + p2 + ... + pb, 1 + 3), 
            num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        '''
        pillars, coors, npoints_per_pillar = [], [], []
        for i, pts in enumerate(batched_pts):
            pts = pts.T
            voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(pts) 

            pillars.append(torch.from_numpy(voxels_out).cuda())
            coors.append(torch.from_numpy(coors_out).long().cuda())
            npoints_per_pillar.append(torch.from_numpy(num_points_per_voxel_out).cuda())

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
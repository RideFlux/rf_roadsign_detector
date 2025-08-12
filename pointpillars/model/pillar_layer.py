import torch
import torch.nn as nn
import torch.nn.functional as F

from pointpillars.ops.voxel_module import Voxelization

class PillarLayer(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super().__init__()
        
        self.voxel_layer = Voxelization(voxel_size=voxel_size,
                                        point_cloud_range=point_cloud_range,
                                        max_num_points=max_num_points,
                                        max_voxels=max_voxels)

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
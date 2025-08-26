import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, in_channels, voxel_size=None, point_cloud_range=None):
        super().__init__()

        self.num_bev_features = in_channels
        self.nx, self.ny, self.nz = point_cloud_range[3] - point_cloud_range[0], \
                                     point_cloud_range[4] - point_cloud_range[1], \
                                     point_cloud_range[5] - point_cloud_range[2]
        self.nx = int(self.nx / voxel_size[0])
        self.ny = int(self.ny / voxel_size[1])
        self.nz = int(self.nz / voxel_size[2])
        assert self.nz == 1

    def forward(self, batch_dict):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict
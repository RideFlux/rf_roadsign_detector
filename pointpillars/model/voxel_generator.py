import torch
import torch.nn as nn
import numpy as np
import cumm.tensorview as tv

class VoxelGeneratorWrapper(nn.Module):
    def __init__(self, vsize_xyz, coors_range_xyz, max_num_points_per_voxel, max_num_voxels):
        super().__init__()
        self.num_point_features = 4 # x, y, z, intensity : Fixed
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels[0]
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels[0]
            )

    def _to_numpy_points(self, points: torch.Tensor) -> np.ndarray:
        if isinstance(points, torch.Tensor):
            t = points.detach()
            if t.is_cuda:
                t = t.cpu()
            t = t.to(torch.float32).contiguous()
            arr = t.numpy()
        elif isinstance(points, np.ndarray):
            arr = points
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32, copy=False)
            arr = np.ascontiguousarray(arr)
        else:
            raise TypeError(f"points must be torch.Tensor or np.ndarray, got {type(points)}")

        if arr.ndim != 2 or arr.shape[1] != self.num_point_features:
            raise ValueError(
                f"Expected points shape (N, {self.num_point_features}), got {arr.shape}."
            )
        return arr

    @torch.no_grad()
    def forward(self, points):
        pts_np = self._to_numpy_points(points)

        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(pts_np)
            if isinstance(voxel_output, dict):
                voxels = voxel_output['voxels']
                coordinates = voxel_output['coordinates']
                num_points = voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            tv_points = tv.from_numpy(pts_np)
            tv_voxels, tv_coordinates, tv_num_points = self._voxel_generator.point_to_voxel(tv_points)
            voxels = tv_voxels.numpy().copy()
            coordinates = tv_coordinates.numpy().copy()
            num_points = tv_num_points.numpy().copy()

        return voxels, coordinates, num_points

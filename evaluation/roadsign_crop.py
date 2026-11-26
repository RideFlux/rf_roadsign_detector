"""The 3 x 64 x 32 projection used to train the roadsign classifier."""

import numpy as np
import torch


ELEVATION_ANGLES = np.array([
    14.882, 11.032, 8.059, 5.057, 3.04, 2.028, 1.86, 1.688, 1.522,
    1.351, 1.184, 1.013, 0.846, 0.675, 0.508, 0.337, 0.169, 0.0,
    -0.169, -0.337, -0.508, -0.675, -0.845, -1.013, -1.184, -1.351,
    -1.522, -1.688, -1.86, -2.028, -2.198, -2.365, -2.536, -2.7,
    -2.873, -3.04, -3.21, -3.375, -3.548, -3.712, -3.884, -4.05,
    -4.221, -4.385, -4.558, -4.72, -4.892, -5.057, -5.229, -5.391,
    -5.565, -5.726, -5.898, -6.061, -7.063, -8.059, -9.06, -9.885,
    -11.032, -12.006, -12.974, -13.93, -18.889, -24.897,
])
VERTICAL_TANGENTS = np.tan(np.radians(ELEVATION_ANGLES))


def build_classifier_crop(points, bbox):
    """Project a predicted box using the training crop's exact spatial rules.

    Work on CPU, as generate_crop_box.py does. The x width is enlarged by
    20%, y spans center +/- 1 m, and z is deliberately not filtered. Crops
    with fewer than 50 or more than 50,000 points stay zero, matching training.
    """
    points = torch.as_tensor(points).detach().cpu()
    bbox = torch.as_tensor(bbox).detach().cpu()
    image = torch.zeros((3, 64, 32), dtype=torch.float32)
    if bbox.shape != (6,) or not torch.isfinite(bbox).all() or (bbox[3:] <= 0).any():
        raise ValueError("Expected a finite [cx, cy, z_min, dx, dy, dz] box with positive dimensions")
    cx, cy, _, dx, _, _ = bbox
    min_x, max_x = cx - dx * 1.2 / 2, cx + dx * 1.2 / 2
    mask = ((points[:, 0] >= min_x) & (points[:, 0] <= max_x)
            & (points[:, 1] >= cy - 1) & (points[:, 1] <= cy + 1))
    points = points[mask]
    if not 50 <= len(points) <= 50000:
        return image

    xyz = points[:, :3].numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        tangent = xyz[:, 2] / np.hypot(xyz[:, 0], xyz[:, 1])
    rows = torch.from_numpy(np.abs(tangent[:, None] - VERTICAL_TANGENTS).argmin(axis=1))
    cols = ((max_x - points[:, 0]) / (max_x - min_x) * 32).long().clamp(0, 31)
    indices = rows * 32 + cols
    count = torch.zeros(64 * 32)
    intensity = torch.zeros_like(count)
    y_distance = torch.zeros_like(count)
    count.index_add_(0, indices, torch.ones(len(points)))
    intensity.index_add_(0, indices, points[:, 3].float())
    y_distance.index_add_(0, indices, points[:, 1].float())
    image[0] = (intensity / count.clamp_min(1) / 255).reshape(64, 32)
    image[1] = (-y_distance / count.clamp_min(1) / 40).reshape(64, 32)
    image[2] = count.reshape(64, 32)
    return image

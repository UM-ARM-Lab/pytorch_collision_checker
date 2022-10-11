import numpy as np
import torch
import torch.nn.functional as F

from pytorch_collision_checker.utils import handle_batch_input


class SDF:
    origin_point: torch.tensor  # [b, 3]
    res: torch.tensor  # [b, 1]
    sdf: torch.tensor  # [b, n_x, n_y, n_z]

    def __init__(self, origin_point, res, sdf):
        self.origin_point = origin_point
        self.res = res
        self.sdf = sdf

    @handle_batch_input(n=2)
    def get_signed_distance(self, positions):
        """
        Note this function does _not_ operate on _multiple_ sdfs in batch, but just _one_ SDF

        Args:
            positions: [b, 3]

        Returns:

        """
        indices = point_to_idx(positions, self.origin_point, self.res).long()
        x_indices, y_indices, z_indices = torch.unbind(indices, dim=-1)
        b, _ = positions.shape
        # NOTE: we handle OOB points by padding
        shape = torch.tensor(self.sdf.shape[1:], dtype=self.sdf.dtype, device=self.sdf.device)
        before = torch.clamp(-indices, min=0).max(dim=0)[0]
        after = torch.clamp(indices - (shape - 1), min=0).max(dim=0)[0]
        before_x, before_y, before_z = before
        after_x, after_y, after_z = after
        padding = [int(before_z), int(after_z), int(before_y), int(after_y), int(before_x), int(after_x)]
        padded_sdf = F.pad(self.sdf, padding, value=999)
        x_indices_padded = x_indices + before_x
        y_indices_padded = y_indices + before_y
        z_indices_padded = z_indices + before_z
        zero_indices = torch.zeros_like(x_indices_padded)
        distances = padded_sdf[zero_indices, x_indices_padded, y_indices_padded, z_indices_padded]
        return distances

    def to(self, dtype=None, device=None):
        other = self.clone()
        device = device if device is not None else other.sdf.device
        dtype = dtype if dtype is not None else other.sdf.dtype
        other.sdf = other.sdf.to(dtype=dtype, device=device)
        other.origin_point = other.origin_point.to(dtype=dtype, device=device)
        other.res = other.res.to(dtype=dtype, device=device)
        return other

    def clone(self):
        return SDF(self.origin_point.clone(), self.res.clone(), self.sdf.clone())


def point_to_idx(points, origin_point, res):
    # round helps with stupid numerics issues
    return torch.round((points - origin_point) / res).long()


def idx_to_point_from_origin_point(row, col, channel, resolution, origin_point):
    x = origin_point[0] + row * resolution - resolution / 2
    y = origin_point[1] + col * resolution - resolution / 2
    z = origin_point[2] + channel * resolution - resolution / 2

    return np.array([x, y, z])


def extent_to_env_size(extent):
    xmin, xmax, ymin, ymax, zmin, zmax = extent
    env_x_m = abs(xmax - xmin)
    env_y_m = abs(ymax - ymin)
    env_z_m = abs(zmax - zmin)
    return env_x_m, env_y_m, env_z_m


def extent_to_env_shape(extent, res):
    extent = np.array(extent).astype(np.float32)
    res = np.float32(res)
    env_x_m, env_y_m, env_z_m = extent_to_env_size(extent)
    env_x_rows = int(env_x_m / res)
    env_y_cols = int(env_y_m / res)
    env_z_channels = int(env_z_m / res)
    return env_x_rows, env_y_cols, env_z_channels

import numpy as np
import torch
import torch.nn.functional as F

from geometry_msgs.msg import Point
from moonshine.numpify import numpify
from pytorch_collision_checker.utils import handle_batch_input
from ros_numpy import msgify
from rviz_voxelgrid_visuals.conversions import vox_to_float_array
from rviz_voxelgrid_visuals_msgs.msg import VoxelgridStamped


class SDF:
    origin_point: torch.tensor  # [b, 3]
    res: torch.tensor  # [b, 1]
    sdf: torch.tensor  # [b, n_x, n_y, n_z]

    def __init__(self, origin_point, res, sdf):
        self.origin_point = origin_point
        self.res = res
        self.sdf = sdf

        from time import perf_counter
        t0 = perf_counter()
        cx = torch.arange(-1, 2, dtype=res.dtype).reshape([3, 1, 1]).tile([1, 3, 3])[None, None]
        cy = torch.arange(-1, 2, dtype=res.dtype).reshape([1, 3, 1]).tile([3, 1, 3])[None, None]
        cz = torch.arange(-1, 2, dtype=res.dtype).reshape([1, 1, 3]).tile([3, 3, 1])[None, None]
        d = self.sdf.unsqueeze(1)
        d_conv_cx = torch.conv3d(d, cx, padding='same')
        d_conv_cy = torch.conv3d(d, cy, padding='same')
        d_conv_cz = torch.conv3d(d, cz, padding='same')
        self.grad = torch.stack([d_conv_cx, d_conv_cy, d_conv_cz], dim=-1).squeeze(1)
        self.penetration_grad = (self.sdf.unsqueeze(-1) <= 0) * self.grad
        print(f'dt to compute gradient: {perf_counter() - t0:.4f}')

    @handle_batch_input(n=2)
    def get_signed_distance(self, positions):
        """
        Note this function does _not_ operate on _multiple_ sdfs in batch, but just _one_ SDF

        Args:
            positions: [b, 3]

        Returns:

        """
        dtype = self.res.dtype
        device = self.res.device
        indices = point_to_idx(positions, self.origin_point, self.res).long()

        shape = torch.tensor(self.sdf.shape[1:], dtype=dtype, device=device)

        is_oob = torch.any((indices < 0) | (indices >= shape), dim=-1)
        batch_oob_i, = torch.where(is_oob)
        batch_ib_i, = torch.where(~is_oob)
        ib_indices = indices[batch_ib_i]
        oob_d = 999
        oob_distances_flat = torch.ones_like(batch_oob_i, dtype=dtype) * oob_d
        ib_x_i, ib_y_i, ib_z_i = torch.unbind(ib_indices, dim=-1)
        ib_zeros_i = torch.zeros_like(ib_x_i)
        ib_distances_flat = self.sdf[ib_zeros_i, ib_x_i, ib_y_i, ib_z_i]
        distances = torch.zeros([positions.shape[0]]).to(dtype=dtype, device=device)
        distances[batch_oob_i] = oob_distances_flat
        distances[batch_ib_i] = ib_distances_flat
        return distances

    @handle_batch_input(n=2)
    def interp_distance_differentiable(self, positions):
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
        other.grad = other.grad.to(dtype=dtype, device=device)
        other.penetration_grad = other.penetration_grad.to(dtype=dtype, device=device)
        return other

    def clone(self):
        return SDF(self.origin_point.clone(), self.res.clone(), self.sdf.clone())

    def to_voxelgrid(self):
        return self.sdf < 0

    def viz(self, pub):
        vg_np = numpify(self.to_voxelgrid().squeeze())
        op_np = numpify(self.origin_point)
        res_np = numpify(self.res)
        for _ in range(10):
            visualize_vg(pub, vg_np, op_np, res_np)


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


def visualize_vg(pub, vg, origin_point, res):
    origin_point_viz = origin_point - res
    msg = VoxelgridStamped()
    msg.header.frame_id = 'world'
    msg.origin = msgify(Point, origin_point_viz)
    msg.scale = float(res)
    msg.occupancy = vox_to_float_array(vg)
    pub.publish(msg)

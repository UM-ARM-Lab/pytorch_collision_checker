from dataclasses import dataclass

import torch


@dataclass
class SDF:
    origin_point: torch.tensor  # [b, 3]
    res: torch.tensor  # [b, 1]
    sdf: torch.tensor  # [b, n_x, n_y, n_z]


def make_sdf(voxelgrid: torch.tensor, origin, res):
    return
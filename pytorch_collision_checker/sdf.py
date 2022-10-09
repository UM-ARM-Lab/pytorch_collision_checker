import torch


class SDF:
    origin_point: torch.tensor  # [b, 3]
    res: torch.tensor  # [b, 1]
    sdf: torch.tensor  # [b, n_x, n_y, n_z]

    def __init__(self, origin_point, res, sdf):
        self.origin_point = origin_point
        self.res = res
        self.sdf = sdf

    def get_signed_distance(self, positions):
        indices = (positions - self.origin_point) / self.res
        return self.sdf[indices]

    def to(self, dtype=None, device=None):
        other = self.clone()
        device = device if device is None else other.sdf.device
        dtype = dtype if dtype is not None else other.sdf.dtype
        other.sdf = other.sdf.to(dtype=dtype, device=device)
        other.origin_point = other.origin_point.to(dtype=dtype, device=device)
        other.res = other.res.to(dtype=dtype, device=device)
        return other

    def clone(self):
        return SDF(self.origin_point.clone(), self.res.clone(), self.sdf.clone())
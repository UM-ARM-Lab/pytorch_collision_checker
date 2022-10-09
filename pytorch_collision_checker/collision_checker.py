import functools
from copy import deepcopy
from typing import Optional, List, Tuple, Dict

import numpy as np
import torch

from pytorch_collision_checker.sdf import SDF
from pytorch_kinematics import Chain, SerialChain, Transform3d
from pytorch_kinematics.frame import Frame, Link, Joint


def is_tensor_like(x):
    return torch.is_tensor(x) or type(x) is np.ndarray


# from arm_pytorch_utilities, standalone since that package is not on pypi yet
def handle_batch_input(n):
    def _handle_batch_input(func):
        """For func that expect 2D input, handle input that have more than 2 dimensions by flattening them temporarily"""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # assume inputs that are tensor-like have compatible shapes and is represented by the first argument
            batch_dims = []
            for arg in args:
                if is_tensor_like(arg) and len(arg.shape) > n:
                    batch_dims = arg.shape[:-(n - 1)]  # last dimension is type dependent; all previous ones are batches
                    break
            # no batches; just return normally
            if not batch_dims:
                return func(*args, **kwargs)

            # reduce all batch dimensions down to the first one
            args = [v.view(-1, *v.shape[-(n - 1):]) if (is_tensor_like(v) and len(v.shape) > 2) else v for v in args]
            ret = func(*args, **kwargs)
            # restore original batch dimensions; keep variable dimension (nx)
            if type(ret) is tuple:
                ret = [v if (not is_tensor_like(v) or len(v.shape) == 0) else (
                    v.view(*batch_dims, *v.shape[-(n - 1):]) if len(v.shape) == n else v.view(*batch_dims)) for v in
                       ret]
            else:
                if is_tensor_like(ret):
                    if len(ret.shape) == n:
                        ret = ret.view(*batch_dims, *ret.shape[-(n - 1):])
                    else:
                        ret = ret.view(*batch_dims)
            return ret

        return wrapper

    return _handle_batch_input


def pairwise_distances(a):
    """

    Args:
        a:  [b, n, k]

    Returns: [b, n, n]

    """
    a_sum_sqr = a.square().sum(dim=-1, keepdims=True)  # [b, n, 1]
    dist = a_sum_sqr - 2 * torch.matmul(a, a.transpose(1, 2)) + a_sum_sqr.transpose(1, 2)  # [b, n, n]
    return dist.sqrt()


def get_radii(chain, spheres):
    radii = []
    repeats = []
    sphere_idx_to_link_idx = []
    for link_idx, link_name in enumerate(chain.get_link_names()):
        if link_name in spheres:
            spheres_for_link = spheres[link_name]
            repeats.append(len(spheres_for_link))
            for sphere_for_link in spheres_for_link:
                sphere_idx_to_link_idx.append(link_idx)
                radii.append(sphere_for_link["radius"])
        else:
            repeats.append(1)
            radii.append(0)  # FIXME: won't actually prevent collision from being detected
    radii = torch.tensor(radii, dtype=chain.dtype, device=chain.device)
    repeats = torch.tensor(repeats, dtype=torch.int, device=chain.device)
    sphere_idx_to_link_idx = torch.tensor(sphere_idx_to_link_idx, dtype=torch.int, device=chain.device)
    return radii, repeats, sphere_idx_to_link_idx


class CollisionChecker:

    def __init__(self, chain: Chain,
                 spheres: Dict,
                 sdf: Optional[SDF] = None,
                 ignore_collision_pairs: List[Tuple[str, str]] = None):
        """

        Args:
            chain:
            spheres:
            sdf:
            ignore_collision_pairs: a list of links where collision is allowed (meaning it is ignored)
        """
        if isinstance(chain, SerialChain):
            raise NotImplementedError("only Chain supported")
        self.chain = deepcopy(chain)
        self.dtype = self.chain.dtype
        self.device = self.chain.device
        self.sdf = sdf
        self.radii, self.repeats, self.sphere_idx_to_link_idx = get_radii(chain, spheres)
        self.link_names = self.chain.get_link_names()
        self.n_spheres = self.radii.shape[0]
        self.ignored_collision_matrix = torch.eye(self.n_spheres, dtype=self.chain.dtype, device=self.chain.device)
        if ignore_collision_pairs is not None:
            cumsum = torch.cumsum(self.repeats, dim=0)
            sphere_start_indices = cumsum - cumsum[0]
            sphere_end_indices = cumsum
            for link_a, link_b in ignore_collision_pairs:
                link_idx_a = self.link_names.index(link_a)
                link_idx_b = self.link_names.index(link_b)
                for sphere_idx_a in torch.arange(sphere_start_indices[link_idx_a], sphere_end_indices[link_idx_a]):
                    for sphere_idx_b in torch.arange(sphere_start_indices[link_idx_b], sphere_end_indices[link_idx_b]):
                        self.ignored_collision_matrix[sphere_idx_a, sphere_idx_b] = 1
                        self.ignored_collision_matrix[sphere_idx_b, sphere_idx_a] = 1

        radii_a = torch.tile(self.radii[None, None], [1, self.n_spheres, 1])
        radii_b = torch.transpose(radii_a, 1, 2)
        self.radii_matrix = radii_a + radii_b

        for link_name, spheres_for_link in spheres.items():
            link_frame = self.chain.find_frame(f"{link_name}_frame")
            for sphere_idx, sphere in enumerate(spheres_for_link):
                pos = torch.tensor(sphere['position'], dtype=self.chain.dtype, device=self.chain.device)
                name = f"{link_name}_sphere_{sphere_idx}"
                offset = Transform3d(pos=pos, dtype=self.chain.dtype, device=self.chain.device)
                joint = Joint(name=name, dtype=self.chain.dtype, device=self.chain.device, joint_type='fixed')
                link_frame.add_child(Frame(name=name + "_frame",
                                           link=Link(name=name, offset=offset),
                                           joint=joint))

    @handle_batch_input(n=2)
    def get_all_self_collision_pairs(self, joint_positions):
        sphere_positions = self.compute_sphere_positions(joint_positions)
        in_self_collision = self.compute_self_distance_matrix(sphere_positions)
        batch_indices, a_indices, b_indices = torch.where(in_self_collision)
        pairs = []
        for batch_idx, sphere_a_idx, sphere_b_idx in torch.stack([batch_indices, a_indices, b_indices], dim=-1):
            link_a_idx = self.sphere_idx_to_link_idx[sphere_a_idx]
            link_b_idx = self.sphere_idx_to_link_idx[sphere_b_idx]
            link_a_name = self.link_names[link_a_idx]
            link_b_name = self.link_names[link_b_idx]
            pair = (link_a_name, link_b_name)
            pair_reverse = (link_b_name, link_a_name)
            if pair not in pairs and pair_reverse not in pairs:
                pairs.append(pair)
        return pairs

    @handle_batch_input(n=2)
    def check_collision(self, joint_positions):
        sphere_positions = self.compute_sphere_positions(joint_positions)
        in_self_collision = self.compute_self_distance_matrix(sphere_positions)
        in_collision_any = in_self_collision.any(dim=2)

        if self.sdf is not None:
            d_to_env = self.sdf.get_signed_distance(sphere_positions)
            in_env_collision = d_to_env < self.radii[None]
            in_collision_any = torch.logical_or(in_collision_any, in_env_collision.any(dim=1))

        return in_collision_any

    @handle_batch_input(n=2)
    def compute_self_distance_matrix(self, sphere_positions):
        d_to_self = pairwise_distances(sphere_positions)
        d_to_self_ignored = d_to_self + self.ignored_collision_matrix * 999
        in_self_collision = d_to_self_ignored < self.radii_matrix
        return in_self_collision

    @handle_batch_input(n=2)
    def compute_sphere_positions(self, joint_positions):
        transforms = self.chain.forward_kinematics(joint_positions)
        sphere_positions = []
        for k, t in transforms.items():
            if 'sphere' in k:
                sphere_positions.append(t.get_matrix()[:, :3, 3])
        sphere_positions_vec = torch.stack(sphere_positions, dim=1)
        return sphere_positions_vec


def get_default_ignores(chain: Chain, spheres: Dict):
    cc = CollisionChecker(chain, spheres)
    n_config_samples = 1000
    rng = np.random.RandomState(0)
    low, high = chain.get_joint_limits()
    pair_count_map = {}
    for _ in range(n_config_samples):
        n_joints = len(chain.get_joint_parameter_names())
        colliding_pairs = cc.get_all_self_collision_pairs(rng.uniform(low, high, size=n_joints))
        for pair in colliding_pairs:
            if pair not in pair_count_map:
                pair_count_map[pair] = 0
            pair_count_map[pair] += 1

    ignore = []
    for pair, count in pair_count_map.items():
        if count >= n_config_samples:
            ignore.append(pair)

    return ignore

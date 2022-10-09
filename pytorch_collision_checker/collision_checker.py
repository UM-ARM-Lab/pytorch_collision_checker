import functools
from typing import Optional, List, Tuple

import numpy as np
import torch

from pytorch_kinematics import Chain, SerialChain


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


class CollisionChecker:

    def __init__(self, chain: Chain,
                 radii: torch.tensor,
                 env: Optional[torch.tensor] = None,
                 ignore_collision_pairs: List[Tuple[str, str]] = None):
        """

        Args:
            chain:
            radii:
            env:
            ignore_collision_pairs: a list of links where collision is allowed (meaning it is ignored)
        """
        if isinstance(chain, SerialChain):
            raise NotImplementedError("only Chain supported")
        self.chain = chain
        self.env = env
        self.radii = radii
        self.link_names = self.chain.get_link_names()
        self.n_links = len(self.link_names)
        self.ignored_collision_matrix = torch.eye(self.n_links)
        if ignore_collision_pairs is not None:
            for link_a, link_b in ignore_collision_pairs:
                idx_a = self.link_names.index(link_a)
                idx_b = self.link_names.index(link_b)
                self.ignored_collision_matrix[idx_a, idx_b] = 1
                self.ignored_collision_matrix[idx_b, idx_a] = 1

        radii_a = torch.tile(self.radii.unsqueeze(1), [1, self.n_links, 1])
        radii_b = torch.transpose(radii_a, 1, 2)
        self.radii_matrix = radii_a + radii_b

    @handle_batch_input(n=2)
    def check_collision(self, joint_positions, return_all_pairs=False):
        """

        Args:
            joint_positions: [b, n_joints]

        Returns:
            boolean tensor [b, 1]

        """
        transforms = self.chain.forward_kinematics(joint_positions)

        positions = [t.get_matrix()[:, :3, 3] for t in transforms.values()]
        positions_vec = torch.stack(positions, dim=1)
        d = pairwise_distances(positions_vec)
        d_ignored = d + self.ignored_collision_matrix * 999
        in_collision = d_ignored < self.radii_matrix
        in_collision_any = in_collision.any(dim=2)

        if return_all_pairs:
            batch_indices, a_indices, b_indices = torch.where(in_collision)
            pairs = []
            for batch_idx, a_idx, b_idx in torch.stack([batch_indices, a_indices, b_indices], dim=-1):
                link_a_name = self.link_names[a_idx]
                link_b_name = self.link_names[b_idx]
                pair = (link_a_name, link_b_name)
                pair_reverse = (link_b_name, link_a_name)
                if pair not in pairs and pair_reverse not in pairs:
                    pairs.append(pair)
            return pairs
        else:
            return in_collision_any


def get_default_ignores(chain: Chain, radii):
    # parent_child_pairs = []
    #
    # def _traverse_chain(parent_frame):
    #     for child in parent_frame.children:
    #         parent_child_pairs.append((parent_frame.link.name, child.link.name))
    #         _traverse_chain(child)
    #
    # _traverse_chain(chain._root)

    # TODO: another idea would be to check which links are in collision for most configs?
    cc = CollisionChecker(chain, radii)
    n_config_samples = 1000
    rng = np.random.RandomState(0)
    low, high = chain.get_joint_limits()
    pair_count_map = {}
    for _ in range(n_config_samples):
        n_joints = len(chain.get_joint_parameter_names())
        colliding_pairs = cc.check_collision(rng.uniform(low, high, size=n_joints), return_all_pairs=True)
        for pair in colliding_pairs:
            if pair not in pair_count_map:
                pair_count_map[pair] = 0
            pair_count_map[pair] += 1

    ignore = []
    for pair, count in pair_count_map.items():
        if count >= n_config_samples:
            ignore.append(pair)

    return ignore

import json
import pathlib
import pickle
from copy import deepcopy
from typing import Optional, List, Tuple, Dict

import torch

import pytorch_kinematics as pk
from moonshine.geometry_torch import transform_points_3d
from pytorch_collision_checker.sdf import SDF
from pytorch_collision_checker.utils import handle_batch_input
from pytorch_kinematics import Chain, SerialChain, Transform3d
from pytorch_kinematics.frame import Joint, Frame, Link
from regrasping_deformables.sdf_autograd import sdf_lookup


def load_model_and_cc(model_filename: pathlib.Path, sdf_filename: Optional[pathlib.Path], dtype, device, chain=None,
                      robot2sdf=None):
    if sdf_filename is not None:
        with sdf_filename.open("rb") as f:
            sdf: SDF = pickle.load(f)
        sdf = sdf.to(dtype=dtype, device=device)
    else:
        sdf = None
    spheres_filename = model_filename.parent / (model_filename.stem + '_spheres.json')
    if chain is None:
        chain = pk.build_chain_from_mjcf(model_filename.open().read())
    spheres = load_spheres(spheres_filename)
    chain = chain.to(dtype=dtype, device=device)
    ignores_filename = model_filename.parent / f'{model_filename.stem}_ignore.pkl'
    if ignores_filename.exists():
        with ignores_filename.open("rb") as f:
            ignore = pickle.load(f)
    else:
        ignore = get_default_ignores(chain, spheres)
        print(f"Caching default ignores in {ignores_filename.as_posix()}")
        with ignores_filename.open("wb") as f:
            pickle.dump(ignore, f)
    cc = CollisionChecker(chain, spheres, sdf=sdf, ignore_collision_pairs=ignore, robot2sdf=robot2sdf)
    return cc


def pairwise_distances(a):
    """

    Args:
        a:  [b, n, k]

    Returns: [b, n, n]

    """
    a_sum_sqr = a.square().sum(dim=-1, keepdims=True)  # [b, n, 1]
    dist = a_sum_sqr - 2 * torch.matmul(a, a.transpose(1, 2)) + a_sum_sqr.transpose(1, 2) + 1e-6  # [b, n, n]
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
            repeats.append(0)
    radii = torch.tensor(radii, dtype=chain.dtype, device=chain.device)
    repeats = torch.tensor(repeats, dtype=torch.int, device=chain.device)
    sphere_idx_to_link_idx = torch.tensor(sphere_idx_to_link_idx, dtype=torch.int, device=chain.device)
    return radii, repeats, sphere_idx_to_link_idx


def make_sphere_frame_name(link_name, sphere_idx):
    name = f"{link_name}_sphere_{sphere_idx}"
    return name


def sort_spheres_to_match_kinematics_order(chain, spheres):
    spheres_sorted = {}
    for link_idx, link_name in enumerate(chain.get_link_names()):
        if link_name in spheres:
            spheres_sorted[link_name] = spheres[link_name]
    return spheres_sorted


class CollisionChecker:

    def __init__(self, chain: Chain,
                 spheres: Dict,
                 sdf: Optional[SDF] = None,
                 ignore_collision_pairs: List[Tuple[str, str]] = None,
                 robot2sdf=None):
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
        self.spheres = sort_spheres_to_match_kinematics_order(chain, spheres)
        self.n_spheres = self.radii.shape[0]
        self.ignored_collision_with_env_mask = torch.zeros(self.n_spheres, dtype=self.dtype, device=self.device)
        self.link_names = self.chain.get_link_names()
        self.n_spheres = self.radii.shape[0]
        self.sphere_names = None
        self.sphere_indices = None
        if robot2sdf is None:
            self.robot2sdf = None
        else:
            self.robot2sdf = torch.unsqueeze(robot2sdf, 0)
        self.ignored_collision_matrix = torch.eye(self.n_spheres, dtype=self.chain.dtype, device=self.chain.device)
        if ignore_collision_pairs is not None:
            cumsum = torch.cumsum(self.repeats, dim=0)
            sphere_start_indices = cumsum - self.repeats
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
                name = make_sphere_frame_name(link_name, sphere_idx)
                offset = Transform3d(pos=pos, dtype=self.chain.dtype, device=self.chain.device)
                joint = Joint(name=name, dtype=self.chain.dtype, device=self.chain.device, joint_type='fixed')
                link_frame.add_child(Frame(name=name + "_frame", link=Link(name=name, offset=offset), joint=joint))

        self.chain.precompute_fk_info()
        self.update_precomputed_indices()

    def update_precomputed_indices(self):
        """ You must call this method if you change the underlying Chain to add or remove frames! """
        self.sphere_names, self.sphere_indices = self.get_sphere_frame_names_and_indices()

        n = len(self.chain.get_joint_parameter_names())
        n_spheres = len(self.sphere_names)
        self.joint_effects_sphere_matrix = torch.zeros([n_spheres, n], dtype=torch.bool, device=self.device)
        for sphere_idx, (sphere_name, sphere_frame_idx) in enumerate(zip(self.sphere_names, self.sphere_indices)):
            parent_idx = sphere_frame_idx
            while parent_idx >= 0:
                joint_idx = self.chain.joint_indices[parent_idx]
                if joint_idx != -1:
                    self.joint_effects_sphere_matrix[sphere_idx, joint_idx] = True
                parent_idx = self.chain.parent_indices[parent_idx]

    @handle_batch_input(n=2)
    def get_all_self_collision_pairs(self, joint_positions):
        sphere_positions = self.compute_sphere_positions(joint_positions)
        in_self_collision = self.compute_in_self_collision(sphere_positions)
        percentage_in_self_collision = in_self_collision.sum(dim=0) / in_self_collision.shape[0]
        a_indices, b_indices = torch.where(percentage_in_self_collision > 0.5)
        pairs = []
        for sphere_a_idx, sphere_b_idx in torch.stack([a_indices, b_indices], dim=-1):
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
    def penetration_cost(self, transforms):
        return self.self_penetration_cost(transforms) + self.env_penetration_cost(transforms)

    @handle_batch_input(n=2)
    def self_penetration_cost(self, transforms):
        sphere_positions = self.compute_sphere_positions_from_fk(transforms)
        self_penetration_matrix = self.compute_self_penetration(sphere_positions)
        self_penetration = self_penetration_matrix.sum(dim=-1).sum(dim=-1)
        return self_penetration

    @handle_batch_input(n=2)
    def env_penetration_cost(self, transforms):
        sphere_positions = self.compute_sphere_positions_from_fk(transforms)
        env_penetration = self.compute_env_penetration(sphere_positions)
        env_penetration = env_penetration.sum(dim=-1)
        return env_penetration

    @handle_batch_input(n=2)
    def check_collision(self, joint_positions=None, return_all=False, transforms=None):
        """ If you've already computed the transforms via FK, you should pass it in instead of re-computing it """
        if transforms is None:
            sphere_positions = self.compute_sphere_positions(joint_positions)
        else:
            sphere_positions = self.compute_sphere_positions_from_fk(transforms)

        in_self_collision = self.compute_in_self_collision(sphere_positions)
        in_collision_any = in_self_collision.any(dim=2)

        if self.sdf is not None:
            if self.robot2sdf is not None:
                sphere_positions_sdf_frame = self.positions_to_sdf_frame(sphere_positions)
            else:
                sphere_positions_sdf_frame = sphere_positions
            d_to_env = self.sdf.get_signed_distance(sphere_positions_sdf_frame)
            d_to_env_masked = d_to_env + (1000 * self.ignored_collision_with_env_mask)
            in_env_collision = d_to_env_masked < self.radii[None]
            in_collision_any = torch.logical_or(in_collision_any, in_env_collision)

        if return_all:
            return in_collision_any
        else:
            return in_collision_any.any(dim=1)

    @handle_batch_input(n=2)
    def positions_to_sdf_frame(self, sphere_positions):
        sphere_positions_sdf_frame = transform_points_3d(self.robot2sdf, sphere_positions)
        return sphere_positions_sdf_frame

    @handle_batch_input(n=3)
    def compute_self_penetration(self, sphere_positions):
        d_to_self_ignored = self.compute_self_distance_matrix(sphere_positions)
        self_penetration = torch.relu(self.radii_matrix - d_to_self_ignored)
        return self_penetration

    @handle_batch_input(n=3)
    def compute_env_penetration(self, sphere_positions):
        if self.robot2sdf is not None:
            sphere_positions_sdf_frame = self.positions_to_sdf_frame(sphere_positions)
        else:
            sphere_positions_sdf_frame = sphere_positions

        d_to_env_differentiable = sdf_lookup(self.sdf, sphere_positions_sdf_frame)
        d_to_env_masked = d_to_env_differentiable + (1000 * self.ignored_collision_with_env_mask)
        env_penetration = torch.relu(self.radii - d_to_env_masked)
        return env_penetration

    @handle_batch_input(n=3)
    def compute_in_self_collision(self, sphere_positions):
        d_to_self_ignored = self.compute_self_distance_matrix(sphere_positions)
        in_self_collision = d_to_self_ignored < self.radii_matrix
        return in_self_collision

    def compute_self_distance_matrix(self, sphere_positions):
        d_to_self = pairwise_distances(sphere_positions)
        d_to_self_ignored = d_to_self + self.ignored_collision_matrix * 999
        return d_to_self_ignored

    def get_sphere_frame_names_and_indices(self):
        names = []
        indices = []
        for link_name, spheres_for_link in self.spheres.items():
            for sphere_idx, sphere_for_link in enumerate(spheres_for_link):
                sphere_frame_name = make_sphere_frame_name(link_name, sphere_idx)
                names.append(sphere_frame_name)
                indices.append(self.chain.frame_to_idx[sphere_frame_name + '_frame'])
        return names, torch.tensor(indices, dtype=torch.long, device=self.device)

    @handle_batch_input(n=2)
    def compute_sphere_positions(self, joint_positions):
        transforms = self.chain.forward_kinematics_fast(joint_positions, self.sphere_indices)
        transforms_dict = dict(zip(self.sphere_names, transforms))
        return self.compute_sphere_positions_from_fk(transforms_dict)

    def compute_sphere_positions_from_fk(self, transforms):
        sphere_positions = []
        for link_name, spheres_for_link in self.spheres.items():
            for sphere_idx, sphere_for_link in enumerate(spheres_for_link):
                sphere_frame_name = make_sphere_frame_name(link_name, sphere_idx)
                t = transforms[sphere_frame_name]
                sphere_positions.append(t[:, :3, 3])
        sphere_positions_vec = torch.stack(sphere_positions, dim=1)
        return sphere_positions_vec

    def ignore_collision_with_env_for_sphere(self, link_name: str, per_link_sphere_idx: int):
        full_sphere_idx = self.get_sphere_idx(link_name, per_link_sphere_idx)
        self.ignored_collision_with_env_mask[full_sphere_idx] = 1

    def unignore_collision_with_env_for_sphere(self, link_name: str, per_link_sphere_idx: int):
        full_sphere_idx = self.get_sphere_idx(link_name, per_link_sphere_idx)
        self.ignored_collision_with_env_mask[full_sphere_idx] = 0

    def get_sphere_idx(self, link_name, per_link_sphere_idx):
        cumsum = torch.cumsum(self.repeats, dim=0)
        sphere_start_indices = cumsum - self.repeats
        sphere_end_indices = cumsum
        link_idx = self.link_names.index(link_name)
        sphere_indices_for_link = torch.arange(sphere_start_indices[link_idx], sphere_end_indices[link_idx])
        full_sphere_idx = sphere_indices_for_link[per_link_sphere_idx]
        return full_sphere_idx

    def link_name_for_sphere_idx(self, sphere_idx):
        return self.link_names[self.sphere_idx_to_link_idx[sphere_idx]]


def get_default_ignores(chain: Chain, spheres: Dict):
    cc = CollisionChecker(chain, spheres)
    n_config_samples = 1000
    low, high = chain.get_joint_limits()
    low = torch.tensor(low, dtype=chain.dtype, device=chain.device)
    high = torch.tensor(high, dtype=chain.dtype, device=chain.device)
    rng = torch.distributions.uniform.Uniform(low, high)
    joint_positions = rng.sample([n_config_samples])
    return cc.get_all_self_collision_pairs(joint_positions)


def load_spheres(path):
    with path.open('r') as f:
        spheres = json.load(f)['spheres']
    return spheres

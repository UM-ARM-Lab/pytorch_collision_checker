import argparse
import pathlib
import pickle

import numpy as np
import torch
from dm_control import mjcf
from dm_control import mujoco
from tqdm import tqdm

import sdf_tools.utils_3d
from pytorch_collision_checker.sdf import SDF


def idx_to_point_from_origin_point(row, col, channel, resolution, origin_point):
    y = origin_point[1] + row * resolution
    x = origin_point[0] + col * resolution
    z = origin_point[2] + channel * resolution

    return np.array([x, y, z])


def extent_to_env_size(extent):
    xmin, xmax, ymin, ymax, zmin, zmax = extent
    env_h_m = abs(ymax - ymin)
    env_w_m = abs(xmax - xmin)
    env_c_m = abs(zmax - zmin)
    return env_h_m, env_w_m, env_c_m


def extent_to_env_shape(extent, res):
    extent = np.array(extent).astype(np.float32)
    res = np.float32(res)
    env_h_m, env_w_m, env_c_m = extent_to_env_size(extent)
    env_h_rows = int(env_h_m / res)
    env_w_cols = int(env_w_m / res)
    env_c_channels = int(env_c_m / res)
    return env_h_rows, env_w_cols, env_c_channels


def get_voxelgrid(model_filename, res, extent, origin_point):
    model = mjcf.from_path(model_filename.as_posix())
    cc_sphere = mjcf.element.RootElement(model='vgb_sphere')
    cc_sphere.worldbody.add('geom', name='geom', type='sphere', size=[res])
    cc_sphere_frame = model.attach(cc_sphere)
    cc_sphere_frame.add('freejoint')
    physics = mjcf.Physics.from_mjcf_model(model)

    res = np.float32(res)
    shape = extent_to_env_shape(extent, res)

    geom_type = mujoco.mju_str2Type('geom')

    def in_collision(xyz):
        attachment_frame = mjcf.get_attachment_frame(cc_sphere)
        physics.bind(attachment_frame).pos = xyz
        mujoco.mj_step1(physics.model.ptr, physics.data.ptr)
        for i, c in enumerate(physics.data.contact):
            geom1_name = mujoco.mj_id2name(physics.model.ptr, geom_type, c.geom1)
            geom2_name = mujoco.mj_id2name(physics.model.ptr, geom_type, c.geom2)
            if c.dist < 0 and (geom1_name == 'vgb_sphere/geom' or geom2_name == 'vgb_sphere/geom'):
                # print(f"Contact at {xyz} between {geom1_name} and {geom2_name}, {c.dist=:.4f} {c.exclude=}")
                return True
        return False

    vg = np.zeros(shape, dtype=np.float32)
    for row, col, channel in tqdm(list(np.ndindex(*shape))):
        xyz = idx_to_point_from_origin_point(row, col, channel, res, origin_point)
        if in_collision(xyz):
            vg[row, col, channel] = 1

    return vg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_filename', type=pathlib.Path)
    parser.add_argument('res', type=float)
    parser.add_argument('xmin', type=float)
    parser.add_argument('xmax', type=float)
    parser.add_argument('ymin', type=float)
    parser.add_argument('ymax', type=float)
    parser.add_argument('zmin', type=float)
    parser.add_argument('zmax', type=float)

    args = parser.parse_args()

    outfilename = args.model_filename.parent / (args.model_filename.stem + ".pkl")
    origin_point = torch.tensor([args.xmin, args.ymin, args.zmin]) + args.res / 2
    res = torch.tensor([args.res])
    extent = np.array([args.xmin, args.xmax, args.ymin, args.ymax, args.zmin, args.zmax])
    vg = get_voxelgrid(args.model_filename, res, extent, origin_point)
    sdf, _ = sdf_tools.utils_3d.compute_sdf_and_gradient(vg, res, origin_point)
    sdf = torch.tensor(sdf)

    sdf = SDF(origin_point=origin_point, res=res, sdf=sdf)

    with outfilename.open("wb") as f:
        pickle.dump(sdf, f)


if __name__ == '__main__':
    main()

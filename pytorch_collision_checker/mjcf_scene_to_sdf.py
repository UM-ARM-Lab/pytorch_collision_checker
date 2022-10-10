import argparse
import pathlib
import pickle

import numpy as np
import torch
from dm_control import mjcf
from dm_control import mujoco
from tqdm import tqdm

import rospy
import sdf_tools.utils_3d
from pytorch_collision_checker.collision_visualizer import visualize_vg, MujocoVisualizer
from pytorch_collision_checker.sdf import idx_to_point_from_origin_point, extent_to_env_shape, SDF
from rviz_voxelgrid_visuals_msgs.msg import VoxelgridStamped


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
        physics.data.qpos[0:3] = xyz  # set the position, uses the "free joint" we made above
        mujoco.mj_step1(physics.model.ptr, physics.data.ptr)  # call step to update collisions
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
    rospy.init_node("mjcf_scene_to_sdf")

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

    outfilename = args.model_filename.parent / (args.model_filename.stem + "_sdf.pkl")
    origin_point = torch.tensor([args.xmin, args.ymin, args.zmin]) + args.res / 2  # center  of the voxel [0,0,0
    res = torch.tensor([args.res])
    extent = np.array([args.xmin, args.xmax, args.ymin, args.ymax, args.zmin, args.zmax])

    # seems like there's a small error here somewhere
    vg = get_voxelgrid(args.model_filename, res, extent, origin_point)

    sdf, _ = sdf_tools.utils_3d.compute_sdf_and_gradient(vg, res, origin_point)
    sdf = torch.tensor(sdf).unsqueeze(0)

    sdf = SDF(origin_point=origin_point, res=res, sdf=sdf)

    mj_viz = MujocoVisualizer()
    env_physics_for_viz = mujoco.Physics.from_xml_path(args.model_filename.as_posix())
    pub = rospy.Publisher("vg", VoxelgridStamped, queue_size=10)
    for _ in range(10):
        visualize_vg(pub, vg, origin_point, res)
        mj_viz.viz(env_physics_for_viz)
        rospy.sleep(0.1)

    with outfilename.open("wb") as f:
        pickle.dump(sdf, f)


if __name__ == '__main__':
    main()

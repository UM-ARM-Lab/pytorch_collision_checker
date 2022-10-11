import argparse
import pathlib

import numpy as np
import torch
from dm_control import mjcf
from dm_control import mujoco

import rospy
from pytorch_collision_checker.collision_checker import load_model_and_cc
from pytorch_collision_checker.collision_visualizer import CollisionVisualizer, MujocoVisualizer


def main():
    torch.set_printoptions(precision=4, sci_mode=False, linewidth=220)
    rospy.init_node('collision_checking_demo')
    parser = argparse.ArgumentParser()
    parser.add_argument('model_filename', type=pathlib.Path)
    parser.add_argument('sdf_filename', type=pathlib.Path)
    parser.add_argument('env_filename', type=pathlib.Path)

    args = parser.parse_args()

    dtype = torch.float64
    device = 'cuda'
    cc = load_model_and_cc(args.model_filename, args.sdf_filename, dtype, device)
    n_joints = len(cc.chain.get_joint_parameter_names(True))

    env_model = mjcf.from_path(args.env_filename.as_posix())
    robot_model = mjcf.from_path(args.model_filename.as_posix())
    env_model.attach(robot_model)
    physics = mjcf.Physics.from_mjcf_model(env_model)

    cc_viz = CollisionVisualizer()
    mj_viz = MujocoVisualizer()

    rng = np.random.RandomState(0)
    for i in range(10):
        joint_positions = rng.randn(1, n_joints)
        # joint_positions = np.zeros([1, 8])

        physics.data.qpos[:] = joint_positions[0]
        mujoco.mj_step1(physics.model.ptr, physics.data.ptr)

        in_collision_any = cc.check_collision(joint_positions, return_all=True)  # [b, n_spheres]
        in_collision = in_collision_any.any()
        if in_collision:
            print("Collision!")
        highlight_indices = torch.where(in_collision_any.squeeze(dim=0))[0]
        sphere_positions = cc.compute_sphere_positions(joint_positions)
        for _ in range(3):
            cc_viz.viz(sphere_positions.squeeze(dim=0), cc.radii.squeeze(dim=0), highlight_indices=highlight_indices)
            mj_viz.viz(physics, alpha=1)
            rospy.sleep(0.1)
        input("press enter to see next config")


if __name__ == '__main__':
    main()

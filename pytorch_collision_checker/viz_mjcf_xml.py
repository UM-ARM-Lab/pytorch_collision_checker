#!/usr/bin/env python
import argparse
import pathlib

from dm_control import mujoco

import rospy
from pytorch_collision_checker.collision_visualizer import MujocoVisualizer


def main():
    rospy.init_node("viz_mjcf_xml")
    parser = argparse.ArgumentParser()
    parser.add_argument('model_filename', type=pathlib.Path)
    args = parser.parse_args()

    mj_viz = MujocoVisualizer()

    print("Add a MarkerArray to RViz and set it to the topic mj_geoms")
    print("You can edit the XML file and leave this script running and see the changes")
    while True:
        rospy.sleep(1)
        try:
            physics = mujoco.Physics.from_xml_string(args.model_filename.open().read())
        except Exception:
            pass
        mj_viz.viz(physics)


if __name__ == '__main__':
    main()

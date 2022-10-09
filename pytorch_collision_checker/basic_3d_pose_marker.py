#!/usr/bin/env python

import numpy as np

from geometry_msgs.msg import Point, Pose
from interactive_markers.interactive_marker_server import *
from visualization_msgs.msg import *


def make_interactive_marker(name: str, position: Point):
    imarker = InteractiveMarker()
    imarker.header.frame_id = "robot_root"
    imarker.pose.position = position
    imarker.pose.orientation.w = 1
    imarker.scale = 0.8

    imarker.name = name

    control = InteractiveMarkerControl()
    control.always_visible = True
    imarker.controls.append(control)

    imarker.controls[0].interaction_mode = InteractiveMarkerControl.MOVE_3D
    q = np.sqrt(2) / 2

    control = InteractiveMarkerControl()
    control.orientation.w = q
    control.orientation.x = q
    control.orientation.y = 0
    control.orientation.z = 0
    control.name = "move_x"
    control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    imarker.controls.append(control)

    control = InteractiveMarkerControl()
    control.orientation.w = q
    control.orientation.x = 0
    control.orientation.y = q
    control.orientation.z = 0
    control.name = "move_z"
    control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    imarker.controls.append(control)

    control = InteractiveMarkerControl()
    control.orientation.w = q
    control.orientation.x = 0
    control.orientation.y = 0
    control.orientation.z = q
    control.name = "move_y"
    control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    imarker.controls.append(control)
    return imarker


class Basic3DPoseInteractiveMarker:
    def __init__(self,
                 cb):
        position = Point(0, 0, 0)
        self.server = InteractiveMarkerServer("basic_3d_imarkers")

        self.marker_name = 'my_imarker'
        self.imarker = make_interactive_marker(self.marker_name, position)
        self.server.insert(self.imarker, self.on_feedback)
        self.cb = cb

        self.server.applyChanges()

    def on_feedback(self, feedback: InteractiveMarkerFeedback):
        self.cb(feedback)
        self.server.applyChanges()

    def set_pose(self, pose: Pose):
        self.server.setPose(self.imarker.name, pose)
        self.server.applyChanges()

    def get_pose(self) -> Pose:
        return self.server.get(self.marker_name).pose

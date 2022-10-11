from copy import deepcopy

from pytorch_collision_checker.utils import homogeneous_np
from tf import transformations

import mujoco
import numpy as np
from dm_control.mjcf import Physics
from mujoco import mju_str2Type, mju_mat2Quat, mjtGeom, mj_id2name

import rospy
from geometry_msgs.msg import Point
from ros_numpy import msgify
from rviz_voxelgrid_visuals.conversions import vox_to_float_array
from rviz_voxelgrid_visuals_msgs.msg import VoxelgridStamped
from visualization_msgs.msg import MarkerArray, Marker

def make_delete_marker(marker_id: int = 0, ns: str = ''):
    m = Marker(action=Marker.DELETEALL, ns=ns, id=marker_id)
    return m


def make_delete_markerarray(marker_id: int = 0, ns: str = ''):
    m = Marker(action=Marker.DELETEALL, ns=ns, id=marker_id)
    msg = MarkerArray(markers=[m])
    return msg

class CollisionVisualizer:

    def __init__(self):
        self.spheres_pub = rospy.Publisher('spheres', MarkerArray, queue_size=10)
        self.geoms_pub = rospy.Publisher('geoms', MarkerArray, queue_size=10)

    def viz(self, sphere_positions, radii, highlight_indices=None):
        self.spheres_pub.publish(make_delete_markerarray())

        assert sphere_positions.ndim == 2
        assert radii.ndim == 1
        msg = MarkerArray()
        idx = 0
        for i, (pos, r) in enumerate(zip(sphere_positions, radii)):
            sphere_msg = Marker()
            sphere_msg.scale.x = 2 * r
            sphere_msg.scale.y = 2 * r
            sphere_msg.scale.z = 2 * r
            sphere_msg.color.a = 0.5
            if highlight_indices is not None and i in highlight_indices:
                sphere_msg.color.r = 0.9
            else:
                sphere_msg.color.r = 0.4
            sphere_msg.color.g = 0.4
            sphere_msg.color.b = 0.4
            sphere_msg.action = Marker.ADD
            sphere_msg.header.frame_id = 'world'
            sphere_msg.pose.position.x = pos[0]
            sphere_msg.pose.position.y = pos[1]
            sphere_msg.pose.position.z = pos[2]
            sphere_msg.pose.orientation.w = 1
            sphere_msg.type = Marker.SPHERE
            sphere_msg.id = idx
            idx += 1
            msg.markers.append(sphere_msg)
        self.spheres_pub.publish(msg)

    def viz_from_spheres_dict(self, transforms, spheres):
        sphere_positions_root_frame = []
        radii = []
        for link_name, spheres_for_link in spheres.items():
            for sphere in spheres_for_link:
                pos_link_frame = sphere['position']
                r = sphere['radius']
                t = transforms[link_name].get_matrix().numpy()[0]
                pos_root_frame = (t @ homogeneous_np(pos_link_frame))[:3]
                sphere_positions_root_frame.append(pos_root_frame)
                radii.append(r)
        sphere_positions_root_frame = np.array(sphere_positions_root_frame)
        radii = np.array(radii)
        self.viz(sphere_positions_root_frame, radii)


class MujocoVisualizer:

    def __init__(self):
        self.default_pub = rospy.Publisher("mj_geoms", MarkerArray, queue_size=10)
        self.publishers = {
        }

    def viz(self, physics: Physics, alpha=1.0, ns: str = ''):
        geoms_marker_msg = MarkerArray()

        for geom_id in range(physics.model.ngeom):
            geom_name = mj_id2name(physics.model.ptr, mju_str2Type('geom'), geom_id)

            geom_bodyid = physics.model.geom_bodyid[geom_id]
            body_name = mj_id2name(physics.model.ptr, mju_str2Type('body'), geom_bodyid)

            geom_marker_msg = Marker()
            geom_marker_msg.action = Marker.ADD
            geom_marker_msg.header.frame_id = 'world'
            geom_marker_msg.ns = f'{body_name}-{geom_name}'
            geom_marker_msg.id = geom_id

            geom_type = physics.model.geom_type[geom_id]
            body_pos = physics.data.xpos[geom_bodyid]
            body_xmat = physics.data.xmat[geom_bodyid]
            body_xquat = np.zeros(4)
            mju_mat2Quat(body_xquat, body_xmat)
            geom_pos = physics.data.geom_xpos[geom_id]
            geom_xmat = physics.data.geom_xmat[geom_id]
            geom_xquat = np.zeros(4)
            mju_mat2Quat(geom_xquat, geom_xmat)
            geom_size = physics.model.geom_size[geom_id]
            geom_rgba = physics.model.geom_rgba[geom_id]
            geom_meshid = physics.model.geom_dataid[geom_id]

            geom_marker_msg.pose.position.x = geom_pos[0]
            geom_marker_msg.pose.position.y = geom_pos[1]
            geom_marker_msg.pose.position.z = geom_pos[2]
            geom_marker_msg.pose.orientation.w = geom_xquat[0]
            geom_marker_msg.pose.orientation.x = geom_xquat[1]
            geom_marker_msg.pose.orientation.y = geom_xquat[2]
            geom_marker_msg.pose.orientation.z = geom_xquat[3]
            geom_marker_msg.color.r = geom_rgba[0]
            geom_marker_msg.color.g = geom_rgba[1]
            geom_marker_msg.color.g = geom_rgba[2]
            geom_marker_msg.color.a = geom_rgba[3] * alpha

            if geom_type == mjtGeom.mjGEOM_BOX:
                geom_marker_msg.type = Marker.CUBE
                geom_marker_msg.scale.x = geom_size[0] * 2
                geom_marker_msg.scale.y = geom_size[1] * 2
                geom_marker_msg.scale.z = geom_size[2] * 2
            elif geom_type == mjtGeom.mjGEOM_CYLINDER:
                geom_marker_msg.type = Marker.CYLINDER
                geom_marker_msg.scale.x = geom_size[0] * 2
                geom_marker_msg.scale.y = geom_size[0] * 2
                geom_marker_msg.scale.z = geom_size[1] * 2
            elif geom_type == mjtGeom.mjGEOM_CAPSULE:
                # FIXME: not accurate, should use 2 spheres and a cylinder?
                geom_marker_msg.type = Marker.CYLINDER
                geom_marker_msg.scale.x = geom_size[0] * 2
                geom_marker_msg.scale.y = geom_size[0] * 2
                geom_marker_msg.scale.z = geom_size[1] * 2

                geom_marker_msg_ball1: Marker = deepcopy(geom_marker_msg)
                geom_marker_msg_ball1.ns = geom_marker_msg.ns + 'b1'
                geom_marker_msg_ball1.type = Marker.SPHERE
                geom_marker_msg_ball1.scale.x = geom_size[0] * 2
                geom_marker_msg_ball1.scale.y = geom_size[0] * 2
                geom_marker_msg_ball1.scale.z = geom_size[0] * 2
                ball1_pos_world = np.zeros(3)
                ball1_pos_local = np.array([0, 0, geom_size[1]])
                geom_xquat_neg = np.zeros(4)
                mujoco.mju_negQuat(geom_xquat_neg, geom_xquat)
                mujoco.mju_rotVecQuat(ball1_pos_world, ball1_pos_local, geom_xquat)
                geom_marker_msg_ball1.pose.position.x += ball1_pos_world[0]
                geom_marker_msg_ball1.pose.position.y += ball1_pos_world[1]
                geom_marker_msg_ball1.pose.position.z += ball1_pos_world[2]

                geom_marker_msg_ball2: Marker = deepcopy(geom_marker_msg)
                geom_marker_msg_ball2.ns = geom_marker_msg.ns + 'b2'
                geom_marker_msg_ball2.type = Marker.SPHERE
                geom_marker_msg_ball2.scale.x = geom_size[0] * 2
                geom_marker_msg_ball2.scale.y = geom_size[0] * 2
                geom_marker_msg_ball2.scale.z = geom_size[0] * 2
                ball2_pos_world = np.zeros(3)
                ball2_pos_local = np.array([0, 0, -geom_size[1]])
                geom_xquat_neg = np.zeros(4)
                mujoco.mju_negQuat(geom_xquat_neg, geom_xquat)
                mujoco.mju_rotVecQuat(ball2_pos_world, ball2_pos_local, geom_xquat)
                geom_marker_msg_ball2.pose.position.x += ball2_pos_world[0]
                geom_marker_msg_ball2.pose.position.y += ball2_pos_world[1]
                geom_marker_msg_ball2.pose.position.z += ball2_pos_world[2]

                geoms_marker_msg.markers.append(geom_marker_msg_ball1)
                geoms_marker_msg.markers.append(geom_marker_msg_ball2)
            elif geom_type == mjtGeom.mjGEOM_SPHERE:
                geom_marker_msg.type = Marker.SPHERE
                geom_marker_msg.scale.x = geom_size[0] * 2
                geom_marker_msg.scale.y = geom_size[0] * 2
                geom_marker_msg.scale.z = geom_size[0] * 2
            elif geom_type == mjtGeom.mjGEOM_MESH:
                mesh_name = mj_id2name(physics.model.ptr, mju_str2Type('mesh'), geom_meshid)
                mesh_name = mesh_name.split("/")[-1]  # skip the model prefix, e.g. val/my_mesh
                geom_marker_msg.type = Marker.MESH_RESOURCE
                geom_marker_msg.mesh_resource = f"package://dm_envs/meshes/{mesh_name}.stl"

                # We use body pos/quat here under the assumption that in the XML, the <geom type="mesh" ... />
                #  has NO POS OR QUAT, but instead that info goes in the <body> tag
                geom_marker_msg.pose.position.x = body_pos[0]
                geom_marker_msg.pose.position.y = body_pos[1]
                geom_marker_msg.pose.position.z = body_pos[2]
                geom_marker_msg.pose.orientation.w = body_xquat[0]
                geom_marker_msg.pose.orientation.x = body_xquat[1]
                geom_marker_msg.pose.orientation.y = body_xquat[2]
                geom_marker_msg.pose.orientation.z = body_xquat[3]

                geom_marker_msg.scale.x = 1
                geom_marker_msg.scale.y = 1
                geom_marker_msg.scale.z = 1
            else:
                rospy.loginfo_once(f"Unsupported geom type {geom_type}")
                continue

            geoms_marker_msg.markers.append(geom_marker_msg)

        pub = self.publishers.get(ns, self.default_pub)
        pub.publish(geoms_marker_msg)
        # print(f"viz took {perf_counter() - t0:0.3f}")


def visualize_vg(pub, vg, origin_point, res):
    origin_point_viz = origin_point - res / 2
    msg = VoxelgridStamped()
    msg.header.frame_id = 'world'
    msg.origin = msgify(Point, origin_point_viz.cpu().numpy())
    msg.scale = float(res.cpu().numpy())
    msg.occupancy = vox_to_float_array(vg)
    pub.publish(msg)

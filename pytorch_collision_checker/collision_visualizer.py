from typing import Dict

import rospy
from visualization_msgs.msg import MarkerArray, Marker


class CollisionVisualizer:

    def __init__(self):
        self.spheres_pub = rospy.Publisher('spheres', MarkerArray, queue_size=10)

    def display_joint_given_transforms(self, transforms: Dict):
        # FIXME: where do radii come from?
        #  how would this function be used?
        spheres = {}
        for link_name, transform in transforms.items():
            transform = transforms[link_name].get_matrix()
            pos = transform[0, :3, 3]
            spheres[link_name] = {
                'position': pos,
                'radius':   radii,
            }
        self.viz_from_spheres_dict(spheres)

    def display_joint_config(self, chain, joint_positions):
        # FIXME: where do radii come from?
        #  how would this function be used?
        pass

    def viz_from_spheres_dict(self, spheres):
        msg = MarkerArray()
        idx = 0
        for link_name, spheres_for_link in spheres.items():
            for sphere in spheres_for_link:
                pos = sphere['position']
                r = sphere['radius']
                sphere_msg = Marker()
                sphere_msg.scale.x = 2 * r
                sphere_msg.scale.y = 2 * r
                sphere_msg.scale.z = 2 * r
                sphere_msg.color.a = 0.7
                sphere_msg.color.r = 0.4
                sphere_msg.color.g = 2.4
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

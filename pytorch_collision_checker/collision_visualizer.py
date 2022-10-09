from typing import Dict

import rospy
from pytorch_kinematics import Chain
from visualization_msgs.msg import MarkerArray, Marker


class CollisionVisualizer:

    def __init__(self):
        self.spheres_pub = rospy.Publisher('spheres', MarkerArray, queue_size=10)
        self.geoms_pub = rospy.Publisher('geoms', MarkerArray, queue_size=10)

    def viz_robot(self, chain: Chain, transforms: Dict, ns: str = ''):
        for link_name, transform in transforms.items():
            transform = transforms[link_name].get_matrix()
            pos = transform[0, :3, 3]
            link = chain.find_link(link_name)
            link.visuals


        msg = MarkerArray()
        idx = 0
        for geom_name, geoms_for_geom in geoms.items():
            for geom in geoms_for_geom:
                pos = geom['position']
                r = geom['radius']
                geom_msg = Marker()
                geom_msg.ns = ns
                geom_msg.scale.x = 2 * r
                geom_msg.scale.y = 2 * r
                geom_msg.scale.z = 2 * r
                geom_msg.color.a = 0.7
                geom_msg.color.r = 0.4
                geom_msg.color.g = 2.4
                geom_msg.color.b = 0.4
                geom_msg.action = Marker.ADD
                geom_msg.header.frame_id = 'world'
                geom_msg.pose.position.x = pos[0]
                geom_msg.pose.position.y = pos[1]
                geom_msg.pose.position.z = pos[2]
                geom_msg.pose.orientation.w = 1
                geom_msg.type = Marker.SPHERE
                geom_msg.id = idx
                idx += 1
                msg.markers.append(geom_msg)
        self.geoms_pub.publish(msg)

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

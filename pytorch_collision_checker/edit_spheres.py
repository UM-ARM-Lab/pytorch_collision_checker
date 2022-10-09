import argparse
import json
import pathlib
import sys
from datetime import datetime
from typing import Dict

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QTreeWidgetItem
from dm_control import mujoco

import pytorch_kinematics as pk
import rospy
from geometry_msgs.msg import Pose
from pytorch_collision_checker.basic_3d_pose_marker import Basic3DPoseInteractiveMarker
from pytorch_collision_checker.collision_visualizer import CollisionVisualizer, MujocoVisualizer
from pytorch_collision_checker.sphere_editor import Ui_MainWindow
from pytorch_kinematics import Chain
from visualization_msgs.msg import InteractiveMarkerFeedback


class EditSphere:
    def __init__(self, model_filename: pathlib.Path, outfilename: pathlib.Path, data: Dict, chain: Chain):
        self.outfilename = outfilename
        self.data = data
        self.chain = chain
        self.transforms = self.chain.forward_kinematics(
            np.zeros(len(self.chain.get_joint_parameter_names(exclude_fixed=True))))

        self.cc_viz = CollisionVisualizer()
        self.mj_viz = MujocoVisualizer()
        self.physics = mujoco.Physics.from_xml_string(model_filename.open().read())
        self.i = Basic3DPoseInteractiveMarker(cb=self.im_cb, frame_id='world')

        if 'spheres' not in self.data:
            self.data['spheres'] = {}

        app = QtWidgets.QApplication(sys.argv)
        main_window = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(main_window)
        self.ui.add_button.clicked.connect(self.on_add)
        self.ui.remove_button.clicked.connect(self.on_remove)
        self.ui.spheres_tree.setHeaderLabels(['Link Name', 'Index', 'Position', 'Radius'])
        for link_name in self.chain.get_link_names():
            item = QTreeWidgetItem()
            item.setText(0, link_name)
            self.ui.spheres_tree.addTopLevelItem(item)
            if link_name not in self.data['spheres']:
                self.data['spheres'][link_name] = []
            for sphere_idx, sphere in enumerate(self.data['spheres'][link_name]):
                self.add_sphere_item_to_tree(item, sphere_idx, sphere)

        self.ui.spheres_tree.itemClicked.connect(self.on_item_clicked)
        self.ui.save_button.clicked.connect(self.on_save)
        self.ui.radius_spinbox.valueChanged.connect(self.on_radius_changed)

        self.publish_spheres()

        self.move_im_to_item(self.ui.spheres_tree.currentItem())
        self.set_radius_from_item(self.ui.spheres_tree.currentItem())

        main_window.show()
        app.exec_()
        self.on_save()

    def on_radius_changed(self, radius):
        item = self.ui.spheres_tree.currentItem()
        if item is None:
            print("Select a sphere in the tree view!")
            return
        if item.childCount() > 0:
            print("Select a sphere in the tree view!")
            return

        link_name = item.parent().text(0)
        sphere_idx = int(item.text(1))
        sphere = self.data['spheres'][link_name][sphere_idx]
        sphere['radius'] = radius

        self.publish_spheres()

        item.setText(3, str(radius))

    def on_item_clicked(self, item, _):
        self.move_im_to_item(item)
        self.set_radius_from_item(item)

    def set_radius_from_item(self, item):
        radius_str = item.text(3)
        if radius_str == '':
            return
        radius = float(radius_str)
        self.ui.radius_spinbox.setValue(radius)

    def move_im_to_item(self, item):
        xyz_str = item.text(2)
        if xyz_str == '':
            return

        pos = np.fromstring(xyz_str, dtype=np.float, sep=', ')
        pose = Pose()
        pose.position.x = pos[0]
        pose.position.y = pos[1]
        pose.position.z = pos[2]
        self.i.set_pose(pose)

    def add_sphere_item_to_tree(self, parent_item, i, sphere):
        child_item = QTreeWidgetItem()
        child_item.setText(1, str(i))
        pos = sphere['position']
        self.set_position_text(child_item, pos)
        child_item.setText(3, f"{sphere['radius']}")
        parent_item.addChild(child_item)
        self.ui.spheres_tree.setCurrentItem(child_item)
        return child_item

    def set_position_text(self, child_item, pos):
        child_item.setText(2, f"{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}")

    def im_cb(self, feedback: InteractiveMarkerFeedback):
        item = self.ui.spheres_tree.currentItem()
        if item is None:
            print("Select a sphere in the tree view!")
            return
        if item.childCount() > 0:
            print("Select a sphere in the tree view!")
            return
        link_name = item.parent().text(0)
        sphere_idx = int(item.text(1))
        pos = self.data['spheres'][link_name][sphere_idx]['position']
        x = feedback.pose.position.x
        y = feedback.pose.position.y
        z = feedback.pose.position.z
        pos[0] = x
        pos[1] = y
        pos[2] = z
        self.publish_spheres()

        self.set_position_text(item, pos)

    def on_save(self):
        with self.outfilename.open("w") as f:
            json.dump(self.data, f)

    def on_remove(self):
        item = self.ui.spheres_tree.currentItem()
        if item is None:
            print("Select a sphere in the tree view!")
            return
        if item.childCount() > 0:
            print("Select a sphere in the tree view!")
            return
        link_name = item.parent().text(0)
        sphere_idx = int(item.text(1))
        self.data['spheres'][link_name].pop(sphere_idx - 1)
        # FIXME:
        # self.ui.spheres_tree.removeItemWidget(item, 1)
        # self.ui.spheres_tree.removeItemWidget(item, 2)
        # self.ui.spheres_tree.removeItemWidget(item, 3)

    def on_add(self):
        parent_item = self.ui.spheres_tree.currentItem()
        if parent_item.parent() is not None:
            parent_item = parent_item.parent()

        sphere_idx = parent_item.childCount()
        link_name = parent_item.text(0)

        pos, radius = self.copy_from_current_item(link_name)

        # move the IM to match the newly created sphere
        pose = Pose()
        pose.position.x = pos[0]
        pose.position.y = pos[1]
        pose.position.z = pos[2]
        self.i.set_pose(pose)

        sphere = {
            'position': pos,
            'radius':   radius,
        }
        self.add_sphere_item_to_tree(parent_item, sphere_idx, sphere)
        self.data['spheres'][link_name].append(sphere)

        self.publish_spheres()

    def copy_from_current_item(self, link_name):
        x, y, z = self.get_link_xyz(link_name)
        item = self.ui.spheres_tree.currentItem()
        if item is None:
            return [0, 0, 0], 0.1
        elif item.parent() is None:
            return [x, y, z], 0.1
        else:
            xyz_str = item.text(2)
            pos = np.fromstring(xyz_str, dtype=np.float, sep=', ').tolist()
            radius = float(item.text(3))
            return pos, radius

    def get_link_xyz(self, link_name):
        transform = self.transforms[link_name].get_matrix()
        x = float(transform[0, 0, 3])
        y = float(transform[0, 1, 3])
        z = float(transform[0, 2, 3])
        return x, y, z

    def publish_spheres(self):
        self.cc_viz.viz_from_spheres_dict(self.data['spheres'])
        self.mj_viz.viz(self.physics)


def main():
    rospy.init_node('edit_spheres')
    parser = argparse.ArgumentParser()
    parser.add_argument('model_filename', type=pathlib.Path)

    args = parser.parse_args()

    outfilename = args.model_filename.parent / (args.model_filename.stem + "_spheres.json")

    if outfilename.exists():
        with outfilename.open("r") as f:
            data = json.load(f)
    else:
        data = {}

    now = datetime.now()  # current date and time
    data['created'] = now.strftime("%m/%d/%Y, %H:%M:%S")
    data['model filename'] = args.model_filename.absolute().as_posix()

    chain = pk.build_chain_from_mjcf(args.model_filename.open().read())
    EditSphere(args.model_filename, outfilename, data, chain)


if __name__ == '__main__':
    main()

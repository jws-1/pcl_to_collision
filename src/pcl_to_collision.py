#!/usr/bin/env python3
import ros_numpy as rnp
import numpy as np

from moveit_msgs.msg import CollisionObject, AttachedCollisionObject
from shape_msgs.msg import SolidPrimitive

from sensor_msgs import PointCloud2
from geometry_msgs.msg import Pose, Point, Quaternion
import moveit_commander
import sys
import rospy

from lasr_vision_msgs import YoloDetection3D


def add_collision_object(id, pointcloud, planning_scene, num_primitives=200):

    pcl = rnp.numpify(pointcloud)
    cloud_obj = np.concatenate(
        (pcl["x"].reshape(-1, 1), pcl["y"].reshape(-1, 1), pcl["z"].reshape(-1, 1)),
        axis=1,
    )

    co = CollisionObject()
    co.id = id

    indices = np.random.choice(cloud_obj.shape[0], num_primitives, replace=False)

    for idx in indices:
        primitive = SolidPrimitive()
        primitive.type = primitive.BOX
        primitive.dimensions = [0.07, 0.07, 0.07]

        co.primitives.append(primitive)
        co.primitive_poses.append(
            Pose(position=Point(*cloud_obj[idx])), orientation=Quaternion(0, 0, 0, 1)
        )

    planning_scene.add_object(co)
    return co


if __name__ == "__main__":
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("pcl_to_collision")

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()

    yolo = rospy.ServiceProxy("/detect_3d", YoloDetection3D)
    pcl_msg = rospy.wait_for_message("/xtion/depth_registered/points", PointCloud2)
    response = yolo(pcl_msg, "yolov8n-seg.pt", 0.3, 0.3)
    pcl_cropped = response.detections[0].cloud_seg

    # Assuming you have a pointcloud in the variable `pcl`
    add_collision_object("pcl_collision", pcl_cropped, scene)

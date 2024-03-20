#!/usr/bin/env python3
import ros_numpy as rnp
import numpy as np

from moveit_msgs.msg import CollisionObject, AttachedCollisionObject
from shape_msgs.msg import SolidPrimitive

from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Pose, Point, Quaternion, PoseArray
import moveit_commander
import sys
import rospy
import cv2
from lasr_vision_msgs.srv import YoloDetection3D


def add_collision_object(
    id, pcl, indices, planning_scene, method="exhaustive", frame_id="map"
):

    # # pcl = rnp.numpify(pointcloud)
    # cloud_obj = np.concatenate(
    #     (pcl["x"].reshape(-1, 1), pcl["y"].reshape(-1, 1), pcl["z"].reshape(-1, 1)),
    #     axis=1,
    # )

    # Unpack pointcloud into xyz array
    pcl_xyz = rnp.point_cloud2.pointcloud2_to_xyz_array(pcl, remove_nans=False)
    pcl_xyz = pcl_xyz.reshape(pcl.height, pcl.width, 3)

    # Extract points of interest
    co = CollisionObject()
    co.id = id

    # if indices.shape[0] > num_primitives:
    # indices = indices[np.random.choice(len(indices), num_primitives, replace=False)]
    xyz_points = [
        pcl_xyz[x][y] for x, y in indices if not np.isnan(pcl_xyz[x][y]).any()
    ]

    # filter outliers, using confidence interval
    xyz_points = np.array(xyz_points)
    mean = np.nanmean(xyz_points, axis=0)
    std = np.std(xyz_points, axis=0)
    xyz_points = xyz_points[
        np.all(np.abs((xyz_points - mean) / std) < 2, axis=1)
    ]  # 2 std from mean

    if method == "exhaustive":
        primitive = SolidPrimitive()
        primitive.type = primitive.SPHERE
        primitive.dimensions = [0.01, 0.01, 0.01]

        pose_array = PoseArray()

        for x, y, z in xyz_points:
            if np.isnan(x) or np.isnan(y) or np.isnan(z):
                continue
            primitive_pose = Pose(
                position=Point(x, y, z), orientation=Quaternion(0, 0, 0, 1)
            )
            co.primitives.append(primitive)
            pose_array.poses.append(primitive_pose)

        co.primitive_poses = pose_array.poses

    elif method == "bounding_cuboid":
        primitive = SolidPrimitive()
        primitive.type = primitive.BOX
        # compute bounding box from indices
        min_x = np.nanmin(xyz_points, axis=0)[0]
        max_x = np.nanmax(xyz_points, axis=0)[0]
        min_y = np.nanmin(xyz_points, axis=0)[1]
        max_y = np.nanmax(xyz_points, axis=0)[1]
        min_z = np.nanmin(xyz_points, axis=0)[2]
        max_z = np.nanmax(xyz_points, axis=0)[2]

        primitive.dimensions = [
            max_x - min_x,
            max_y - min_y,
            max_z - min_z,
        ]
        primitive_pose = Pose(
            position=Point(min_x, min_y, min_z),
            orientation=Quaternion(0, 0, 0, 1),
        )
        co.primitives.append(primitive)
        co.primitive_poses.append(primitive_pose)
    co.header = pcl.header
    # co.header.frame_id = "base_footprint"
    # co.header.frame_id = frame_id

    rospy.loginfo("Added to planning scene!!")
    rospy.loginfo(co)
    planning_scene.add_object(co)
    return co


if __name__ == "__main__":
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("pcl_to_collision")

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()

    yolo = rospy.ServiceProxy("/yolov8/detect3d", YoloDetection3D)
    pcl_msg = rospy.wait_for_message("/xtion/depth_registered/points", PointCloud2)
    response = yolo(pcl_msg, "yolov8n-seg.pt", 0.1, 0.1)

    pcl_segment_pub = rospy.Publisher(
        "/pcl_segment", PointCloud2, queue_size=10, latch=True
    )

    pcl = np.frombuffer(pcl_msg.data, dtype=np.uint8)
    pcl = pcl.reshape(pcl_msg.height, pcl_msg.width, -1)

    frame = np.frombuffer(pcl_msg.data, dtype=np.uint8)
    frame = frame.reshape(pcl_msg.height, pcl_msg.width, 32)
    frame = frame[:, :, 16:19].copy()
    frame = np.ascontiguousarray(frame, dtype=np.uint8)

    detection = next(
        filter(
            lambda d: d.name in ["bowl", "cup", "bottle", "vase", "toothbrush"],
            response.detected_objects,
        ),
        None,
    )

    assert detection is not None, "No bowl detected"

    # Compute mask from contours
    contours = np.array(detection.xyseg).reshape(-1, 2)
    mask = np.zeros(shape=frame.shape[:2])
    cv2.fillPoly(mask, pts=[contours], color=(255, 255, 255))

    # Extract mask indices from bounding box
    indices = np.argwhere(mask)

    # pcl_cropped_np = np.take(
    #     pcl.reshape(pcl.shape[0] * pcl.shape[1], -1), indices, axis=0
    # )

    # pcl_cropped = PointCloud2()
    # pcl_cropped.header = pcl_msg.header
    # pcl_cropped.header.frame_id = "base_footprint"
    # pcl_cropped.height = pcl_cropped_np.shape[0]
    # pcl_cropped.width = 1
    # pcl_cropped.fields = pcl_msg.fields
    # pcl_cropped.is_bigendian = pcl_msg.is_bigendian
    # pcl_cropped.point_step = pcl_msg.point_step
    # pcl_cropped.row_step = pcl_cropped_np.shape[1]
    # pcl_cropped.is_dense = pcl_msg.is_dense
    # pcl_cropped.data = pcl_cropped_np.flatten().tobytes()

    # pcl_segment_pub.publish(pcl_cropped)
    add_collision_object("pcl_collision", pcl_msg, indices, scene, method="exhaustive")
    rospy.spin()

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Vector3

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

from ultralytics import YOLO
from dis_tutorial3.msg import Coordinates

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker

import tf2_geometry_msgs as tfg
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy


# from rclpy.parameter import Parameter
# from rcl_interfaces.msg import SetParametersResult

class detect_faces(Node):

	def __init__(self):
		super().__init__('detect_faces')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
		])

		marker_topic = "/people_marker"
		coordinates_topic = "/coordinates"

		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

		self.marker_id = 0

		self.detection_color = (0,0,255)
		self.device = self.get_parameter('device').get_parameter_value().string_value

		self.bridge = CvBridge()
		self.scan = None

		self.rgb_image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.rgb_callback, qos_profile_sensor_data)
		self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data)

		self.marker_pub = self.create_publisher(Marker, marker_topic, QoSReliabilityPolicy.BEST_EFFORT)
		self.coordinates_pub = self.create_publisher(Coordinates, coordinates_topic, QoSReliabilityPolicy.BEST_EFFORT)

		self.model = YOLO("yolov8n.pt")

		self.faces = []

		self.get_logger().info(f"Node has been initialized! Will publish face markers to {marker_topic}.")

	def rgb_callback(self, data):

		self.faces = []

		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

			self.get_logger().info(f"Running inference on image...")

			# run inference
			res = self.model.predict(cv_image, imgsz=(256, 320), show=False, verbose=False, classes=[0], device=self.device)

			# iterate over results
			for x in res:
				bbox = x.boxes.xyxy
				if bbox.nelement() == 0: # skip if empty
					continue

				self.get_logger().info(f"Person has been detected!")

				bbox = bbox[0]

				# draw rectangle
				cv_image = cv2.rectangle(cv_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), self.detection_color, 3)

				cx = int((bbox[0]+bbox[2])/2)
				cy = int((bbox[1]+bbox[3])/2)

				# draw the center of bounding box
				cv_image = cv2.circle(cv_image, (cx,cy), 5, self.detection_color, -1)

				self.faces.append((cx,cy))

				# coords = Coordinates()
				# coords.x = cx
				# coords.y = cy

				# self.get_logger().info(f"Coordinates are: {coords.x}, {coords.y}")

				# self.coordinates_pub.publish(coords)

				# self.get_logger().info(f"{listOfCordinates}")

				# all_nodes = [el for el in listOfCordinates if el[2] == True]

				# current_node = None
				# if len(all_nodes) > 0:
				# 	all_nodes[0]

				# if current_node != None:

				# 	self.get_logger().info(f"Current node: {current_node}")

				# 	v1 = np.array([bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1], bbox[1][1] - bbox[0][1]])

				# 	v2 = np.array([bbox[2][0] - bbox[0][0], bbox[2][1] - bbox[0][1], bbox[2][1] - bbox[0][1]])

				# 	normala = np.cross(v1, v2)
				# 	self.get_logger().info(f"normala: {normala}")



				#marker = Marker()
				#marker.header().frame_id = "base_link"
				#marker.type = Marker.ARROW
				#marker.action = Marker.ADD
				#marker.scale = Vector3(0.1, 0.2, 0.2)

				#marker.color.a = 1.0
				#marker.color.r = 1.0
				#marker.pose.orientation.w = 1.0
				#marker.points.append(Point(cx, cy, 0.143))
				#marker.points.append(Points(cx + dx))



			cv2.imshow("image", cv_image)
			key = cv2.waitKey(1)
			if key==27:
				print("exiting")
				exit()
			
		except CvBridgeError as e:
			print(e)

	def pointcloud_callback(self, data):

		# get point cloud attributes
		height = data.height
		width = data.width
		point_step = data.point_step
		row_step = data.row_step		

		# iterate over face coordinates
		for x,y in self.faces:

			# get 3-channel representation of the poitn cloud in numpy format
			a = pc2.read_points_numpy(data, field_names= ("x", "y", "z"))
			a = a.reshape((height,width,3))

			# read center coordinates
			d = a[y,x,:]

			# create marker
			marker = Marker()

			marker.header.frame_id = "/base_link"
			marker.header.stamp = data.header.stamp

			marker.type = 2
			marker.id = 0

			# Set the scale of the marker
			scale = 0.1
			marker.scale.x = scale
			marker.scale.y = scale
			marker.scale.z = scale

			# Set the color
			marker.color.r = 1.0
			marker.color.g = 1.0
			marker.color.b = 1.0
			marker.color.a = 1.0

			# Set the pose of the marker
			marker.pose.position.x = float(d[0])
			marker.pose.position.y = float(d[1])
			marker.pose.position.z = float(d[2])

			self.marker_pub.publish(marker)

			point_in_robot_frame = PointStamped()
			point_in_robot_frame.header.frame_id = "/base_link"
			point_in_robot_frame.header.stamp = self.get_clock().now().to_msg()

			point_in_robot_frame.point.x = float(d[0])
			point_in_robot_frame.point.y = float(d[1])
			point_in_robot_frame.point.z = float(d[2])

			# Now we look up the transform between the base_link and the map frames
			# and then we apply it to our PointStamped
			time_now = rclpy.time.Time()
			timeout = Duration(seconds=0.1)
			try:
				# An example of how you can get a transform from /base_link frame to the /map frame
				# as it is at time_now, wait for timeout for it to become available
				trans = self.tf_buffer.lookup_transform("map", "base_link", time_now, timeout)
				self.get_logger().info(f"Looks like the transform is available.")

				# Now we apply the transform to transform the point_in_robot_frame to the map frame
				# The header in the result will be copied from the Header of the transform
				point_in_map_frame = tfg.do_transform_point(point_in_robot_frame, trans)
				self.get_logger().info(f"We transformed a PointStamped: {point_in_map_frame.point.x}, {point_in_map_frame.point.y}")
				# self.get_logger().info(f"{point_in_map_frame}")

				# If the transformation exists, create a marker from the point, in order to visualize it in Rviz
				marker_in_map_frame = self.create_marker(point_in_map_frame, self.marker_id)

				# Publish the marker
				self.marker_pub.publish(marker_in_map_frame)
				self.get_logger().info(f"The marker has been published to /breadcrumbs. You are able to visualize it in Rviz")

				# Increase the marker_id, so we dont overwrite the same marker.
				self.marker_id += 1

				coords = Coordinates()
				coords.x = point_in_map_frame.point.x
				coords.y = point_in_map_frame.point.y

				# self.get_logger().info(f"Coordinates are: {coords.x}, {coords.y}")

				self.coordinates_pub.publish(coords)

			except TransformException as te:
				self.get_logger().info(f"Cound not get the transform: {te}")

			# transform_points = TransformPoints()
	
	def create_marker(self, point_stamped, marker_id):
		"""You can see the description of the Marker message here: https://docs.ros2.org/galactic/api/visualization_msgs/msg/Marker.html"""
		marker = Marker()
		
		marker.header = point_stamped.header
		marker.type = marker.CUBE
		marker.action = marker.ADD
		marker.id = marker_id
		# Set the scale of the marker
		scale = 0.15
		marker.scale.x = scale
		marker.scale.y = scale
		marker.scale.z = scale
		# Set the color
		marker.color.r = 1.0
		marker.color.g = 0.0
		marker.color.b = 0.0
		marker.color.a = 1.0
		# Set the pose of the marker
		marker.pose.position.x = float(point_stamped.point.x)
		marker.pose.position.y = float(point_stamped.point.y)
		marker.pose.position.z = float(point_stamped.point.z)
		
		return marker

def main():
	print('Face detection node starting.')

	rclpy.init(args=None)
	node = detect_faces()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
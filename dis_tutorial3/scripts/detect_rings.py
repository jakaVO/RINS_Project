#!/usr/bin/python3

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

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import tf2_ros

from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3, Pose
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2


from dis_tutorial3.msg import Coordinates

qos_profile = QoSProfile(
		  durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
		  reliability=QoSReliabilityPolicy.RELIABLE,
		  history=QoSHistoryPolicy.KEEP_LAST,
		  depth=1)

class RingDetector(Node):
	def __init__(self):
		super().__init__('transform_point')

		# Basic ROS stuff
		timer_frequency = 2
		timer_period = 1/timer_frequency

		ring_topic = "/ring"

		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)
		
		# An object we use for converting images between ROS format and OpenCV format
		self.bridge = CvBridge()
		
		# Marker array object used for visualizations
		self.marker_array = MarkerArray()
		self.marker_num = 1
		
		# Dodana globalna spremenljivka za depth image
		self.depth_image = None
		
		# Subscribe to the image and/or depth topic
		self.image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.image_callback, 1)
		self.depth_sub = self.create_subscription(Image, "/oakd/rgb/preview/depth", self.depth_callback, 1)
		
		self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data)
		
		self.ring_publisher = self.create_publisher(Coordinates, ring_topic, QoSReliabilityPolicy.BEST_EFFORT)
		
		self.center_x = None
		self.center_y = None
		
		self.correct_ring_found = False
		
		cv2.namedWindow("Binary Image", cv2.WINDOW_NORMAL)
		cv2.namedWindow("Detected contours", cv2.WINDOW_NORMAL)
		cv2.namedWindow("Detected rings", cv2.WINDOW_NORMAL)
		cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)  

	def pointcloud_callback(self, data):

		if self.correct_ring_found:

			self.get_logger().info(f"Koordinate kroga1: ")
			self.get_logger().info("(" + str(self.center_y) + ", " + str(self.center_x) + ")")

			# get point cloud attributes
			height = data.height
			width = data.width
			
			# get 3-channel representation of the poitn cloud in numpy format
			a = pc2.read_points_numpy(data, field_names= ("x", "y", "z"), skip_nans=True)
			a = a.reshape((height,width,3))

			print(self.y_min, self.y_max, self.x_min, self.x_max)
			print(height)
			print(width)

			point = None
			for y in range(self.y_min, self.y_max):
				for x in range(self.x_min, self.x_max):
					if not np.any(np.isnan(a[x, y, :])) and not np.any(np.isinf(a[x, y, :])):
						point = a[x, y, :]
						print("Point:")
						print(point)
						break

			# x = self.center_x
			# y = self.center_y

			# actions = ["add_x", "add_y", "sub_x", "sub_y"]
			# incr_by = 

			# while point is None:
			# 	if not np.any(np.isnan(a[y, x, :])) and not np.any(np.isinf(a[y, x, :])):
			# 		point = a[y, x, :]
			# 		print("Point:")
			#  		print(point)
			# 	else:


			# point = a[self.y_max, self.center_x, :]

			print("Point:")
			print(point)
			
			# read center coordinates
			# d = a[self.center_y, self.center_x, :]

			self.get_logger().info(f"Koordinate kroga2: ")
			self.get_logger().info("(" + str(self.center_y) + ", " + str(self.center_x) + ")")

			point_in_robot_frame = PointStamped()
			point_in_robot_frame.header.frame_id = data.header.frame_id
			point_in_robot_frame.header.stamp = rclpy.time.Time().to_msg()
			point_in_robot_frame.point.x = float(point[0])
			point_in_robot_frame.point.y = float(point[1])
			point_in_robot_frame.point.z = float(point[2])

			self.get_logger().info("d: (" + str(point_in_robot_frame.point.x) + ", " + str(point_in_robot_frame.point.y) + ")")

			time_now = rclpy.time.Time()
			timeout = Duration(seconds=0.1)

			try:
				trans = self.tf_buffer.lookup_transform("map", "base_link", time_now, timeout)
				self.get_logger().info(f"Looks like the transform is available.")

				point_in_map_frame = tfg.do_transform_point(point_in_robot_frame, trans)
				self.get_logger().info(f"We transformed a PointStamped: {point_in_map_frame.point.x}, {point_in_map_frame.point.y}, {point_in_map_frame.point.z}")

				coords = Coordinates()
				coords.x = point_in_map_frame.point.x
				coords.y = point_in_map_frame.point.y
				# self.get_logger().info(f"Coordinates are: {coords.x}, {coords.y}")
				
				self.ring_publisher.publish(coords)
			except TransformException as te:
				self.get_logger().info("exceptionnnnnnnnnn!!!!!")
				
			self.correct_ring_found = False


	def image_callback(self, data):
		# self.get_logger().info(f"I got a new image! Will try to find rings...")

		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

		blue = cv_image[:,:,0]
		green = cv_image[:,:,1]
		red = cv_image[:,:,2]

		# Tranform image to grayscale
		gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
		# gray = red

		# Apply Gaussian Blur
		# gray = cv2.GaussianBlur(gray,(3,3),0)

		# Do histogram equalization
		# gray = cv2.equalizeHist(gray)

		# Binarize the image, there are different ways to do it
		#ret, thresh = cv2.threshold(img, 50, 255, 0)
		#ret, thresh = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
		thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 30)
		cv2.imshow("Binary Image", thresh)
		cv2.waitKey(1)

		# Extract contours
		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

		# Example of how to draw the contours, only for visualization purposes
		cv2.drawContours(gray, contours, -1, (255, 0, 0), 3)
		cv2.imshow("Detected contours", gray)
		cv2.waitKey(1)

		# Fit elipses to all extracted contours
		elps = []
		for cnt in contours:
			#     print cnt
			#     print cnt.shape
			if cnt.shape[0] >= 20:
				ellipse = cv2.fitEllipse(cnt)
				elps.append(ellipse)


		# Find two elipses with same centers
		candidates = []
		for n in range(len(elps)):
			for m in range(n + 1, len(elps)):
				# e[0] is the center of the ellipse (x,y), e[1] are the lengths of major and minor axis (major, minor), e[2] is the rotation in degrees
				
				e1 = elps[n]
				e2 = elps[m]
				dist = np.sqrt(((e1[0][0] - e2[0][0]) ** 2 + (e1[0][1] - e2[0][1]) ** 2))
				angle_diff = np.abs(e1[2] - e2[2])

				# The centers of the two elipses should be within 5 pixels of each other (is there a better treshold?)
				if dist >= 5:
					continue

				# The rotation of the elipses should be whitin 4 degrees of eachother
				if angle_diff>4:
					continue

				e1_minor_axis = e1[1][0]
				e1_major_axis = e1[1][1]

				e2_minor_axis = e2[1][0]
				e2_major_axis = e2[1][1]

				if e1_major_axis>=e2_major_axis and e1_minor_axis>=e2_minor_axis: # the larger ellipse should have both axis larger
					le = e1 # e1 is larger ellipse
					se = e2 # e2 is smaller ellipse
				elif e2_major_axis>=e1_major_axis and e2_minor_axis>=e1_minor_axis:
					le = e2 # e2 is larger ellipse
					se = e1 # e1 is smaller ellipse
				else:
					continue # if one ellipse does not contain the other, it is not a ring
				
				# # The widths of the ring along thcreate_subscriptione major and minor axis should be roughly the same
				# border_major = (le[1][1]-se[1][1])/2
				# border_minor = (le[1][0]-se[1][0])/2
				# border_diff = np.abs(border_major - border_minor)

				# if border_diff>4:
				#     continue
					
				candidates.append((e1,e2))

		# print("Processing is done! found", len(candidates), "candidates for rings")

		# Plot the rings on the image
		for c in candidates:

			# the centers of the ellipses
			e1 = c[0]
			e2 = c[1]

			# drawing the ellipses on the image
			cv2.ellipse(cv_image, e1, (0, 255, 0), 2)
			cv2.ellipse(cv_image, e2, (0, 255, 0), 2)

			# Get a bounding box, around the first ellipse ('average' of both elipsis)
			size = (e1[1][0]+e1[1][1])/2
			center = (e1[0][1], e1[0][0])

			x1 = int(center[0] - size / 2)
			x2 = int(center[0] + size / 2)
			x_min = x1 if x1>0 else 0
			x_max = x2 if x2<cv_image.shape[0] else cv_image.shape[0]

			y1 = int(center[1] - size / 2)
			y2 = int(center[1] + size / 2)
			y_min = y1 if y1 > 0 else 0
			y_max = y2 if y2 < cv_image.shape[1] else cv_image.shape[1]

			center_x = int(round(e1[0][0]))  # Round and convert the x-coordinate to integer
			center_y = int(round(e1[0][1]))  # Round and convert the y-coordinate to integer

			if self.depth_image[center_y, center_x] == 0:
				self.get_logger().info(f"najdu sm taprav inf krog AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA...")
				coords = Coordinates()
				coords.x = float(center_x)
				coords.y = float(center_y)

				self.correct_ring_found = True

				# self.get_logger().info(data)

				# read center coordinates
				# d = a[coords.y,coords.x,:]

				# self.get_logger().info(d)

				self.center_x = center_x
				self.center_y = center_y
				self.y_min = y_min
				self.y_max = y_max
				self.x_min = x_min
				self.x_max = x_max


				# self.ring_publisher.publish(coords)
			
			if len(candidates)>0:
				cv2.imshow("Detected rings",cv_image)
				cv2.waitKey(1)

	def depth_callback(self,data):

		try:
			self.depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
			
		except CvBridgeError as e:
			print(e)

		self.depth_image[self.depth_image==np.inf] = 0
		# self.get_logger().info(f"depth_image={self.depth_image}-------...")
		# Do the necessairy conversion so we can visuzalize it in OpenCV
		image_1 = self.depth_image / 65536.0 * 255
		image_1 = image_1/np.max(image_1)*255

		image_viz = np.array(image_1, dtype=np.uint8)

		cv2.imshow("Depth window", image_viz)
		cv2.waitKey(1)


def main():

	rclpy.init(args=None)
	rd_node = RingDetector()

	rclpy.spin(rd_node)

	cv2.destroyAllWindows()


if __name__ == '__main__':
	main() 

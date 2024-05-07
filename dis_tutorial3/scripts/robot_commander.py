#! /usr/bin/env python3
# Mofidied from Samsung Research America
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from enum import Enum
import time

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Quaternion, PoseStamped, PoseWithCovarianceStamped
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import Spin, NavigateToPose
from turtle_tf2_py.turtle_tf2_broadcaster import quaternion_from_euler

from irobot_create_msgs.action import Dock, Undock
from irobot_create_msgs.msg import DockStatus

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration as rclpyDuration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from dis_tutorial3.msg import Coordinates
from map_goals import MapGoals
# from transform_point import TransformPoints
from nav_msgs.msg import OccupancyGrid

import numpy as np
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
import math

from visualization_msgs.msg import Marker
from playsound import playsound
import pyttsx3
from pydub import AudioSegment
from pydub.playback import play

import pygame
from gtts import gTTS
from tempfile import TemporaryFile

from std_msgs.msg import String

import tf_transformations

qos_profile = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)

class TaskResult(Enum):
    UNKNOWN = 0
    SUCCEEDED = 1
    CANCELED = 2
    FAILED = 3

amcl_pose_qos = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)

listOfCordinates = [
        [-1.43,-0.6, False],
        [-0.105,-1.78, False],
        [1.11,-2.0, False],
        [3.09,-1.0, False],
        [1.05,-0.00697, False],
        [2.25,1.95, False],
        [-1.53,1.35, False],
        [-1.51,4.55, False],
        [-0.432,3.2, False],
        [1.53,3.25, False],
    ]

class RobotCommander(Node):

    def __init__(self, node_name='robot_commander', namespace=''):
        super().__init__(node_name=node_name, namespace=namespace)
        
        self.pose_frame_id = 'map'
        
        # Flags and helper variables
        self.goal_handle = None
        self.result_future = None
        self.feedback = None
        self.status = None
        self.initial_pose_received = False
        self.is_docked = None
        #self.map_goals = MapGoals()

        self.map_data = {"map_load_time":None,
                         "resolution":None,
                         "width":None,
                         "height":None,
                         "origin":None} # origin will be in the format [x,y,theta]

        self.faces_coords_list = []
        self.already_visited = []

        time.sleep(3)

        # ROS2 subscribers
        self.occupancy_grid_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, qos_profile)

        self.arm_publisher = self.create_publisher(String, "/arm_command", QoSReliabilityPolicy.BEST_EFFORT)

        self.create_subscription(DockStatus,
                                 'dock_status',
                                 self._dockCallback,
                                 qos_profile_sensor_data)
        
        # self.create_subscription(Coordinates,
        #                          'coordinates',
        #                          self._coordinatesCallback,
        #                          qos_profile_sensor_data)

        self.create_subscription(Coordinates,
                                 'ring',
                                 self._ringCallback,
                                 qos_profile_sensor_data)

        # self.create_subscription(Marker,
        #                          'marker',
        #                          self._markerCallback,
        #                          qos_profile_sensor_data)

        self.localization_pose_sub = self.create_subscription(PoseWithCovarianceStamped,
                                                              'amcl_pose',
                                                              self._amclPoseCallback,
                                                              amcl_pose_qos)
        
        # ROS2 publishers
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped,
                                                      'initialpose',
                                                      10)
        
        # ROS2 Action clients
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.spin_client = ActionClient(self, Spin, 'spin')
        self.undock_action_client = ActionClient(self, Undock, 'undock')
        self.dock_action_client = ActionClient(self, Dock, 'dock')

        self.get_logger().info(f"Robot commander has been initialized!")

    def map_callback(self, msg):
        self.get_logger().info(f"Read a new Map (Occupancy grid) from the topic.")
        # reshape the message vector back into a map
        self.map_np = np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        # fix the direction of Y (origin at top for OpenCV, origin at bottom for ROS2)
        self.map_np = np.flipud(self.map_np)
        # change the colors so they match with the .pgm image
        self.map_np[self.map_np==0] = 127
        self.map_np[self.map_np==100] = 0
        # load the map parameters
        self.map_data["map_load_time"]=msg.info.map_load_time
        self.map_data["resolution"]=msg.info.resolution
        self.map_data["width"]=msg.info.width
        self.map_data["height"]=msg.info.height
        quat_list = [msg.info.origin.orientation.x,
                     msg.info.origin.orientation.y,
                     msg.info.origin.orientation.z,
                     msg.info.origin.orientation.w]
        self.map_data["origin"]=[msg.info.origin.position.x,
                                 msg.info.origin.position.y,
                                 tf_transformations.euler_from_quaternion(quat_list)[-1]]

    def map_pixel_to_world(self, x, y, theta=0):
        ### Convert a pixel in an numpy image, to a real world location
        ### Works only for theta=0
        assert not self.map_data["resolution"] is None

        # Apply resolution, change of origin, and translation
        # 
        world_x = x*self.map_data["resolution"] + self.map_data["origin"][0]
        world_y = (self.map_data["height"]-y)*self.map_data["resolution"] + self.map_data["origin"][1]

        # Apply rotation
        return world_x, world_y
        
    def destroyNode(self):
        self.nav_to_pose_client.destroy()
        super().destroy_node()     

    def say_hello(self):
        text = "What is up"
        textToSpeech = gTTS(text=text, lang='en')
        temp_file = TemporaryFile()
        textToSpeech.write_to_fp(temp_file)
        temp_file.seek(0)
        pygame.mixer.init()
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(1)

        temp_file.close()

    def goToPose(self, pose, behavior_tree=''):
        """Send a `NavToPose` action request."""
        self.debug("Waiting for 'NavigateToPose' action server")
        while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.info("'NavigateToPose' action server not available, waiting...")

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        goal_msg.behavior_tree = behavior_tree

        self.info('Navigating to goal: ' + str(pose.pose.position.x) + ' ' +
                  str(pose.pose.position.y) + '...')
        # time.sleep(3)
        send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg,
                                                                   self._feedbackCallback)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.error('Goal to ' + str(pose.pose.position.x) + ' ' +
                       str(pose.pose.position.y) + ' was rejected!')
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True

    def spin(self, spin_dist=1.57, time_allowance=10):
        self.debug("Waiting for 'Spin' action server")
        while not self.spin_client.wait_for_server(timeout_sec=1.0):
            self.info("'Spin' action server not available, waiting...")
        goal_msg = Spin.Goal()
        goal_msg.target_yaw = spin_dist
        goal_msg.time_allowance = Duration(sec=time_allowance)
        self.info(f'Spinning to angle {goal_msg.target_yaw}....')
        send_goal_future = self.spin_client.send_goal_async(goal_msg, self._feedbackCallback)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.error('Spin request was rejected!')
            return False
        self.result_future = self.goal_handle.get_result_async()
        return True
    
    def undock(self):
        """Perform Undock action."""
        self.info('Undocking...')
        self.undock_send_goal()

        while not self.isUndockComplete():
            time.sleep(0.1)

    def undock_send_goal(self):
        goal_msg = Undock.Goal()
        self.undock_action_client.wait_for_server()
        goal_future = self.undock_action_client.send_goal_async(goal_msg)

        rclpy.spin_until_future_complete(self, goal_future)

        self.undock_goal_handle = goal_future.result()

        if not self.undock_goal_handle.accepted:
            self.error('Undock goal rejected')
            return

        self.undock_result_future = self.undock_goal_handle.get_result_async()

    def isUndockComplete(self):
        """
        Get status of Undock action.

        :return: ``True`` if undocked, ``False`` otherwise.
        """
        if self.undock_result_future is None or not self.undock_result_future:
            return True

        rclpy.spin_until_future_complete(self, self.undock_result_future, timeout_sec=0.1)

        if self.undock_result_future.result():
            self.undock_status = self.undock_result_future.result().status
            if self.undock_status != GoalStatus.STATUS_SUCCEEDED:
                self.info(f'Goal with failed with status code: {self.status}')
                return True
        else:
            return False

        self.info('Undock succeeded')
        return True

    def cancelTask(self):
        """Cancel pending task request of any type."""
        self.info('Canceling current task.')
        if self.result_future:
            future = self.goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, future)
        return

    def isTaskComplete(self):
        """Check if the task request of any type is complete yet."""
        if not self.result_future:
            # task was cancelled or completed
            return True
        rclpy.spin_until_future_complete(self, self.result_future, timeout_sec=0.10)
        if self.result_future.result():
            self.status = self.result_future.result().status
            if self.status != GoalStatus.STATUS_SUCCEEDED:
                self.debug(f'Task with failed with status code: {self.status}')
                return True
        else:
            # Timed out, still processing, not complete yet
            return False

        self.debug('Task succeeded!')
        return True

    def getFeedback(self):
        """Get the pending action feedback message."""
        return self.feedback

    def getResult(self):
        """Get the pending action result message."""
        if self.status == GoalStatus.STATUS_SUCCEEDED:
            return TaskResult.SUCCEEDED
        elif self.status == GoalStatus.STATUS_ABORTED:
            return TaskResult.FAILED
        elif self.status == GoalStatus.STATUS_CANCELED:
            return TaskResult.CANCELED
        else:
            return TaskResult.UNKNOWN

    def waitUntilNav2Active(self, navigator='bt_navigator', localizer='amcl'):
        """Block until the full navigation system is up and running."""
        self._waitForNodeToActivate(localizer)
        if not self.initial_pose_received:
            time.sleep(1)
        self._waitForNodeToActivate(navigator)
        self.info('Nav2 is ready for use!')
        return

    def _waitForNodeToActivate(self, node_name):
        # Waits for the node within the tester namespace to become active
        self.debug(f'Waiting for {node_name} to become active..')
        node_service = f'{node_name}/get_state'
        state_client = self.create_client(GetState, node_service)
        while not state_client.wait_for_service(timeout_sec=1.0):
            self.info(f'{node_service} service not available, waiting...')

        req = GetState.Request()
        state = 'unknown'
        while state != 'active':
            self.debug(f'Getting {node_name} state...')
            future = state_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is not None:
                state = future.result().current_state.label
                self.debug(f'Result of get_state: {state}')
            time.sleep(2)
        return
    
    def YawToQuaternion(self, angle_z = 0.):
        quat_tf = quaternion_from_euler(0, 0, angle_z)

        # Convert a list to geometry_msgs.msg.Quaternion
        quat_msg = Quaternion(x=quat_tf[0], y=quat_tf[1], z=quat_tf[2], w=quat_tf[3])
        return quat_msg

    def _amclPoseCallback(self, msg):
        self.debug('Received amcl pose')
        self.initial_pose_received = True
        self.current_pose = msg.pose
        return

    def _feedbackCallback(self, msg):
        self.debug('Received action feedback message')
        self.feedback = msg.feedback
        return
    
    def _dockCallback(self, msg: DockStatus):
        self.is_docked = msg.is_docked

    def distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def calculate_final_pos(self, robot_x, robot_y, face_x, face_y):
        dist_to_face = self.distance(robot_x, robot_y, face_x, face_y)

        self.get_logger().info(f"Distance to move: {dist_to_face - 0.5}")

        adjusted_distance = max(0, dist_to_face - 0.5)

        dir_x = (face_x - robot_x) / dist_to_face
        dir_y = (face_y - robot_y) / dist_to_face

        final_x = robot_x + dir_x * adjusted_distance
        final_y = robot_y + dir_y * adjusted_distance

        return final_x, final_y

    def _markerCallback(self, msg: Marker):

        self.map_coordinate_x = msg.pose.position.x
        self.map_coordinate_y = msg.pose.position.y
        self.get_logger().info(f"Received map coordinates are: {self.map_coordinate_x}, {self.map_coordinate_y}")

        world_x, world_y = self.map_pixel_to_world(self.map_coordinate_x, self.map_coordinate_y)
        self.get_logger().info(f"Transformed coordinates are: {world_x}, {world_y}")

        current_node = None
        count = 0
        for coordinates in listOfCordinates:
            if coordinates[2]:
                current_node = count
            count += 1
        
        currRoboCordinates_x = self.current_pose.pose.position.x
        currRoboCordinates_y = self.current_pose.pose.position.y
        self.get_logger().info(f"Robot current coordinates are: {currRoboCordinates_x}, {currRoboCordinates_y}")
        
        goalRoboCordinates_x, goalRoboCordinates_y = self.calculate_final_pos(currRoboCordinates_x, currRoboCordinates_y, world_x, world_y)
        self.get_logger().info(f"Goal coordinates are: {goalRoboCordinates_x}, {goalRoboCordinates_y}")

        # self.cancelTask()

        self.get_logger().info(f"Sleeping for 3 seconds ...")
        time.sleep(3)

        # Finally send it a goal to reach
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()

        goal_pose.pose.position.x = goalRoboCordinates_x
        goal_pose.pose.position.y = goalRoboCordinates_y
        # goal_pose.pose.orientation = self.YawToQuaternion(2.5)
        goal_pose.pose.orientation = self.current_pose.pose.orientation

        self.goToPose(goal_pose)

        while not self.isTaskComplete():
            self.info("Waiting for the task to complete...")
            time.sleep(3)
        
        self.moveAcrossMap()

    def _coordinatesCallback(self, msg: Coordinates):
        self.map_x = msg.x
        self.map_y = msg.y

        self.get_logger().info(f"Received base/link coordinates are: {self.map_x}, {self.map_y}")

        current_node = None
        count = 0
        for coordinates in listOfCordinates:
            if coordinates[2]:
                current_node = count
            count += 1
        
        currRoboCordinates_x = self.current_pose.pose.position.x
        currRoboCordinates_y = self.current_pose.pose.position.y
        self.get_logger().info(f"Robot current coordinates are: {currRoboCordinates_x}, {currRoboCordinates_y}")
        
        goalRoboCordinates_x, goalRoboCordinates_y = self.calculate_final_pos(currRoboCordinates_x, currRoboCordinates_y, self.map_x, self.map_y)
        self.get_logger().info(f"Goal coordinates are: {goalRoboCordinates_x}, {goalRoboCordinates_y}")

        # self.cancelTask()

        # self.get_logger().info(f"Sleeping for 3 seconds ...")
        # time.sleep(3)
        visit = True

        for face in self.already_visited:
            if self.distance(self.map_x, self.map_y, face[0], face[1]) < 0.6:
                visit = False
                break
        if visit:
            self.faces_coords_list.append((goalRoboCordinates_x, goalRoboCordinates_y, self.current_pose.pose.orientation))
        # world_x, world_y = self.map_pixel_to_world(self.map_x, self.map_y)
        # self.get_logger().info(f"Transformed coordinates are: {world_x}, {world_y}")

        # self.transform_points = TransformPoints(self.world_x, self.world_y)

    def _ringCallback(self, coords: Coordinates):
        center_x = coords.x
        center_y = coords.y

        self.get_logger().info(f"Received ring pixel coordinates are: {center_x}, {center_y}")    

        # world_x, world_y = self.map_pixel_to_world(center_x, center_y)
        # self.get_logger().info(f"Transformed coordinates are: {world_x}, {world_y}")

        # Finally send it a goal to reach
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()

        goal_pose.pose.position.x = center_x
        goal_pose.pose.position.y = center_y
        goal_pose.pose.orientation = self.YawToQuaternion(0.0)
        # goal_pose.pose.orientation = self.current_pose.pose.orientation

        arm_command = String()
        arm_command.data = "look_for_parking"
        self.arm_publisher.publish(arm_command)

        self.goToPose(goal_pose)

        while not self.isTaskComplete():
            self.info("Waiting for the task to complete...")
            time.sleep(3)
        
        self.moveAcrossMap()

    def setInitialPose(self, pose):
        msg = PoseWithCovarianceStamped()
        msg.pose.pose = pose
        msg.header.frame_id = self.pose_frame_id
        msg.header.stamp = 0
        self.info('Publishing Initial Pose')
        self.initial_pose_pub.publish(msg)
        return

    def info(self, msg):
        self.get_logger().info(msg)
        return

    def warn(self, msg):
        self.get_logger().warn(msg)
        return

    def error(self, msg):
        self.get_logger().error(msg)
        return

    def debug(self, msg):
        self.get_logger().debug(msg)
        return

    def moveAcrossMap(self):

        current_node = 0
        count = 0
        for coordinates in listOfCordinates:
            if coordinates[2]:
                current_node = count + 1
                break
            count += 1

        for el in listOfCordinates[current_node:]:
            el[2] = True
            x=el[0]
            y=el[1]
            # Finally send it a goal to reach
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = 'map'
            goal_pose.header.stamp = self.get_clock().now().to_msg()

            goal_pose.pose.position.x = x
            goal_pose.pose.position.y = y
            # goal_pose.pose.orientation = self.YawToQuaternion(0.0)
            goal_pose.pose.orientation = self.current_pose.pose.orientation

            self.goToPose(goal_pose)

            while not self.isTaskComplete():

                self.info("Waiting for the task to complete...")
                time.sleep(1)

            # self.faces_coords_list = []

            self.spin(-360.0)
            self.info("sleep 10...")
            time.sleep(10)

            # check if there is any face to say hello to
            if len(self.faces_coords_list) > 0:
                
                curr_x, curr_y, orientation = self.faces_coords_list[0]
                if len(self.faces_coords_list) > 1:
                    self.faces_coords_list = self.faces_coords_list[1:]
                else:
                    self.faces_coords_list = []

                # Finally send it a goal to reach
                goal_pose = PoseStamped()
                goal_pose.header.frame_id = 'map'
                goal_pose.header.stamp = self.get_clock().now().to_msg()

                goal_pose.pose.position.x = curr_x
                goal_pose.pose.position.y = curr_y
                # goal_pose.pose.orientation = self.YawToQuaternion(2.5)
                goal_pose.pose.orientation = orientation

                self.already_visited.append((curr_x, curr_y))

                self.goToPose(goal_pose)

                while not self.isTaskComplete():
                    self.info("Waiting for the task to complete...")
                    time.sleep(3)

                self.say_hello()
                
                # engine = pyttsx3.init()
                # engine.say("Hello")
                # engine.runAndWait()

                # playsound("/home/pi/Documents/RINS_Project/src/dis_tutorial3/voice/voice.wav")
                # song = AudioSegment.from_wav("/home/pi/Documents/RINS_Project/src/dis_tutorial3/voice/voice.wav")
                # play(song)

            self.faces_coords_list = []
            # self.moveAcrossMap()

            el[2] = False

def main(args=None):
    
    rclpy.init(args=args)
    rc = RobotCommander()

    # Wait until Nav2 and Localizer are available
    rc.waitUntilNav2Active()

    # Check if the robot is docked, only continue when a message is recieved
    while rc.is_docked is None:
        rclpy.spin_once(rc, timeout_sec=0.5)
        
    # If it is docked, undock it first
    if rc.is_docked:
        rc.undock()

    rc.moveAcrossMap()

    rc.destroyNode()

    # And a simple example
if __name__=="__main__":
    main()
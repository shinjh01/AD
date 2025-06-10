#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/03/28
# @author:aiden
# autonomous driving
import os
import cv2
import math
import time
import queue
import rclpy
import threading
import numpy as np
import sdk.pid as pid
import sdk.fps as fps
from rclpy.node import Node
import sdk.common as common
# from app.common import Heart
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from interfaces.msg import ObjectsInfo
from std_srvs.srv import SetBool, Trigger
from sdk.common import colors, plot_one_box
from example.self_driving import lane_detect
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from ros_robot_controller_msgs.msg import BuzzerState, SetPWMServoState, PWMServoState, ButtonState

class SelfDrivingNode(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.last_park_detect_time = 0  # 마지막으로 주차 표지판을 탐지한 시간        
        self.name = name
        self.is_running = True
        self.pid = pid.PID(0.4, 0.0, 0.05)
        self.param_init()

        self.fps = fps.FPS()  
        self.image_queue = queue.Queue(maxsize=2)
        self.depth_image = None
        self.classes = ['go', 'right', 'park', 'red', 'green', 'crosswalk']
        self.display = True
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.colors = common.Colors()
        # signal.signal(signal.SIGINT, self.shutdown)
        self.machine_type = os.environ.get('MACHINE_TYPE')
        self.lane_detect = lane_detect.LaneDetector("yellow")

        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        self.servo_state_pub = self.create_publisher(SetPWMServoState, 'ros_robot_controller/pwm_servo/set_state', 1)
        self.result_publisher = self.create_publisher(Image, '~/image_result', 1)
#        self.button_publisher = self.create_publisher(ButtonState, '/ros_robot_controller/button', 1)
#        self.create_service(Trigger, '~/button', self.button_callback) # exit the game

        self.create_service(Trigger, '~/enter', self.enter_srv_callback) # enter the game
        self.create_service(Trigger, '~/exit', self.exit_srv_callback) # exit the game
        self.create_service(SetBool, '~/set_running', self.set_running_srv_callback)

        # ButtonPressReceiver integration

        self.create_subscription(ButtonState, '/ros_robot_controller/button', self.button_callback, 10)

        # self.heart = Heart(self.name + '/heartbeat', 5, lambda _: self.exit_srv_callback(None))
        timer_cb_group = ReentrantCallbackGroup()
        self.client = self.create_client(Trigger, '/yolov5_ros2/init_finish')
        self.client.wait_for_service()
        self.start_yolov5_client = self.create_client(Trigger, '/yolov5/start', callback_group=timer_cb_group)
        self.start_yolov5_client.wait_for_service()
        self.stop_yolov5_client = self.create_client(Trigger, '/yolov5/stop', callback_group=timer_cb_group)
        self.stop_yolov5_client.wait_for_service()

        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)

    def button_callback(self, msg):
        self.get_logger().info(f"Button received: id={msg.id}, state={msg.state}")
        if msg.id == 1 and msg.state == 1:  # Button 1 short press
            self.get_logger().info("Button 1 pressed, starting self-driving")
            #self.is_running = True  # Start the self-driving process
            self.enter_srv_callback(Trigger.Request(), Trigger.Response())
            self.start = True
        elif msg.id == 2 and msg.state == 1:  # Button 1 short press
            self.get_logger().info("Button 2 pressed, stop self-driving")
            self.reset_motor_position()
            #self.is_running = False  # Start the self-driving process
            self.exit_srv_callback(Trigger.Request(), Trigger.Response())  # Call the reset motor position function

    def reset_motor_position(self):
        """
        Reset the motor position to 0.
        """
        twist = Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0
        self.mecanum_pub.publish(twist)  # Publish the zeroed Twist message
        self.get_logger().info("Motor position reset to 0")

    def init_process(self):
        self.timer.cancel()

        self.mecanum_pub.publish(Twist())
        if not self.get_parameter('only_line_follow').value:
            self.send_request(self.start_yolov5_client, Trigger.Request())
        time.sleep(1)
        
        if 1:#self.get_parameter('start').value:
            self.display = True
            self.enter_srv_callback(Trigger.Request(), Trigger.Response())
            request = SetBool.Request()
            request.data = True
            #파라미터로 처리하는것이나 어차피 시작은 멈춘 상태여야하므로.
            #self.set_running_srv_callback(request, SetBool.Response())

        #self.park_action() 
        threading.Thread(target=self.main, daemon=True).start()
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def param_init(self):
        self.start = False
        self.enter = False
        self.right = True

        self.have_turn_right = False
        self.detect_turn_right = False
        self.detect_far_lane = False
        self.park_x = -1  # obtain the x-pixel coordinate of a parking sign
        self.park_depth = -1  # obtain the x-pixel coordinate of a parking sign

        self.start_turn_time_stamp = 0
        self.count_turn = 0
        self.start_turn = False  # start to turn

        self.count_right = 0
        self.count_right_miss = 0
        self.turn_right = False  # right turning sign

        self.last_park_detect = False
        self.count_park = 0  
        self.stop = False  # stopping sign
        self.start_park = False  # start parking sign

        self.count_crosswalk = 0
        self.crosswalk_distance = 0  # distance to the zebra crossing
        self.crosswalk_length = 0.1 + 0.3  # the length of zebra crossing and the robot

        self.start_slow_down = False  # slowing down sign
        self.normal_speed = 0.3  # normal driving speed
        self.slow_down_speed = 0.2  # slowing down speed

        self.traffic_signs_status = None  # record the state of the traffic lights
        self.red_loss_count = 0

        self.object_sub = None
        self.image_sub = None
        self.objects_info = []

    def get_node_state(self, request, response):
        response.success = True
        return response

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def enter_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "self driving enter")
        with self.lock:
            self.start = False
            camera = 'depth_cam'#self.get_parameter('depth_camera_name').value
            self.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image' , self.image_callback, 1)
            self.create_subscription(ObjectsInfo, '/yolov5_ros2/object_detect', self.get_object_callback, 1)
            #self.create_subscription(Image, '/ascamera/camera_publisher/depth0/image_raw', self.depth_callback, 1)
            self.mecanum_pub.publish(Twist())
            self.enter = True
        response.success = True
        response.message = "enter"
        return response

    def exit_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "self driving exit")
        self.get_logger().info(f"plese start button")

        with self.lock:
            self.start = False
            self.enter = False
            try:
                if self.image_sub is not None:
                    self.image_sub.unregister()
                if self.object_sub is not None:
                    self.object_sub.unregister()
            except Exception as e:
                self.get_logger().info('\033[1;32m%s\033[0m' % str(e))
            self.mecanum_pub.publish(Twist())
        self.param_init()
        response.success = True
        response.message = "exit"
        return response

    def set_running_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "set_running")
        with self.lock:
            self.start = request.data
            if not self.start:
                self.mecanum_pub.publish(Twist())
        response.success = True
        response.message = "set_running"
        return response

    def shutdown(self, signum, frame):  # press 'ctrl+c' to close the program
        self.is_running = False

    def image_callback(self, ros_image):  # callback target checking
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        if self.image_queue.full():
            # if the queue is full, remove the oldest image
            self.image_queue.get()
        # put the image into the queue
        self.image_queue.put(rgb_image)

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"Failed to process depth image: {e}")
        
    # parking processing
    def park_action(self):
        self.get_logger().info(f"--- park_action:machine_type : {self.machine_type}")

        if self.machine_type == 'MentorPi_Mecanum': 
            twist = Twist()
            twist.linear.y = -0.2
            self.mecanum_pub.publish(twist)
            time.sleep(0.38/0.2)
        elif self.machine_type == 'MentorPi_Acker':
            twist = Twist()
            twist.linear.x = 0.15
            twist.angular.z = twist.linear.x*math.tan(-0.5061)/0.145
            self.mecanum_pub.publish(twist)
            time.sleep(3)

            twist = Twist()
            twist.linear.x = 0.15
            twist.angular.z = -twist.linear.x*math.tan(-0.5061)/0.145
            self.mecanum_pub.publish(twist)
            time.sleep(2)

            twist = Twist()
            twist.linear.x = -0.15
            twist.angular.z = twist.linear.x*math.tan(-0.5061)/0.145
            self.mecanum_pub.publish(twist)
            time.sleep(1.5)

        else:
            twist = Twist()
            twist.angular.z = -1
            self.mecanum_pub.publish(twist)
            time.sleep(1.5)
            self.mecanum_pub.publish(Twist())
            twist = Twist()
            twist.linear.x = 0.2
            self.mecanum_pub.publish(twist)
            time.sleep(0.65/0.2)
            self.mecanum_pub.publish(Twist())
            twist = Twist()
            twist.angular.z = 1
            self.mecanum_pub.publish(twist)
            time.sleep(1.5)

        self.reset_motor_position()
        self.exit_srv_callback(Trigger.Request(), Trigger.Response()) 
        self.mecanum_pub.publish(Twist())


    def main(self):
        self.get_logger().info('\033[1;32m -0- %s\033[0m' % self.machine_type)

        latency = 0
        while self.is_running:
            time_start = time.time()
            try:
                image = self.image_queue.get(block=True, timeout=1)
            except queue.Empty:
                if not self.is_running:
                    break
                else:
                    continue

            result_image = image.copy()
            if self.start:
                h, w = image.shape[:2]

                # obtain the binary image of the lane
                binary_image = self.lane_detect.get_binary(image)

                twist = Twist()

                # if detecting the zebra crossing, start to slow down
                #self.get_logger().info('\033[1;33m -- %s\033[0m  / latency : %s ' % (self.crosswalk_distance , latency))
                #횡단 보도 감지 및 감속 
                # 횡단보다와의 거리가 70픽셀 이상이고 아직 감속전이라면 3번 연속 감지시 감속 시작.
                if 70 < self.crosswalk_distance and not self.start_slow_down:  # The robot starts to slow down only when it is close enough to the zebra crossing
                    #반복시 감속 처리 확인
                    self.count_crosswalk += 1
                    if self.count_crosswalk == 3:  # judge multiple times to prevent false detection
                        self.count_crosswalk = 0
                        self.start_slow_down = True  # sign for slowing down
                        self.count_slow_down = time.time()  # fixing time for slowing down
                else:  # need to detect continuously, otherwise reset
                    self.count_crosswalk = 0

                # 감속처리 및 신호등 인식
                # 감속 플래그가 켜지면 신호등 상태를 확인합니다.
                # 빨간불이면 정지, 초록불이면 감속 후 통과.
                # 신호등이 없거나 정지 상태가 아니면 감속 속도로 주행, 일정 시간이 지나면 감속 해제.
                # 감속 조건이 아니면 정상 속도로 주행.
                # deceleration processing
                if self.start_slow_down:
                    if self.traffic_signs_status is not None:
                        area = abs(self.traffic_signs_status.box[0] - self.traffic_signs_status.box[2]) * abs(self.traffic_signs_status.box[1] - self.traffic_signs_status.box[3])
                        if self.traffic_signs_status.class_name == 'red' and area < 1000:  # If the robot detects a red traffic light, it will stop
                            self.mecanum_pub.publish(Twist())
                            self.stop = True
                        elif self.traffic_signs_status.class_name == 'green':  # If the traffic light is green, the robot will slow down and pass through
                            twist.linear.x = self.slow_down_speed
                            self.stop = False
                    if not self.stop:  # In other cases where the robot is not stopped, slow down the speed and calculate the time needed to pass through the crosswalk. The time needed is equal to the length of the crosswalk divided by the driving speed
                        twist.linear.x = self.slow_down_speed
                        if time.time() - self.count_slow_down > self.crosswalk_length / twist.linear.x:
                            self.start_slow_down = False
                else:
                    twist.linear.x = self.normal_speed  # go straight with normal speed

                #주차 표지판 인식.
                # If the robot detects a stop sign and a crosswalk, it will slow down to ensure stable recognition
                # if 0 < self.park_x and 300 < self.park_depth and 2100 > self.park_depth:
                #     current_time = time.time()
                #     if current_time - self.last_park_detect_time > 1:  # 1초 이상 탐지되지 않으면 초기화
                #         self.count_park = 0

                #     self.get_logger().info(f"--- self.park_x : {self.park_x} , park_depth : {self.park_depth}, count_park : {self.count_park}")
                #     self.count_park += 1  
                #     self.park_x = -1 # park 표지판 초기화
                #     self.park_depth = -1
                #     if self.count_park >= 5: 
                #         twist.linear.x = self.slow_down_speed
                #         travel_time = self.park_depth / twist.linear.x  # 이동 시간 계산 (거리 / 속도)
                #         self.start_park = True 
                #         self.get_logger().info(f"Moving forward for {travel_time:.2f} seconds to cover {self.park_depth:.2f} meters.")
                #         time.sleep(travel_time)  # 이동 시간만큼 대기
                #         self.mecanum_pub.publish(Twist())  # 정지
                #         threading.Thread(target=self.park_action).start()

                # If the robot detects a stop sign and a crosswalk, it will slow down to ensure stable recognition

                if 200 < self.park_x and 700 > self.park_x and 10 < self.park_depth and 110 > self.park_depth:
                    self.get_logger().info(f"--- self.park_x : {self.park_x} , park_depth : {self.park_depth}")
                    self.park_x = -1
                    self.park_depth = -1
                    twist.linear.x = self.slow_down_speed
                    if not self.start_park:  # When the robot is close enough to the crosswalk, it will start parking
                        self.count_park += 1  
                        if self.count_park >= 5:
                            self.mecanum_pub.publish(Twist())  
                            self.start_park = True
                            self.stop = True
                            threading.Thread(target=self.park_action).start()
                    else:
                        self.count_park = 0  

                # 차선 추적 및 PID 제어
                # 차선 중심 좌표(lane_x)가 감지되고 정지 상태가 아니면
                # 차선이 오른쪽으로 치우쳐 있으면(150 이상) 우회전 동작을 시작.
                # 그렇지 않으면 PID 제어로 차선 중심을 따라 주행.
                # 각 차종에 따라 회전 방식이 다름.
                # 차선이 감지되지 않으면 PID 상태를 초기화.
                # line following processing
                result_image, lane_angle, lane_x = self.lane_detect(binary_image, image.copy())  # the coordinate of the line while the robot is in the middle of the lane
                #self.get_logger().info('\033[1;33m lane_x :  %s , output : %s \033[0m ' % (lane_x, self.pid.output))
                if lane_x >= 0 and not self.stop:  
                    if lane_x > 150 or self.turn_right:  
                        self.count_turn += 1
                        if self.count_turn > 5 and not self.start_turn:
                            if self.turn_right:
                                self.get_logger().info(f"move right :  {self.turn_right}")

                            self.start_turn = True
                            self.count_turn = 0
                            self.start_turn_time_stamp = time.time()
                            self.turn_right = False
                        if self.machine_type != 'MentorPi_Acker':
                            twist.angular.z =  twist.linear.x * math.tan(-0.5061) / 0.145 # -0.45  # turning speed
                        else:
                            twist.angular.z = twist.linear.x * math.tan(-0.5061) / 0.145
                    else:  # use PID algorithm to correct turns on a straight road
                        self.count_turn = 0
                        if time.time() - self.start_turn_time_stamp > 2 and self.start_turn:
                            self.start_turn = False
                        if not self.start_turn:
                            self.pid.SetPoint = 130  # the coordinate of the line while the robot is in the middle of the lane
                            self.pid.update(lane_x)
                            if self.machine_type != 'MentorPi_Acker':
                                twist.angular.z = twist.linear.x * math.tan(common.set_range(self.pid.output, -0.1, 0.1)) / 0.145 # common.set_range(self.pid.output, -0.1, 0.1)
                            else:
                                twist.angular.z = twist.linear.x * math.tan(common.set_range(self.pid.output, -0.1, 0.1)) / 0.145
                        else:
                            if self.machine_type == 'MentorPi_Acker':
                                twist.angular.z = 0.15 * math.tan(-0.5061) / 0.145
                    self.mecanum_pub.publish(twist)  
                else:
                    self.pid.clear()

                #rqt로 볼때 화면에 인식 박스를 그려줌. 기본 실행시 오히려
                #성능상 이점이 없으므로 False처리. 추후 argument로 받도록 변경
                if True and self.objects_info:
                    for i in self.objects_info:
                        box = i.box
                        class_name = i.class_name
                        cls_conf = i.score
                        cls_id = self.classes.index(class_name)
                        color = colors(cls_id, True)
                        plot_one_box(
                            box,
                            result_image,
                            color=color,
                            label="{}:{:.2f}".format(class_name, cls_conf),
                        )

            else:
                time.sleep(0.01)

            self.last_main_while(result_image, time_start)

        self.mecanum_pub.publish(Twist())
        rclpy.shutdown()

    def last_main_while(self, result_image, time_start):
        bgr_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        # self.display가 True일 때만 FPS 표시 등 디스플레이 관련 처리를 합니다
        if self.display:
            self.fps.update()
            #초당 FPS계산 및 오버레이.
            bgr_image = self.fps.show_fps(bgr_image)
        #rqt 확인 용 퍼블리쉬
        self.result_publisher.publish(self.bridge.cv2_to_imgmsg(bgr_image, "bgr8"))

        #한 루프가 0.03초(약 33FPS)보다 빨리 끝났으면 남은 시간만큼 대기합니다.
        latency = time.time() - time_start
        time_d = 0.03 - latency
        #일정한 주기로 루프가 돌도록 보장합니다.
        if time_d > 0:
            time.sleep(time_d)


    # Obtain the target detection result
    def get_object_callback(self, msg):
        self.objects_info = msg.objects
        if self.objects_info == []:  # If it is not recognized, reset the variable
            self.traffic_signs_status = None
            self.crosswalk_distance = 0
        else:
            min_distance = 0
            for i in self.objects_info:
                class_name = i.class_name
                center = (int((i.box[0] + i.box[2])/2), int((i.box[1] + i.box[3])/2))
                
                if class_name == 'crosswalk':  
                    if center[1] > min_distance:  # Obtain recent y-axis pixel coordinate of the crosswalk
                        min_distance = center[1]
                elif class_name == 'right':  # obtain the right turning sign
                    self.count_right += 1
                    self.count_right_miss = 0
                    if self.count_right >= 5:  # If it is detected multiple times, take the right turning sign to true
                        self.turn_right = True
                        self.count_right = 0
                        self.get_logger().info(f"right :  {self.turn_right}")

                elif class_name == 'park':  # obtain the center coordinate of the parking sign
                    self.park_x = center[0]
                    self.park_depth = center[1] # self.depth_image[center[1], center[0]]  # 중심 좌표의 깊이 값
                    self.last_park_detect_time = time.time()  # 마지막 탐지 시간 갱신
                    self.get_logger().info(f"park_depth {self.park_depth}, park_x : {self.park_x} , center[1] : {center[1]}")

                elif class_name == 'red' or class_name == 'green':  # obtain the status of the traffic light
                    self.traffic_signs_status = i
               

            self.get_logger().info('\033[1;32m%s\033[0m' % class_name)
            self.crosswalk_distance = min_distance

def main():
    node = SelfDrivingNode('self_driving')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
 
if __name__ == "__main__":
    main()

    

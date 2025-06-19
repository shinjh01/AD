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
import socket
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
from ros_robot_controller_msgs.msg import BuzzerState, SetPWMServoState, PWMServoState
####
from ros_robot_controller_msgs.msg import RGBStates, RGBState, ButtonState
####

############# 
# 메인 돌때 이미지가 1프레임이 들어옴. 
#이 이미지에서 횡단보도를 찾아야하는데 거리가 멀거나 회전시 횡단보도가 찌그러져 보이므로
# 해당 이미지를 펴준다. 
# 1. 640,480 이미지 중에 특정 영역을 사다리꼴로 지정 
# (해당 부분은 실제 들어온 이미지를 참고하여 하드코딩)
# 2. 해당 부분의 사다리꼴을 직사각형의 이미지를 변환.
#############
def warp_perspective(image):
    h, w = image.shape[:2]

    # 1. 원근 변환을 위한 4점 (실제 테스트 환경에 따라 수동 튜닝 필요)
    # 횡단보다가 해당 위치에 찌그러져서 존재한다는 가정을 하고 해당 위치의 관심 구역을 추가 
    # 펴고 싶은 사다리꼴 영역의 네 꼭짓점 좌표
    src = np.float32([
        [w * 0.1, h * 0.8], # w 64, h 384
        [w * 0.9, h * 0.8], # 576, 384
        [w * 0.6, h * 0.6], # 384, 288,
        [w * 0.4, h * 0.6]  # 256, 288 
    ])

    # 변환후에 내가 보고 싶은 직사각형의 모양 
    dst = np.float32([
        [w * 0.2, h], 
        [w * 0.8, h],
        [w * 0.8, 0],
        [w * 0.2, 0]
    ])

    matrix = cv2.getPerspectiveTransform(src, dst) # 변환 행렬 계산 
    warped = cv2.warpPerspective(image, matrix, (w, h)) # 사다리꼴을 직사각형으로 변환

    return warped

############# 
# warp_perspective 에서 직사각형에 처리된 이미지 내에 횡단보고가 있는지 처리.
# #횡단보도 open cv로 디텍팅.
# param : 횡단보도 원본 이미지
# return : 횡단보도 디텍팅오브젝트 / 횡단보도 원본이미지
############# 
def detect_crosswalk_v2(image):
    original = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 1. 흰색 마스크 (HSV 기반) - 색상/채도/ 명도
    # ● RGB와 HSV의 차이
    #   RGB는 색상을 빨강(R), 초록(G), 파랑(B) 값으로 표현합니다.
    #   HSV는 색상(Hue), 채도(Saturation), 명도(Value)로 표현합니다.
    # ● 흰색 검출에 HSV가 유리한 이유
    #   흰색은 RGB에서 R, G, B 값이 모두 높을 때 나타나지만,
    #   조명, 그림자, 카메라 노이즈 등으로 인해 RGB 값만으로 흰색을 정확히 구분하기 어렵습니다.
    #   HSV에서는 흰색이 "채도(S)가 낮고, 명도(V)가 높음"이라는 특징이 있습니다.
    #   즉, 색상(H)은 상관없고, S(채도)는 낮고, V(명도)는 높으면 흰색 계열로 볼 수 있습니다.

    # H(색상): 0 ~ 180 (전체 색상 범위, 흰색은 색상에 영향 없음)
    # S(채도): 0 ~ 40 (채도가 낮아야 흰색, 즉 거의 무채색)
    # V(명도): 180 ~ 255 (명도가 높아야 흰색, 즉 밝은 색)
    # 즉, 이 마스크는 "채도가 낮고 밝은 픽셀"만 남깁니다.
    # → 흰색 실선(횡단보도, 차선 등)만 효과적으로 추출할 수 있습니다.

    lower_white = np.array([0, 0, 180]) # 왜 0,0,180 인가?
    upper_white = np.array([180, 40, 255]) #왜?
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # 2. Morphology로 잡음 제거 + 채우기
    # 연산 범위를 결정합니다. (5x5 픽셀 영역)
    kernel = np.ones((5, 5), np.uint8) 

    #모폴로지 클로징(Close) 연산을 적용합니다.
    #클로징이란?
    #팽창(Dilation) 후 침식(Erosion)
    #작은 구멍(검은 점)이나 끊어진 부분을 메워줍니다.
    #역할:
    #횡단보도 실선이 끊어져 있거나, 작은 검은 점(노이즈)이 있을 때 이를 메워서 실선을 더 잘 검출할 수 있게 합니다.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    #     설명:
    # 모폴로지 오프닝(Open) 연산을 적용합니다.
    # 오프닝이란?
    # 침식(Erosion) 후 팽창(Dilation)
    # 작은 흰 점(노이즈)을 제거합니다.
    # 역할:
    # 배경에 남아있는 작은 흰색 노이즈(점, 잡음)를 제거해서 실선만 남기기 쉽게 만듭니다.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 3. 컨투어 추출 - (윤곽선)
    # 이 코드는 이진화된 마스크 이미지에서 외곽선(윤곽선, contour)을 찾는 함수입니다.
    # 횡단보도 흰색 실선이 잘 추출되어 있다면, 실선마다 하나씩 컨투어가 검출됩니다.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vertical_boxes = []
    for cnt in contours:
        #한 윤관석의 좌표 추출
        x, y, w, h = cv2.boundingRect(cnt)

        #좌표 비율
        aspect_ratio = h / float(w + 1e-5)
        area = w * h

        # 조건: 세로형 직사각형 + 크기 제한
        #→ 세로로 긴(높이가 너비보다 1.5~10배 크고), 면적이 500픽셀 이상인 컨투어만 선택합니다.
        if 1.5 < aspect_ratio < 10 and area > 500:
            #조건에 맞는 정보만 배열에 넣기
            vertical_boxes.append((x, y, w, h))
            #시각화
            cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 4. 수직 패턴 개수가 충분하면 횡단보도
    crosswalk_detected = len(vertical_boxes) >= 5

    #if crosswalk_detected:
    #    cv2.putText(original, 'CROSSWALK DETECTED', (30, 30),
    #                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return crosswalk_detected, original

class SelfDrivingNode(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.name = name
        self.is_running = True
        self.pid = pid.PID(0.4, 0.0, 0.05)
        self.param_init()
        # LED통신을 위한 소켓 연동.(port연결)
        self.HOST = '127.0.0.1'  # LTS 250610
        self.PORT = 65432        # LTS 250610
        
        
        # LTS 250610
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.HOST, self.PORT))
        
        self.send_message("redon")
        self.send_message("greenoff")

        self.fps = fps.FPS()  
        self.image_queue = queue.Queue(maxsize=2)
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
        
        #### LED RGB
        self.rgb_pub= self.create_publisher(RGBStates, '/ros_robot_controller/set_rgb', 10)
        self.get_logger().info('RGB Controller Node has been started.')
        
        self.msg_on = RGBStates()
        self.msg_off = RGBStates()
        self.msg_left = RGBStates()
        self.msg_right = RGBStates()
        
        self.msg_left.states = [
                RGBState(index=1, red=255, green=255, blue=0),  # 1번 LED 빨강색
                RGBState(index=2, red=0, green=0, blue=0)   # 2번 LED 초록색
        ]
        self.msg_right.states = [
                RGBState(index=1, red=0, green=0, blue=0),  # 1번 LED 빨강색
                RGBState(index=2, red=255, green=255, blue=0)   # 2번 LED 초록색
        ]
        self.msg_on.states = [
                RGBState(index=1, red=255, green=255, blue=0),  # 1번 LED 빨강색
                RGBState(index=2, red=255, green=255, blue=0)   # 2번 LED 초록색
        ]
        self.msg_off.states = [
                RGBState(index=1, red=0, green=0, blue=0),  # 1번 LED 빨강색
                RGBState(index=2, red=0, green=0, blue=0)   # 2번 LED 초록색
        ]
        #self.rgb_pub.publish(msg)
        
        ###################################################################################################
        
        # rgb_sub = message_filters.Subscriber(Image, '/ascamera/camera_publisher/rgb0/image')
        # depth_sub = message_filters.Subscriber( Image, '/ascamera/camera_publisher/depth0/image_raw')
        
        # ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.05)
        # ts.registerCallback(self.callback)
        # self.get_logger().info('Depth value at (200, 200)')
        
        # Define the directory to save images
        self.IMAGE_SAVE_DIR = "logimage"  # Saving images in the 'logimage' folder
        ### 버튼 클릭 처리 가능하도록 구독시작
        ###################################################################################################
        self.create_subscription(ButtonState, '/ros_robot_controller/button', self.button_callback, 10)
        self.get_logger().info('ButtonPressReceiver node started')
        ######################################################################################################
        
        self.result_publisher = self.create_publisher(Image, '~/image_result', 1)

        ### if press 1 then 'set_running' service call
        self.set_running_client = self.create_client(SetBool, '~/set_running')
        ######
        ### if press 2 then 'exit' service call
        self.exit_client = self.create_client(Trigger, '~/exit')
        ######
                
        self.create_service(Trigger, '~/enter', self.enter_srv_callback) # enter the game
        self.create_service(Trigger, '~/exit', self.exit_srv_callback) # exit the game
        self.create_service(SetBool, '~/set_running', self.set_running_srv_callback)
        # self.heart = Heart(self.name + '/heartbeat', 5, lambda _: self.exit_srv_callback(None))
        timer_cb_group = ReentrantCallbackGroup()
        self.client = self.create_client(Trigger, '/yolov5_ros2/init_finish')
        self.client.wait_for_service()
        self.start_yolov5_client = self.create_client(Trigger, '/yolov5/start', callback_group=timer_cb_group)
        self.start_yolov5_client.wait_for_service()
        self.stop_yolov5_client = self.create_client(Trigger, '/yolov5/stop', callback_group=timer_cb_group)
        self.stop_yolov5_client.wait_for_service()

        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)
        

        
        #################
        
    # def callback(self, rgb_msg, depth_msg):
    #         # Convert ROS Image messages to OpenCV format
    #         rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
    #         depth = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
    #         #self.get_logger().info('Depth value at (200, 200): %d' % depth[200, 200])
    ###########################myadd######################################################################
    def start_blinking_right_led(self, duration=3.0, interval=0.5):
        """
        우회전 LED를 깜빡이는 함수.
        :param duration: 깜빡임을 유지할 총 시간 (초)
        :param interval: LED를 켜고 끄는 간격 (초)
        """
        self.get_logger().info("\033[1;34m[blink_right_led start]]\033[0m")
        self.blinking_start_time = time.time()
        self.blinking_active = True
        self._blink_right_led_toggle(interval, duration)

    ############# 
    # #우회전 LED 토글(깜빡임)
    ############# 
    def _blink_right_led_toggle(self, interval, duration):
        if not self.blinking_active:
            return

        current_time = time.time()
        if current_time - self.blinking_start_time > duration:
            self.rgb_pub.publish(self.msg_off) # 깜빡이 종료 후 LED 끄기
            self.blinking_active = False
            self.get_logger().info("\033[1;32m[blink_right_led end]\033[0m")
            return

        # 현재 LED 상태를 토글
        if hasattr(self, '_right_led_on') and self._right_led_on:
            self.rgb_pub.publish(self.msg_off)
            self._right_led_on = False
        else:
            self.rgb_pub.publish(self.msg_right) # self.msg_right는 2번 LED만 켜도록 설정되어 있음
            self._right_led_on = True

        # 다음 토글을 위한 타이머 설정
        self.blinking_timer = threading.Timer(interval, self._blink_right_led_toggle, args=[interval, duration])
        self.blinking_timer.start()
    
    ############# 
    # # 드라이빙 멈춤
    ############# 
    def stop_driving(self):
        self.start = False  # 주행 정지
        self.mecanum_pub.publish(Twist())  # 멈추기 위한 Twist 명령
        self.get_logger().info('Driving has been stopped.')
        self.send_message("redon")
        self.send_message("greenoff")
        
    # def resume_driving(self):
    #     self.start = True  # 주행 시작
    #     request = SetBool.Request()
    #     request.data = True
    #     self.set_running_srv_callback(request, SetBool.Response())  # 주행 재개
    #     self.get_logger().info('Driving has been resumed.')
        
    ############# 
    # # 시작 종료 버튼용 콜백 함수. 
    #############         
    def button_callback(self, msg):
        if msg.id == 1:
            self.process_button_press('Button 1', msg.state)
            self.send_message("redoff")
            self.send_message("greenon")
            ## client service call
            # 1) 요청 객체 생성
            self.display = True
            self.enter_srv_callback(Trigger.Request(), Trigger.Response())
            request = SetBool.Request()
            request.data = True
            
            self.set_running_srv_callback(request, SetBool.Response())
            #self.resume_driving()
            #self.rgb_pub.publish(msg)
        elif msg.id == 2:
            self.process_button_press('Button 2', msg.state)
            ## client service call
            
            self.stop_driving()
            #self.rgb_pub.publish(msg)
    
    def process_button_press(self, button_name, state):
        if state == 1:
            self.get_logger().info(f'{button_name} short press detected')
            # You can add additional logic here for short press
        elif state == 2:
            self.get_logger().info(f'{button_name} long press detected')
            # You can add additional logic here for long press
###############################################

    # LTS 250610
    def send_message(self, message):
        self.client_socket.sendall(message.encode())
        
    def init_process(self):
        self.timer.cancel()

        self.mecanum_pub.publish(Twist())
        if not self.get_parameter('only_line_follow').value:
            self.send_request(self.start_yolov5_client, Trigger.Request())
        time.sleep(1)
        
        ############# 
        # # ROS 시작시 enter_srv_callbak제거 (버튼 클릭시 작동 되도록)
        ############# 
        if False:#self.get_parameter('start').value:
            self.display = True
            self.enter_srv_callback(Trigger.Request(), Trigger.Response())
            request = SetBool.Request()
            request.data = True
            self.set_running_srv_callback(request, SetBool.Response())

        #self.park_action() 
        threading.Thread(target=self.main, daemon=True).start()
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def param_init(self):
        self.start = False
        self.enter = False
        self.right = True

        ############# 
        # # 우회전 감지를 위한 파라미터
        ############# 
        self.right_turn_state = "IDLE"
        self.right_detect_time = 0.0    # 감지된 시점
        self.right_front_flag = False
        self.detect_far_lane = False
        self.park_x = -1  # obtain the x-pixel coordinate of a parking sign

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
        self.normal_speed = 0.24  # normal driving speed
        self.slow_down_speed = 0.1  # slowing down speed

        self.traffic_signs_status = None  # record the state of the traffic lights
        self.red_loss_count = 0

        self.object_sub = None
        self.image_sub = None
        self.objects_info = []
        
        self.crosswalk_stop = False
        self.crosswalk_stop_time = 0
        self.last_crosswalk_pause_time = 0
        self.crosswalk_count = 0
        
        self.main_count = 0
        

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
            #depth_sub = message_filters.Subscriber( Image, '/ascamera/camera_publisher/depth0/image_raw')
            self.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image' , self.image_callback, 1)
            #self.create_subscription(Image, '/ascamera/camera_publisher/depth0/image_raw' , self.depth_image_callback, 1)
            
            self.create_subscription(ObjectsInfo, '/yolov5_ros2/object_detect', self.get_object_callback, 1)
            self.mecanum_pub.publish(Twist())
            self.enter = True
        response.success = True
        response.message = "enter"
        return response

    def exit_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "self driving exit")
        with self.lock:
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
############# 
# # 뎁스 카메라 사용X
#############         
    def depth_image_callback(self, ros_image):  # callback target checking
        cv_depth = self.bridge.imgmsg_to_cv2(ros_image, "16UC1")
        #self.get_logger().info('Depth value at (200, 200): %d' % cv_depth[200, 200])
        
        # rgb_image = np.array(cv_image, dtype=np.uint8)
        # if self.image_queue.full():
        #     # if the queue is full, remove the oldest image
        #     self.image_queue.get()
        # # put the image into the queue
        # self.image_queue.put(rgb_image)

############# 
# # 우회전.
#############
    def turn_right_action(self):
        self.get_logger().info("\033[1;34m[우회전 실행 - 자연 회전]\033[0m")
        self.rgb_pub.publish(self.msg_right)

        twist = Twist()
        
        if self.machine_type == 'MentorPi_Mecanum':
            twist.linear.x = 0.15       # 전진 속도 유지
            twist.angular.z = -0.8      # 우회전 회전값 (음수)

        elif self.machine_type == 'MentorPi_Acker':
            twist.linear.x = 0.15
            twist.angular.z = twist.linear.x * math.tan(-0.5061) / 0.145  # Ackermann steering

        else:
            twist.linear.x = 0.15
            twist.angular.z = -0.8

        self.mecanum_pub.publish(twist)
        
        self.start_turn_right = True
        self.turn_right_start_time = time.time()

    
############# 
# # 붉은 색.
############# 
    def red_detection(self,image):
        #image = cv2.imread('sample.jpg')  # 이미지 경로 지정
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # 2. LAB 색상 범위 정의 (직접 지정)
        # 예시: 밝기(L) 50~200, A 120~140, B 140~170
        lower_bound = np.array([64, 159, 139])
        upper_bound = np.array([175, 255, 213])

        # 3. 마스크 생성
        mask = cv2.inRange(lab_image, lower_bound, upper_bound)

        # 4. 살아남은 픽셀 수 세기
        pixel_count = cv2.countNonZero(mask)
        return pixel_count    

    # parking processing
    def park_action(self):
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
        self.mecanum_pub.publish(Twist())
############# 
# # 주차 후 깜빡임
############# 
        while(True):
            self.send_message("redoff")
            self.send_message("greenoff")
            self.rgb_pub.publish(self.msg_off)
            time.sleep(1)
            self.send_message("redon")
            self.send_message("greenon")
            self.rgb_pub.publish(self.msg_on)
            time.sleep(1)

    def main(self):
        
        while self.is_running:
            self.main_count += 1
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
############# 
# #횡단보도 플래그 이고 감지 시간 1.1초 이하시 멈춤
############# 

                if (self.crosswalk_stop == True) and time.time() - self.crosswalk_stop_time <= 1.1:
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.mecanum_pub.publish(twist)
                    continue
############# 
# #횡단보도 플래그 이고 감지 시간 1.1초 이상이면 진행
############# 
                elif (self.crosswalk_stop == True) and time.time() - self.crosswalk_stop_time > 1.1:
                    self.crosswalk_stop = False
                    self.send_message("redoff")
                    self.send_message("greenon")
############# 
# #우회 전 상태처리

############# 
                # 우회전 상태 처리
                if self.right_turn_state == "DETECTED":
                    #### 우회전 표지판 디텍팅되었지만 우회전 시작 전이면 시작
                    if self.right_front_flag == False:
                        self.right_detect_time = time.time()
                        self.right_front_flag = True
                    #### 우회전 시작 후 1.5초 내라면 z축에 0.2로 추가 후 continue    
                    if time.time() - self.right_detect_time <= 1.5:
                        twist.linear.x = 0.25
                        twist.angular.z = 0.2      # lts 250613 0 -> 0.2, continue 추가
                        self.mecanum_pub.publish(twist)
                        continue
                    #### 우회전 완료되었다면 TURNING으로 2번이상 우회전 되지 않도록 처리
                    else:   
                        self.turn_right_start_time = time.time()
                        self.right_turn_state = "TURNING"
                        self.right_front_flag = False

############# 
# # 우회전 완료 후 1.7초 후라면 z축 0으로 해주고
# x축에 노말 스피드. 
############# 

                if self.right_turn_state == "TURNING":
                    if time.time() - self.turn_right_start_time >= 1.7:
                        twist = Twist()
                        twist.linear.x = self.normal_speed
                        twist.angular.z = 0.0
                        self.mecanum_pub.publish(twist)

                        #self.rgb_pub.publish(self.msg_off)
                        self.get_logger().info("\033[1;32m[turn right end]\033[0m")
                        self.right_turn_state = "DONE"
                    ### 아니라면 -0.65 각도로 우회전(왜?)    
                    else:
                        twist = Twist()
                        twist.linear.x = 0.2
                        twist.angular.z = -0.65
                        self.mecanum_pub.publish(twist)
                        self.start_blinking_right_led()
                        continue


                # if detecting the zebra crossing, start to slow down
                current_time = time.time()
              
############# 
# # 우회 전 후 1.7초간 직진까지 완료되면 횡단보도 추출
############# 
                if self.right_turn_state == "DONE":
                    warped_img = warp_perspective(result_image)
                    cross_detected , _ = detect_crosswalk_v2(warped_img)
                else:
############# 
# # 그 외 상황에서는 3프레임 이후(메인 반복문이 3번씩 돌때마다) 횡단보도 디텍팅 추출
############# 
                    if self.main_count % 3 == 0:
                        warped_img = warp_perspective(result_image)
                        cross_detected , _ = detect_crosswalk_v2(warped_img)
                    else:
                        cross_detected = False 

############# 
# # 횡단보도 디텍팅 시 횡단보도 디텍팅 처리 시작(상단 코드에서 진행.)
#############                         
                if cross_detected and (current_time - self.last_crosswalk_pause_time > 1.5): #and not self.start_slow_down:  # The robot starts to slow down only when it is close enough to the zebra crossing 250613 LTS 
                    self.send_message("redon")
                    self.send_message("greenoff")
                    #self.start = False  # 주행 정지
                    self.crosswalk_stop = True
                    self.crosswalk_stop_time = current_time
                    self.last_crosswalk_pause_time = current_time
                    self.crosswalk_count += 1
                    continue
                

############# 
# #신호등 처리
############# 

                # deceleration processing
                #if self.start_slow_down:
                if self.traffic_signs_status is not None:
                    area = abs(self.traffic_signs_status.box[0] - self.traffic_signs_status.box[2]) * abs(self.traffic_signs_status.box[1] - self.traffic_signs_status.box[3])
                    if self.traffic_signs_status.class_name == 'red': 
                        #red_count = self.red_detection(result_image)
                        self.get_logger().info(f"red area : {area}")
                        #self.get_logger().info(f"red count : {red_count}")
                        #and red_count > 600 and red_count < 1000
                    if self.traffic_signs_status.class_name == 'red' and area > 300 and area < 1000 : # If the robot detects a red traffic light, it will stop
                        self.send_message("redon")
                        self.send_message("greenoff")
                        self.mecanum_pub.publish(Twist())
                        self.stop = True
                        self.get_logger().info("\033[31mred\033[0m")
                    elif self.traffic_signs_status.class_name == 'red' and area <= 300 : # If the robot detects a red traffic light, it will stop
                        self.stop = False
                        twist.linear.x = 0.18
                        
                    elif self.traffic_signs_status.class_name == 'green':  # If the traffic light is green, the robot will slow down and pass through
                        twist.linear.x = self.normal_speed
                        self.stop = False
                        self.send_message("redoff")
                        self.send_message("greenon")
                        self.get_logger().info("\033[33mgreen\033[0m")
                else:
                    twist.linear.x = self.normal_speed 
                    self.stop = False


############# 
# #우회전 종료됴ㅚ고 횡단보도 판별되고 park가 표지판 디텍팅시에
############# 
                # If the robot detects a stop sign and a crosswalk, it will slow down to ensure stable recognition
                if 0 < self.park_x and cross_detected and self.right_turn_state == "DONE":
                    self.mecanum_pub.publish(Twist())  
                    self.start_park = True
                    self.stop = True
                    self.stop_driving()
                    self.get_logger().info("\033[31mfinal park\033[0m")
                    threading.Thread(target=self.park_action).start()
                    continue
                    
                    #twist.linear.x = self.slow_down_speed
                    if not self.start_park and 180 < self.crosswalk_distance:  # When the robot is close enough to the crosswalk, it will start parking
                        self.count_park += 1  
                        if self.count_park >= 15:  
                            self.mecanum_pub.publish(Twist())  
                            self.start_park = True
                            self.stop = True
                            threading.Thread(target=self.park_action).start()
                    else:
                        self.count_park = 0  

                # line following processing
                result_image, lane_angle, lane_x = self.lane_detect(binary_image, image.copy())  # the coordinate of the line while the robot is in the middle of the lane
                self.get_logger().info(f"{lane_x}")
                
                # 250613 lts
                if lane_x >= 0 and not self.stop:  
                    if lane_x > 140: #우회전 차선 크기 변경 150 -> 140
                        self.count_turn += 1
                        if self.count_turn > 5 and not self.start_turn:
                            self.start_blinking_right_led()
                            #start_blinking_right_led()
                            self.start_turn = True
                            self.count_turn = 0
                            self.start_turn_time_stamp = time.time()
                            # HSM
                            
                        if self.machine_type != 'MentorPi_Acker':
                            twist.angular.z = -0.45 #-0.45  # turning speed        

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
                                twist.angular.z = common.set_range(self.pid.output, -0.1, 0.1)
                                self.get_logger().info(f"{lane_x} PID {twist.angular.z}")
                            else:
                                twist.angular.z = twist.linear.x * math.tan(common.set_range(self.pid.output, -0.1, 0.1)) / 0.145
                        else:
                            if self.machine_type == 'MentorPi_Acker':
                                twist.angular.z = 0.15 * math.tan(-0.5061) / 0.145
                    self.mecanum_pub.publish(twist)  
                else:
                    self.pid.clear()

             
                if self.objects_info:
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

            
            bgr_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            if self.display:
                self.fps.update()
                bgr_image = self.fps.show_fps(bgr_image)

            
            self.result_publisher.publish(self.bridge.cv2_to_imgmsg(bgr_image, "bgr8"))

           
            time_d = 0.03 - (time.time() - time_start)
            if time_d > 0:
                time.sleep(time_d)
        self.mecanum_pub.publish(Twist())
        rclpy.shutdown()

    def get_object_callback(self, msg):
        self.objects_info = msg.objects

        if not self.objects_info:
            self.traffic_signs_status = None
            self.crosswalk_distance = 0
            return

        # 가장 score 높은 객체 1개만 선택
        top_obj = max(self.objects_info, key=lambda x: x.score)
        class_name = top_obj.class_name
        center = (int((top_obj.box[0] + top_obj.box[2]) / 2), int((top_obj.box[1] + top_obj.box[3]) / 2))

        # 클래스별 처리 로직
        if class_name == 'crosswalk':
            self.crosswalk_distance = center[1]

        elif class_name == 'right':
            if self.right_turn_state == "IDLE" and self.crosswalk_count >= 3:
                self.right_turn_state = "DETECTED"
                self.right_detect_time = time.time()
            self.count_right += 1 # 해당 플래그 사용안함.
            self.count_right_miss = 0
            if self.count_right >= 5:
                self.turn_right = True
                self.count_right = 0

        elif class_name == 'go':
            pass  # 현재 go_sign은 단순 True 여부라 별도 상태 업데이트 생략

        elif class_name == 'park':
            self.park_x = center[0]

        elif class_name in ['red', 'green']:
            self.traffic_signs_status = top_obj
############# 
# go가 아닌 경우, 그리고 우회전이 완료되지 않은 경우만 park_x 무효화
############# 
        # 
        if class_name != 'go' and self.right_turn_state != "DONE":
            self.park_x = -1

        self.get_logger().info('\033[1;32m%s\033[0m' % class_name)

def main():
    node = SelfDrivingNode('self_driving')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
 
if __name__ == "__main__":
    main()

import unittest
from unittest.mock import MagicMock, patch
import sys

sys.path.append('/home/js/project/AD/ros2_ws/src/example')

# 필요한 라이브러리 Mock 처리 (Node만 MagicMock, 나머지는 그대로)
mock_modules = {
    'cv2': MagicMock(),
    'cv_bridge': MagicMock(),
    'sensor_msgs.msg': MagicMock(),
    'geometry_msgs.msg': MagicMock(),
    'interfaces.msg': MagicMock(),
    'std_srvs.srv': MagicMock(),
    'rclpy.executors': MagicMock(),
    'rclpy.callback_groups': MagicMock(),
    'ros_robot_controller_msgs.msg': MagicMock(),
    'gpiozero': MagicMock(),
    'numpy': MagicMock(),
    'sdk': MagicMock(),
    'sdk.pid': MagicMock(),
    'sdk.fps': MagicMock(),
    'sdk.common': MagicMock(),
    'rclpy' : MagicMock(),
    'example.self_driving.lane_detect': MagicMock(),
    # 나머지는 필요할 때만 추가
}

for module_name, mock_module in mock_modules.items():
    sys.modules[module_name] = mock_module

from example.self_driving.self_driving import SelfDrivingNode

class MockTriggerResponse:
    def __init__(self, success=False, message=""):
        self.success = success
        self.message = message

class MockTriggerRequest:
    pass

class TestSelfDrivingNode(unittest.TestCase):
    def setUp(self):
        self.node = SelfDrivingNode('test_self_driving')
        self.node.get_logger = MagicMock()
        self.node.lock = MagicMock()
        self.node.start = False
        self.node.enter = False
        self.node.image_sub = None
        self.node.object_sub = None
        self.node.mecanum_pub = MagicMock()
        self.node.create_subscription = MagicMock()
        self.node.param_init = MagicMock()
        self.node.rgb_color_publish = MagicMock()

    def test_get_node_state(self):
        request = MockTriggerRequest()
        response = MockTriggerResponse()
        result = self.node.get_node_state(request, response)
        print(result)
        print(type(result))
        print(f"result.success ::: {result.success}")
        self.assertTrue(result.success == True)

if __name__ == '__main__':
    unittest.main()
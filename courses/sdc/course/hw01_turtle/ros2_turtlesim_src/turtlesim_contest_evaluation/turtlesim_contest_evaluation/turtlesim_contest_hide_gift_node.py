import sys
import rclpy

from rclpy.node import Node
from turtlesim.srv import SetPen, TeleportAbsolute
from turtlesim_contest_interface.msg import Rect


class DrawRectNode(Node):

    def __init__(self):
        super().__init__('contest_gift_hider')

        turtle_name = 'turtle1'
        self.turtlesim_pen_client = self.create_client(SetPen, f'{turtle_name}/set_pen')
        self.services_available = False

        if not self.turtlesim_pen_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error(f'{turtle_name}/set_pen service not available')
            return

        self.turtlesim_teleport_client = self.create_client(TeleportAbsolute, f'{turtle_name}/teleport_absolute')
        if not self.turtlesim_teleport_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error(f'{turtle_name}/teleport_absolute service not available')
            return

        self.services_available = True

    def draw_rect(self, bottom_left_x, bottom_left_y, top_right_x, top_right_y):
        self._disable_pen()
        self._teleport_to(bottom_left_x, bottom_left_y)
        
        self._enable_pen()
        self._teleport_to(bottom_left_x, top_right_y)
        self._teleport_to(top_right_x, top_right_y)
        self._teleport_to(top_right_x, bottom_left_y)
        self._teleport_to(bottom_left_x, bottom_left_y)

        y = bottom_left_y
        y_resolution = 0.025
        while y < top_right_y - y_resolution:
            self._draw_line(bottom_left_x, y, top_right_x, y)
            y += y_resolution
            self._draw_line(top_right_x, y, bottom_left_x, y)
            y += y_resolution
        
        self._disable_pen()
        self._teleport_to(0.0, 0.0)

        self._set_default_pen()

    def _disable_pen(self):
        disabled_pen = SetPen.Request()
        disabled_pen.off = 1
        self.future = self.turtlesim_pen_client.call_async(disabled_pen)
        rclpy.spin_until_future_complete(self, self.future)

    def _enable_pen(self):
        green_pen = SetPen.Request()
        green_pen.r=0
        green_pen.g=255
        green_pen.b=0
        green_pen.width = 2
        green_pen.off=0
        self.future = self.turtlesim_pen_client.call_async(green_pen)
        rclpy.spin_until_future_complete(self, self.future)

    def _set_default_pen(self):
        default_pen = SetPen.Request()
        default_pen.r=179
        default_pen.g=184
        default_pen.b=255
        default_pen.width = 3
        default_pen.off=0
        self.future = self.turtlesim_pen_client.call_async(default_pen)
        rclpy.spin_until_future_complete(self, self.future)

    def _draw_line(self, x1, y1, x2, y2):
        self._teleport_to(x1, y1)
        self._teleport_to(x2, y2)

    def _teleport_to(self, x, y):
        target_position = TeleportAbsolute.Request()
        target_position.x = x
        target_position.y = y
        self.future = self.turtlesim_teleport_client.call_async(target_position)
        rclpy.spin_until_future_complete(self, self.future)


def main(args=None):
    
    if len(sys.argv) !=5:
        print(f'Usage: ros2 run turtlesim_contest_evaluation turtlesim_contest_hide_gift_node gift_center_x gift_center_y gift_width gift_height')
        return
    
    rclpy.init(args=args)
    draw_rect_node = DrawRectNode()

    if not draw_rect_node.services_available:
        print(f'Turtlesim services is not available. Is the turtlesim running?')
        return

    gift_center_x = float(sys.argv[1])
    gift_center_y = float(sys.argv[2])
    gift_witdh = float(sys.argv[3])
    gift_height = float(sys.argv[4])

    draw_rect_node.draw_rect(gift_center_x - gift_witdh*0.5, gift_center_y - gift_height*0.5,
                                        gift_center_x + gift_witdh*0.5, gift_center_y + gift_height*0.5)

    draw_rect_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

import math
import rclpy
from enum import Enum
from rclpy.node import Node
from rclpy.timer import Timer
from rclpy.task import Future
from turtlesim.srv import SetPen
from rclpy.action import ActionServer
from dataclasses import dataclass, field
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from geometry_msgs.msg import Twist
from turtlesim.msg import Pose, Color

from turtlesim_contest_interface.action import FindHiddenGift   # type: ignore[attr-defined]
from turtlesim_contest_interface.msg import Rect                # type: ignore[attr-defined]


@dataclass
class State:
    pose: Pose = field(default_factory=Pose)
    color: Color = field(default_factory=Color)

@dataclass
class PoseXY:
    x: float
    y: float


class Turns(Enum):
    left = - math.pi
    right = 0.0
    up = math.pi / 2
    down = - math.pi / 2

def target_angle(from_pt: PoseXY, to_pt: PoseXY) -> float:
    dx = to_pt.x - from_pt.x
    dy = to_pt.y - from_pt.y
    return math.atan2(dy, dx)


def angle_diff(target: float, current: float) -> float:
    d = target - current
    return (d + math.pi) % (2 * math.pi) - math.pi


YIELD_SLEEP_TIME = 0.02


class FindHiddenGiftNode(Node):
    def __init__(self):
        super().__init__('turtlesim_contest_submission_node')
        turtle_name = 'turtle1'

        self.state = State()

        self._pose_ready: Future = Future()
        self._color_ready: Future = Future()

        io_group = ReentrantCallbackGroup()
        action_group = MutuallyExclusiveCallbackGroup()

        self.pen_client = self.create_client(
            SetPen,
            f'{turtle_name}/set_pen',
            callback_group=io_group,
        )

        self.subscription_pose = self.create_subscription(
            msg_type=Pose,
            topic=f'/{turtle_name}/pose',
            callback=self.on_pose,
            qos_profile=1,
            callback_group=io_group,
        )
        self.subscription_color_sensor = self.create_subscription(
            msg_type=Color,
            topic=f'/{turtle_name}/color_sensor',
            callback=self.on_color,
            qos_profile=1,
            callback_group=io_group,
        )

        self.publisher_vel = self.create_publisher(
            msg_type=Twist,
            topic=f'/{turtle_name}/cmd_vel',
            qos_profile=1,
        )

        self.action_server = ActionServer(
            node=self,
            action_type=FindHiddenGift,
            action_name='/find_hidden_gift',
            execute_callback=self.execute_callback,
            callback_group=action_group,
        )

    # --- utils ---

    def _sleep(self, seconds: float) -> Future:
        """
        Non-blocking sleep for rclpy coroutines.
        The executor progresses the coroutine while timers fire.
        """
        fut = Future()
        timer_holder: dict[str, Timer] = {}

        def _cb():
            if not fut.done():
                fut.set_result(True)

            t = timer_holder.get("t")

            if t is not None:
                t.cancel()

        timer = self.create_timer(seconds, _cb)
        timer_holder["t"] = timer

        return fut

    def _yield(self) -> Future:
        return self._sleep(YIELD_SLEEP_TIME)

    # --- lol ---

    async def _disable_pen(self):
        if not self.pen_client.wait_for_service(timeout_sec=10.0):
            raise ValueError("`/set_pen` client server didn't answer")

        req = SetPen.Request()
        req.off = 1
        fut = self.pen_client.call_async(req)

        await fut

    # --- subscribers ---

    def on_pose(self, msg: Pose) -> None:
        # print(f"on_pose: {msg}")
        self.state.pose = msg
        
        if not self._pose_ready.done():
            self._pose_ready.set_result(True)

    def on_color(self, msg: Color) -> None:
        # print(f"on_color: {msg}")
        self.state.color = msg

        if not self._color_ready.done():
            self._color_ready.set_result(True)
    
    async def _wait_state(self) -> None:
        if not self._pose_ready.done():
            await self._pose_ready
        if not self._color_ready.done():
            await self._color_ready

    # --- checkers ---

    def _is_at_position(self, x: float, y: float, tol: float = 0.05):
        p = self.state.pose

        return (
            p is not None
            and math.isclose(p.x, float(x), abs_tol=tol)
            and math.isclose(p.y, float(y), abs_tol=tol)
        )

    def _is_within_search_area(self, search_area: Rect) -> bool:
        p = self.state.pose

        return (
            p is not None
            and p.x >= search_area.bottom_left.x
            and p.x <= search_area.top_right.x
            and p.y >= search_area.bottom_left.y
            and p.y <= search_area.top_right.y
        )

    def _is_present(self) -> bool:
        c = self.state.color

        if c is None:
            return False

        return (
            c is not None
            and c.r == 0 
            and c.g == 255
            and c.b == 0
        )

    # --- motion ---

    def _stop_moving(self) -> None:
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0

        self.publisher_vel.publish(msg)

    async def _move_forward(self, move_speed, move_time: float | int = 0) -> None:
        msg = Twist()
        msg.linear.x = float(move_speed)
        msg.angular.z = 0.0

        self.publisher_vel.publish(msg)

        if move_time > 0:
            await self._sleep(float(move_time))
            self._stop_moving()

    async def _rotate_to(self, target_theta: float, tol: float = 0.01, k: float = 4.0, max_w: float = 5.0):
        target_theta = float(target_theta) % (2 * math.pi)

        self._stop_moving()

        msg = Twist()
        msg.linear.x = 0.0

        while True:
            await self._yield()

            err = angle_diff(target_theta, self.state.pose.theta)

            if abs(err) <= tol:
                msg.angular.z = 0.0
                self.publisher_vel.publish(msg)
                return

            w = k * err
            w = max(-max_w, min(max_w, w))
            msg.angular.z = w

            self.publisher_vel.publish(msg)

    async def _go_to_point(
        self,
        target_point: PoseXY,
        *,
        max_v: float = 5.0,
        k_lin: float = 1.0,
        k_ang: float = 5.0,
        tol: float = 0.05,
        max_w: float = 5.0,
    ) -> None:
        msg = Twist()

        while True:
            await self._yield()

            p = self.state.pose
            dx = target_point.x - p.x
            dy = target_point.y - p.y
            dist = math.hypot(dx, dy)

            if dist <= tol:
                self._stop_moving()
                return

            desired = math.atan2(dy, dx)
            err = angle_diff(desired, p.theta)

            msg.angular.z = float(max(-max_w, min(max_w, k_ang * err)))

            if abs(err) < 0.3:
                msg.linear.x = float(min(max_v, k_lin * dist))
            else:
                msg.linear.x = 0.0

            self.publisher_vel.publish(msg)

    # --- publisher ---
    async def _check_found(self, search_area) -> bool:
        if self._is_within_search_area(search_area) and self._is_present():
            self._stop_moving()
            await self._yield()

            return self._is_within_search_area(search_area) and self._is_present()

        return False


    async def _find_present(self, search_area: Rect, move_speed: float = 5.0, turning_time: float = 0.15) -> bool:
        move_speed = float(move_speed)

        if await self._check_found(search_area):
            return True

        await self._go_to_point(PoseXY(0, 0))
        # await self._go_to_point(search_area.bottom_left)

        if await self._check_found(search_area):
            return True

        # bottom = search_area.bottom_left.y
        # top = search_area.top_right.y
        # right = search_area.top_right.x
        bottom = 0
        top = 11
        right = 11

        await self._rotate_to(math.pi / 2)  # face up
        up = True

        while True:
            await self._yield()

            if await self._check_found(search_area):
                return True

            p = self.state.pose

            if up and p.y >= top:
                up = False

                await self._rotate_to(Turns.right.value)
                await self._move_forward(move_speed, turning_time)
                if await self._check_found(search_area):
                    return True
                await self._rotate_to(Turns.down.value)

            if not up and p.y <= bottom:
                up = True

                await self._rotate_to(Turns.right.value)
                await self._move_forward(move_speed, turning_time)
                if await self._check_found(search_area):
                    return True
                await self._rotate_to(Turns.up.value)

                if p.x >= right:
                    break

            await self._move_forward(move_speed)

        self._stop_moving()
        return False

    async def _check_out_direction(self, direction: Turns, move_speed: float = 2.0) -> Pose:
        move_speed = float(move_speed)

        if not self._is_present():
            raise ValueError("Ended search at no present")

        await self._rotate_to(direction.value)
        
        while self._is_present():
            await self._move_forward(move_speed)
            await self._yield()

        self._stop_moving()
        await self._rotate_to(math.pi + direction.value)

        # while not self._is_present():
        for i in range(10):
            if not self._is_present():
                await self._move_forward(0.1)
                await self._yield()

        p = self.state.pose

        await self._move_forward(0.5, 0.5)

        return p

    async def _determine_present(self, move_speed: float = 2.0) -> PoseXY:
        move_speed = float(move_speed)

        left = await self._check_out_direction(Turns.left, move_speed)
        right = await self._check_out_direction(Turns.right, move_speed)
        up = await self._check_out_direction(Turns.up, move_speed)
        down = await self._check_out_direction(Turns.down, move_speed)

        # print("Initial:", self.state.pose.x, self.state.pose.y, self._is_present())
        # print("left:", left.x, left.y, self._is_present())
        # print("right:", right.x, right.y, self._is_present())
        # print("up:", up.x, up.y, self._is_present())
        # print("down:", down.x, down.y, self._is_present())

        return PoseXY((left.x + right.x) / 2.0, (down.y + up.y) / 2.0)

    async def execute_callback(self, goal_handle):
        await self._disable_pen()
        await self._wait_state()

        goal: FindHiddenGift.Goal = goal_handle.request
        search_area = goal.search_area

        found = await self._find_present(search_area)

        feedback = FindHiddenGift.Feedback()
        feedback.gift_ever_detected = found
        goal_handle.publish_feedback(feedback)

        result = FindHiddenGift.Result()
        result.gift_found = found
        result.gift_coordinates.x = 0.0
        result.gift_coordinates.y = 0.0

        if found:
            xy = await self._determine_present()
            result.gift_coordinates.x = xy.x
            result.gift_coordinates.y = xy.y

        goal_handle.succeed()

        return result


def main(): 
    rclpy.init()
    node = FindHiddenGiftNode()

    # executor = MultiThreadedExecutor(num_threads=1) # NOTE
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.spin()

    node.destroy_node()
    rclpy.shutdown()

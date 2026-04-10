#!/usr/bin/env python3
"""
ROS-side controller:
  - subscribes plan/odom/scan/pointcloud
  - builds observation
  - calls policy_infer.py for model inference
  - publishes model output directly to /cmd_vel
"""

import math
import os
import sys
from collections import deque

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Float32MultiArray, String

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if SCRIPT_DIR not in sys.path:
    # Put source scripts path first to avoid importing catkin wrapper scripts
    # under devel/lib/me5413_world (same module name, no symbols exported).
    sys.path.insert(0, SCRIPT_DIR)

from policy_infer import PolicyInfer  # noqa: E402


def clamp(value, low, high):
    return max(low, min(high, value))


def yaw_from_quat(q):
    # yaw from quaternion (x, y, z, w)
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class RlController:
    def __init__(self):
        # 读取话题名参数
        self.plan_topic = rospy.get_param("~plan_topic", "/move_base/NavfnROS/plan")
        self.odom_topic = rospy.get_param("~odom_topic", "/odometry/filtered")
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.cloud_topic = rospy.get_param("~cloud_topic", "/mid/points")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")

        # 读取控制器参数
        self.rate_hz = rospy.get_param("~rate_hz", 10.0)
        self.lookahead = rospy.get_param("~lookahead", 0.8)
        self.max_lin = rospy.get_param("~max_lin", 0.6)
        self.max_ang = rospy.get_param("~max_ang", 1.2)
        self.goal_tolerance = rospy.get_param("~goal_tolerance", 0.35)
        self.stop_distance = rospy.get_param("~stop_distance", 0.8)
        self.front_angle_deg = rospy.get_param("~front_angle_deg", 35.0)
        self.self_min_range = rospy.get_param("~self_min_range", 0.30)
        self.frame_stack = int(rospy.get_param("~frame_stack", 1))

        # Pure RL policy selection
        self.policy_backend = rospy.get_param("~policy_backend", "torchscript")
        self.policy_model_path = rospy.get_param("~policy_model_path", "")

        # 初始化变量
        self.plan = None
        self.odom = None
        self.scan_front = None
        self.scan_left = None
        self.scan_right = None
        self.cloud_front = None
        self.cloud_left = None
        self.cloud_right = None

        # 初始化策略推理器
        self.policy = PolicyInfer(
            backend=self.policy_backend,
            model_path=self.policy_model_path,
            max_lin=self.max_lin,
            max_ang=self.max_ang,
            frame_stack=self.frame_stack,
        )
        if not self.policy.ready:
            rospy.logfatal(
                "Policy is not ready. backend=%s model=%s",
                self.policy_backend,
                self.policy_model_path if self.policy_model_path else "<none>",
            )
            raise RuntimeError("RL policy load failed")

        self.obs_history = deque(maxlen=self.frame_stack)

        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=10)
        self.debug_obs_pub = rospy.Publisher(
            "/rl_controller/debug_obs", Float32MultiArray, queue_size=10
        )
        self.debug_action_pub = rospy.Publisher(
            "/rl_controller/debug_action", Float32MultiArray, queue_size=10
        )
        self.debug_target_pub = rospy.Publisher(
            "/rl_controller/debug_target", Float32MultiArray, queue_size=10
        )
        self.debug_state_pub = rospy.Publisher(
            "/rl_controller/debug_state", String, queue_size=10
        )
        # 订阅全局路经/里程计/激光数据/点云数据
        rospy.Subscriber(self.plan_topic, Path, self.plan_cb, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=1)
        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_cb, queue_size=1)
        rospy.Subscriber(self.cloud_topic, PointCloud2, self.cloud_cb, queue_size=1)
        # 定时器&设置一个关闭回调
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.timer_cb)
        rospy.on_shutdown(self.stop_robot)

        rospy.loginfo(
            "rl_controller started: backend=%s model=%s plan=%s odom=%s scan=%s cloud=%s cmd=%s",
            self.policy_backend,
            self.policy_model_path if self.policy_model_path else "<none>",
            self.plan_topic,
            self.odom_topic,
            self.scan_topic,
            self.cloud_topic,
            self.cmd_vel_topic,
        )

    def plan_cb(self, msg):
        self.plan = msg

    def odom_cb(self, msg):
        self.odom = msg

    def scan_cb(self, msg):
        front = []
        left = []
        right = []
        front_rad = math.radians(self.front_angle_deg)


        for i, r in enumerate(msg.ranges):
            if math.isinf(r) or math.isnan(r):
                continue
            if r < msg.range_min or r > msg.range_max:
                continue
            # Ignore very close returns from robot body/noise.
            if r < self.self_min_range:
                continue
            angle = msg.angle_min + i * msg.angle_increment
            if abs(angle) <= front_rad:
                front.append(r)
            elif 0.0 < angle <= math.radians(90.0):
                left.append(r)
            elif -math.radians(90.0) <= angle < 0.0:
                right.append(r)

        self.scan_front = min(front) if front else None
        self.scan_left = min(left) if left else None
        self.scan_right = min(right) if right else None

    def cloud_cb(self, msg):
        front = []
        left = []
        right = []
        front_rad = math.radians(self.front_angle_deg)

        try:
            for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                x, y, _ = p
                if x <= 0.0:
                    continue
                dist = math.hypot(x, y)
                if dist < self.self_min_range:
                    continue
                angle = math.atan2(y, x)
                if abs(angle) <= front_rad:
                    front.append(dist)
                elif 0.0 < angle <= math.radians(90.0):
                    left.append(dist)
                elif -math.radians(90.0) <= angle < 0.0:
                    right.append(dist)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "cloud parse failed: %s", str(exc))
            return

        self.cloud_front = min(front) if front else None
        self.cloud_left = min(left) if left else None
        self.cloud_right = min(right) if right else None

    def nearest_obstacle(self):
        front = self.scan_front if self.scan_front is not None else self.cloud_front
        left = self.scan_left if self.scan_left is not None else self.cloud_left
        right = self.scan_right if self.scan_right is not None else self.cloud_right
        return front, left, right

    def _build_obs_vector(self, obs_dict):
        base = [
            float(obs_dict.get("dist_goal", 0.0)),
            float(obs_dict.get("yaw_err", 0.0)),
            float(obs_dict.get("front", 10.0)),
            float(obs_dict.get("left", 10.0)),
            float(obs_dict.get("right", 10.0)),
            float(obs_dict.get("v_curr", 0.0)),
            float(obs_dict.get("w_curr", 0.0)),
        ]
        if len(self.obs_history) == 0:
            for _ in range(self.frame_stack):
                self.obs_history.append(list(base))
        else:
            self.obs_history.append(list(base))

        stacked = []
        for f in self.obs_history:
            stacked.extend(f)
        return stacked

    def get_current_pose(self):
        if self.odom is None:
            return None
        pose = self.odom.pose.pose
        x = pose.position.x
        y = pose.position.y
        yaw = yaw_from_quat(pose.orientation)
        v_curr = self.odom.twist.twist.linear.x
        w_curr = self.odom.twist.twist.angular.z
        return x, y, yaw, v_curr, w_curr

    # 从全局路径中选取一个局部跟踪点
    def pick_target(self, x, y):
        if self.plan is None or len(self.plan.poses) == 0:
            return None, None
        target = None
        for pose_stamped in self.plan.poses:
            p = pose_stamped.pose.position
            if math.hypot(p.x - x, p.y - y) >= self.lookahead:
                target = p
                break
        if target is None:
            raise RuntimeError(
                "No lookahead target found on current plan. "
                "Please increase path density or reduce lookahead."
            )
        final_goal = self.plan.poses[-1].pose.position
        return target, final_goal

    def compute_cmd(self):
        pose = self.get_current_pose()
        if pose is None:
            self.debug_state_pub.publish(String(data="wait_odom"))
            self.debug_action_pub.publish(
                Float32MultiArray(data=[0.0, 0.0, 0.0, -1.0, 0.0, 10.0, 10.0, 10.0])
            )
            return 0.0, 0.0
        x, y, yaw, v_curr, w_curr = pose

        target, final_goal = self.pick_target(x, y)
        if target is None:
            self.debug_state_pub.publish(String(data="wait_plan_or_goal"))
            self.debug_target_pub.publish(
                Float32MultiArray(
                    data=[
                        float(x),
                        float(y),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float(self.lookahead),
                    ]
                )
            )
            self.debug_action_pub.publish(
                Float32MultiArray(data=[0.0, 0.0, 0.0, -2.0, 0.0, 10.0, 10.0, 10.0])
            )
            return 0.0, 0.0

        dist_goal = math.hypot(final_goal.x - x, final_goal.y - y)
        if dist_goal < self.goal_tolerance:
            self.debug_state_pub.publish(String(data="goal_reached"))
            self.debug_action_pub.publish(
                Float32MultiArray(data=[0.0, 0.0, 0.0, dist_goal, 0.0, 10.0, 10.0, 10.0])
            )
            return 0.0, 0.0

        heading = math.atan2(target.y - y, target.x - x)
        yaw_err = math.atan2(math.sin(heading - yaw), math.cos(heading - yaw))
        front, left, right = self.nearest_obstacle()

        obs = {
            "dist_goal": dist_goal,
            "yaw_err": yaw_err,
            "front": float(front if front is not None else 10.0),
            "left": float(left if left is not None else 10.0),
            "right": float(right if right is not None else 10.0),
            "v_curr": float(v_curr),
            "w_curr": float(w_curr),
        }

        obs_vec = self._build_obs_vector(obs)
        self.debug_obs_pub.publish(Float32MultiArray(data=obs_vec))
        model_cmd = self.policy.predict(obs_vec)
        if model_cmd is None:
            self.debug_state_pub.publish(String(data="policy_infer_failed"))
            rospy.logwarn_throttle(2.0, "Policy inference returned None, stopping robot.")
            self.debug_action_pub.publish(
                Float32MultiArray(
                    data=[
                        -1.0,  # valid flag
                        0.0,   # lin
                        0.0,   # ang
                        obs["dist_goal"],
                        obs["yaw_err"],
                        obs["front"],
                        obs["left"],
                        obs["right"],
                    ]
                )
            )
            return 0.0, 0.0

        lin, ang = model_cmd
        self.debug_state_pub.publish(String(data="policy_ok"))
        self.debug_target_pub.publish(
            Float32MultiArray(
                data=[
                    float(x),
                    float(y),
                    float(target.x),
                    float(target.y),
                    float(final_goal.x),
                    float(final_goal.y),
                    float(self.lookahead),
                ]
            )
        )
        # Hard safety brake only (not rule steering).
        if front is not None and front < self.stop_distance:
            lin = min(0.0, lin)

        lin = clamp(lin, -self.max_lin, self.max_lin)
        ang = clamp(ang, -self.max_ang, self.max_ang)
        self.debug_action_pub.publish(
            Float32MultiArray(
                data=[
                    1.0,  # valid flag
                    float(lin),
                    float(ang),
                    obs["dist_goal"],
                    obs["yaw_err"],
                    obs["front"],
                    obs["left"],
                    obs["right"],
                ]
            )
        )
        rospy.loginfo_throttle(
            1.0,
            "RL dbg | v=%.3f w=%.3f | dist=%.2f yaw_err=%.2f | front=%.2f left=%.2f right=%.2f",
            lin,
            ang,
            obs["dist_goal"],
            obs["yaw_err"],
            obs["front"],
            obs["left"],
            obs["right"],
        )
        return lin, ang

    def timer_cb(self, _):
        lin, ang = self.compute_cmd()
        cmd = Twist()
        cmd.linear.x = lin
        cmd.angular.z = ang
        self.cmd_pub.publish(cmd)

    def stop_robot(self):
        cmd = Twist()
        self.cmd_pub.publish(cmd)


if __name__ == "__main__":
    rospy.init_node("rl_controller") # 初始化 ROS 节点，节点名叫 rl_controller
    RlController()
    rospy.spin()

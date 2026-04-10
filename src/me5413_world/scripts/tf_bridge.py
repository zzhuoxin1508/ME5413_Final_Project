#!/usr/bin/env python3
"""TF Bridge: publishes map->odom transform with smooth blending.

Replaces AMCL's built-in TF broadcast. On the ramp (when /tf_bridge/on_ramp
is True), the transform is derived purely from odometry. After the ramp,
it gradually blends from odom-based to AMCL-based over BLEND_DURATION seconds.

This prevents the "teleport" problem where AMCL suddenly jumps after the
robot leaves the ramp and enters the upper floor.
"""
import math
import rospy
import tf2_ros
import tf.transformations as tft
from geometry_msgs.msg import TransformStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool


class TFBridge:
    def __init__(self):
        rospy.init_node("tf_bridge")

        self.BLEND_DURATION = rospy.get_param("~blend_duration", 20.0)
        self.MAX_JUMP = rospy.get_param("~max_jump", 0.3)
        self.rate = rospy.get_param("~rate", 50.0)

        self.br = tf2_ros.TransformBroadcaster()

        # Current map->odom transform (as x, y, yaw)
        self.tf_x = 0.0
        self.tf_y = 0.0
        self.tf_yaw = 0.0

        # AMCL's suggested map->odom (computed from amcl_pose and odom)
        self.amcl_tf_x = 0.0
        self.amcl_tf_y = 0.0
        self.amcl_tf_yaw = 0.0
        self.amcl_received = False

        # Odom pose
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0

        # State
        self.on_ramp = False
        self.blend_start_time = None
        self.initialized = False

        rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.amcl_cb, queue_size=1)
        rospy.Subscriber("/odometry/filtered", Odometry, self.odom_cb, queue_size=1)
        rospy.Subscriber("/tf_bridge/on_ramp", Bool, self.ramp_cb, queue_size=1)

        rospy.Timer(rospy.Duration(1.0 / self.rate), self.publish_tf)
        rospy.loginfo("tf_bridge started (blend_duration=%.1fs, max_jump=%.2fm)",
                      self.BLEND_DURATION, self.MAX_JUMP)

    def quat_to_yaw(self, q):
        _, _, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
        return yaw

    def odom_cb(self, msg):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        self.odom_yaw = self.quat_to_yaw(msg.pose.pose.orientation)

    def amcl_cb(self, msg):
        # AMCL gives us the robot pose in map frame.
        # map->odom = amcl_pose * inv(odom_pose)
        amcl_x = msg.pose.pose.position.x
        amcl_y = msg.pose.pose.position.y
        amcl_yaw = self.quat_to_yaw(msg.pose.pose.orientation)

        # map->odom transform: difference between AMCL pose (in map) and odom pose
        new_tf_x = amcl_x - self.odom_x
        new_tf_y = amcl_y - self.odom_y
        new_tf_yaw = self._normalize_angle(amcl_yaw - self.odom_yaw)

        # Check jump
        if self.amcl_received:
            dx = new_tf_x - self.amcl_tf_x
            dy = new_tf_y - self.amcl_tf_y
            jump = math.hypot(dx, dy)
            if jump > self.MAX_JUMP and self.initialized:
                rospy.logwarn("tf_bridge: AMCL TF jump rejected: %.2fm (max %.2fm)", jump, self.MAX_JUMP)
                return

        self.amcl_tf_x = new_tf_x
        self.amcl_tf_y = new_tf_y
        self.amcl_tf_yaw = new_tf_yaw
        self.amcl_received = True

        if not self.initialized:
            # First AMCL update: snap to it
            self.tf_x = new_tf_x
            self.tf_y = new_tf_y
            self.tf_yaw = new_tf_yaw
            self.initialized = True
            rospy.loginfo("tf_bridge: initialized map->odom from AMCL")

    def ramp_cb(self, msg):
        was_on_ramp = self.on_ramp
        self.on_ramp = msg.data
        if was_on_ramp and not self.on_ramp:
            # Just left ramp: start blending
            self.blend_start_time = rospy.Time.now()
            rospy.loginfo("tf_bridge: leaving ramp, starting %.1fs blend to AMCL", self.BLEND_DURATION)
        elif not was_on_ramp and self.on_ramp:
            self.blend_start_time = None
            rospy.loginfo("tf_bridge: entering ramp, freezing map->odom TF")

    def _normalize_angle(self, a):
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a

    def _get_blend_alpha(self):
        """0.0 = keep current (odom-based), 1.0 = fully AMCL."""
        if self.on_ramp:
            return 0.0
        if self.blend_start_time is None:
            return 1.0
        elapsed = (rospy.Time.now() - self.blend_start_time).to_sec()
        return min(1.0, elapsed / self.BLEND_DURATION)

    def publish_tf(self, _):
        if not self.initialized:
            return

        # Blend towards AMCL's suggested TF
        alpha = self._get_blend_alpha()
        if alpha > 0.0 and self.amcl_received:
            self.tf_x += alpha * 0.02 * (self.amcl_tf_x - self.tf_x)
            self.tf_y += alpha * 0.02 * (self.amcl_tf_y - self.tf_y)
            dyaw = self._normalize_angle(self.amcl_tf_yaw - self.tf_yaw)
            self.tf_yaw += alpha * 0.02 * dyaw

        # Publish
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "map"
        t.child_frame_id = "odom"
        t.transform.translation.x = self.tf_x
        t.transform.translation.y = self.tf_y
        t.transform.translation.z = 0.0
        q = tft.quaternion_from_euler(0, 0, self.tf_yaw)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.br.sendTransform(t)


if __name__ == "__main__":
    TFBridge()
    rospy.spin()

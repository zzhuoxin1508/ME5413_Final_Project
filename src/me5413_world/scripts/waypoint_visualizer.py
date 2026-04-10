#!/usr/bin/env python3
"""Visualize all mission waypoints as RViz markers.

Reads waypoint params from mission_manager namespace and publishes
visualization_msgs/MarkerArray on /mission/waypoint_markers.
Add this topic in RViz as MarkerArray to see all waypoints.
"""
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from tf.transformations import quaternion_from_euler


def make_sphere(ns, idx, x, y, color, scale=0.4):
    m = Marker()
    m.header.frame_id = "map"
    m.header.stamp = rospy.Time.now()
    m.ns = ns
    m.id = idx
    m.type = Marker.SPHERE
    m.action = Marker.ADD
    m.pose.position.x = x
    m.pose.position.y = y
    m.pose.position.z = 0.1
    m.pose.orientation.w = 1.0
    m.scale.x = m.scale.y = m.scale.z = scale
    m.color.r, m.color.g, m.color.b, m.color.a = color
    return m


def make_arrow(ns, idx, x, y, yaw, color):
    m = Marker()
    m.header.frame_id = "map"
    m.header.stamp = rospy.Time.now()
    m.ns = ns + "_arrow"
    m.id = idx
    m.type = Marker.ARROW
    m.action = Marker.ADD
    m.pose.position.x = x
    m.pose.position.y = y
    m.pose.position.z = 0.15
    qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)
    m.pose.orientation.x = qx
    m.pose.orientation.y = qy
    m.pose.orientation.z = qz
    m.pose.orientation.w = qw
    m.scale.x = 0.6
    m.scale.y = 0.08
    m.scale.z = 0.08
    m.color.r, m.color.g, m.color.b, m.color.a = color
    return m


def make_text(ns, idx, x, y, text, color):
    m = Marker()
    m.header.frame_id = "map"
    m.header.stamp = rospy.Time.now()
    m.ns = ns + "_text"
    m.id = idx
    m.type = Marker.TEXT_VIEW_FACING
    m.action = Marker.ADD
    m.pose.position.x = x
    m.pose.position.y = y
    m.pose.position.z = 0.7
    m.pose.orientation.w = 1.0
    m.scale.z = 0.35
    m.color.r, m.color.g, m.color.b, m.color.a = color
    m.text = text
    return m


def make_line_strip(ns, points, color, width=0.06):
    m = Marker()
    m.header.frame_id = "map"
    m.header.stamp = rospy.Time.now()
    m.ns = ns + "_line"
    m.id = 0
    m.type = Marker.LINE_STRIP
    m.action = Marker.ADD
    m.scale.x = width
    m.color.r, m.color.g, m.color.b, m.color.a = color
    m.pose.orientation.w = 1.0
    for p in points:
        pt = Point()
        pt.x, pt.y, pt.z = float(p[0]), float(p[1]), 0.1
        m.points.append(pt)
    return m


def build_markers():
    arr = MarkerArray()

    routes = [
        ("lower", "/mission_manager/lower_waypoints",
         (0.1, 0.7, 1.0, 1.0)),   # blue
        ("ramp", "/mission_manager/ramp_waypoints",
         (1.0, 0.6, 0.1, 1.0)),   # orange
        ("upper", "/mission_manager/upper_waypoints",
         (0.2, 1.0, 0.2, 1.0)),   # green
    ]

    for ns, param, color in routes:
        wpts = rospy.get_param(param, [])
        if not wpts:
            continue
        for i, p in enumerate(wpts):
            if len(p) < 3:
                continue
            x, y, yaw = float(p[0]), float(p[1]), float(p[2])
            arr.markers.append(make_sphere(ns, i, x, y, color))
            arr.markers.append(make_arrow(ns, i, x, y, yaw, color))
            arr.markers.append(make_text(ns, i, x, y, "{}_{}".format(ns, i), (1, 1, 1, 1)))
        arr.markers.append(make_line_strip(ns, wpts, color))

    # exit goal (yellow)
    exit_goal = rospy.get_param("/mission_manager/exit_goal", None)
    if exit_goal and len(exit_goal) >= 3:
        x, y, yaw = float(exit_goal[0]), float(exit_goal[1]), float(exit_goal[2])
        arr.markers.append(make_sphere("exit", 0, x, y, (1.0, 1.0, 0.0, 1.0), scale=0.5))
        arr.markers.append(make_arrow("exit", 0, x, y, yaw, (1.0, 1.0, 0.0, 1.0)))
        arr.markers.append(make_text("exit", 0, x, y, "exit_goal", (1, 1, 0, 1)))

    # final goals (red)
    final_goals = rospy.get_param("/mission_manager/final_goals_by_box", [])
    for item in final_goals:
        if len(item) < 4:
            continue
        bid = int(item[0])
        x, y, yaw = float(item[1]), float(item[2]), float(item[3])
        arr.markers.append(make_sphere("final", bid, x, y, (1.0, 0.1, 0.1, 1.0), scale=0.5))
        arr.markers.append(make_arrow("final", bid, x, y, yaw, (1.0, 0.1, 0.1, 1.0)))
        arr.markers.append(make_text("final", bid, x, y, "final_box{}".format(bid), (1, 0.6, 0.6, 1)))

    return arr


def main():
    rospy.init_node("waypoint_visualizer")
    pub = rospy.Publisher("/mission/waypoint_markers", MarkerArray, queue_size=1, latch=True)

    # wait briefly so params from mission_manager are loaded
    rospy.sleep(1.0)

    rate = rospy.Rate(1.0)
    while not rospy.is_shutdown():
        arr = build_markers()
        if arr.markers:
            pub.publish(arr)
        rate.sleep()


if __name__ == "__main__":
    main()

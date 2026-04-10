#!/usr/bin/env python3
"""Evaluate waypoint-following performance from a rosbag.

Parses NAV_GOAL / STATE_* entries from /rosout written by mission_manager.py,
and verifies whether the robot actually reached each goal using the ground
truth trajectory from /gazebo/ground_truth/state.

Usage:
    python3 scripts/eval_waypoints.py eval_run.bag
    python3 scripts/eval_waypoints.py eval_run.bag --tolerance 0.45 --gt-topic /gazebo/ground_truth/state
"""
import argparse
import math
import re
from typing import List, Optional, Tuple

import rosbag


NAV_GOAL_RE = re.compile(
    r"NAV_GOAL name=(?P<name>\S+) x=(?P<x>-?[0-9.]+) y=(?P<y>-?[0-9.]+) yaw=(?P<yaw>-?[0-9.]+)"
)
NAV_RETRY_RE = re.compile(r"NAV_RETRY name=(?P<name>\S+) retry=(?P<retry>\d+)")
STATE_ENTER_RE = re.compile(r"STATE_ENTER (?P<state>\S+) t=(?P<t>[0-9.]+)")
COUNT_UPDATE_RE = re.compile(r"COUNT_UPDATE box=(?P<box>\d+) total=(?P<total>-?\d+)")
FINAL_DECISION_RE = re.compile(r"FINAL_DECISION box=(?P<box>\d+) count=(?P<count>-?\d+)")


def evaluate(bag_path: str, tolerance: float, gt_topic: str) -> None:
    goals: List[dict] = []
    retries: List[Tuple[float, str, int]] = []
    states: List[Tuple[float, str]] = []
    box_counts: List[Tuple[float, int, int]] = []
    final_decision: Optional[Tuple[int, int]] = None

    traj: List[Tuple[float, float, float]] = []  # (t, x, y)

    path_length = 0.0
    last_xy: Optional[Tuple[float, float]] = None

    with rosbag.Bag(bag_path, "r") as bag:
        topics_in_bag = set(bag.get_type_and_topic_info()[1].keys())
        if gt_topic not in topics_in_bag:
            print(f"[warn] ground-truth topic {gt_topic} not in bag. "
                  f"Available topics: {sorted(topics_in_bag)}")
        for topic, msg, t in bag.read_messages(topics=["/rosout", gt_topic]):
            ts = t.to_sec()
            if topic == "/rosout":
                text = getattr(msg, "msg", "")

                m = NAV_GOAL_RE.search(text)
                if m:
                    goals.append(
                        {
                            "t_sent": ts,
                            "name": m.group("name"),
                            "x": float(m.group("x")),
                            "y": float(m.group("y")),
                            "yaw": float(m.group("yaw")),
                            "reached": False,
                            "min_dist": float("inf"),
                            "t_reached": None,
                        }
                    )
                    continue

                m = NAV_RETRY_RE.search(text)
                if m:
                    retries.append((ts, m.group("name"), int(m.group("retry"))))
                    continue

                m = STATE_ENTER_RE.search(text)
                if m:
                    states.append((ts, m.group("state")))
                    continue

                m = COUNT_UPDATE_RE.search(text)
                if m:
                    box_counts.append((ts, int(m.group("box")), int(m.group("total"))))
                    continue

                m = FINAL_DECISION_RE.search(text)
                if m:
                    final_decision = (int(m.group("box")), int(m.group("count")))
                    continue

            elif topic == gt_topic:
                x = msg.pose.pose.position.x
                y = msg.pose.pose.position.y
                traj.append((ts, x, y))
                if last_xy is not None:
                    path_length += math.hypot(x - last_xy[0], y - last_xy[1])
                last_xy = (x, y)

    if not goals:
        print("[error] no NAV_GOAL entries in /rosout — is this the right bag?")
        return

    # assign per-goal active time window: from t_sent to the next goal's t_sent
    for i, g in enumerate(goals):
        t_start = g["t_sent"]
        t_end = goals[i + 1]["t_sent"] if i + 1 < len(goals) else float("inf")

        if traj:
            for (ts, x, y) in traj:
                if ts < t_start:
                    continue
                if ts > t_end:
                    break
                d = math.hypot(g["x"] - x, g["y"] - y)
                if d < g["min_dist"]:
                    g["min_dist"] = d
                    if not g["reached"] and d < tolerance:
                        g["reached"] = True
                        g["t_reached"] = ts

    reached = [g for g in goals if g["reached"]]
    missed = [g for g in goals if not g["reached"]]

    # ---- Report ----
    print("=" * 60)
    print(f"Bag              : {bag_path}")
    print(f"GT topic         : {gt_topic}  ({len(traj)} samples)")
    print(f"Tolerance        : {tolerance:.2f} m")
    print("=" * 60)
    print(f"Total goals sent : {len(goals)}")
    print(f"Reached          : {len(reached)}  ({100.0*len(reached)/len(goals):.1f}%)")
    print(f"Missed           : {len(missed)}")
    print(f"Retries          : {len(retries)}")
    print(f"Path length (GT) : {path_length:.2f} m")

    # time stats from state machine
    state_times = {s: t for t, s in states}
    if "SCAN_LOWER" in state_times:
        t0 = state_times["SCAN_LOWER"]
        t_end = None
        for s in ("DONE", "FAIL"):
            if s in state_times:
                t_end = state_times[s]
                break
        if t_end is not None:
            print(f"Mission time     : {t_end - t0:.2f} s "
                  f"({'SUCCESS' if 'DONE' in state_times else 'FAILED'})")

    if box_counts or final_decision:
        print("-" * 60)
        if box_counts:
            latest = {}
            for ts, bid, total in box_counts:
                latest[bid] = total
            print("Box counts       :", {k: latest[k] for k in sorted(latest)})
        if final_decision:
            print(f"Final decision   : box={final_decision[0]} count={final_decision[1]}")

    print("-" * 60)
    print("Per-goal details:")
    print(f"  {'idx':>3} {'name':<16} {'x':>8} {'y':>8} {'min_d':>7} {'reach':>6}")
    for i, g in enumerate(goals):
        flag = "OK" if g["reached"] else "MISS"
        min_d = g["min_dist"] if math.isfinite(g["min_dist"]) else float("nan")
        print(f"  {i:>3} {g['name']:<16} {g['x']:>8.2f} {g['y']:>8.2f} "
              f"{min_d:>7.2f} {flag:>6}")

    if missed:
        print("-" * 60)
        print("Missed goals:")
        for g in missed:
            min_d = g["min_dist"] if math.isfinite(g["min_dist"]) else float("nan")
            print(f"  - {g['name']} ({g['x']:.2f},{g['y']:.2f})  closest {min_d:.2f} m")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate waypoint reach rate from a rosbag.")
    parser.add_argument("bag", help="Path to evaluation rosbag")
    parser.add_argument("--tolerance", type=float, default=0.45,
                        help="Reach tolerance in meters (default: 0.45, matches mission_manager)")
    parser.add_argument("--gt-topic", default="/gazebo/ground_truth/state",
                        help="Ground truth odometry topic (default: /gazebo/ground_truth/state)")
    args = parser.parse_args()
    evaluate(args.bag, args.tolerance, args.gt_topic)


if __name__ == "__main__":
    main()

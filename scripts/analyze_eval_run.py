#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
from typing import Dict, Optional

import rosbag


STATE_ENTER_SCAN_LOWER_RE = re.compile(r"STATE_ENTER SCAN_LOWER t=([0-9.]+)")
STATE_ENTER_DONE_RE = re.compile(r"STATE_ENTER DONE t=([0-9.]+)")
STATE_ENTER_FAIL_RE = re.compile(r"STATE_ENTER FAIL t=([0-9.]+)")


def safe_float(v: Optional[float]) -> str:
    if v is None:
        return ""
    return f"{v:.3f}"


def parse_run_id_from_bag(bag_path: str) -> str:
    base = os.path.basename(bag_path)
    # e.g., run_01.bag -> run_01
    if base.endswith(".bag"):
        return base[:-4]
    return base


def analyze_bag(bag_path: str) -> Dict[str, object]:
    result = {
        "run_id": parse_run_id_from_bag(bag_path),
        "success": 0,
        "final_state": "UNKNOWN",
        "total_time_s": None,
        "nav_retry_count": 0,
        "failed_plan_count": 0,
        "path_length_m": 0.0,
        "min_scan_range_m": None,
    }

    scan_enter_elapsed = None
    done_elapsed = None
    fail_elapsed = None
    scan_enter_stamp = None
    done_stamp = None
    fail_stamp = None

    last_x = None
    last_y = None

    with rosbag.Bag(bag_path, "r") as bag:
        for topic, msg, t in bag.read_messages(
            topics=["/rosout", "/odometry/filtered", "/front/scan"]
        ):
            ts = t.to_sec()

            if topic == "/rosout":
                text = getattr(msg, "msg", "")

                m = STATE_ENTER_SCAN_LOWER_RE.search(text)
                if m and scan_enter_elapsed is None:
                    scan_enter_elapsed = float(m.group(1))
                    scan_enter_stamp = ts

                m = STATE_ENTER_DONE_RE.search(text)
                if m:
                    done_elapsed = float(m.group(1))
                    done_stamp = ts
                    result["success"] = 1
                    result["final_state"] = "DONE"

                m = STATE_ENTER_FAIL_RE.search(text)
                if m:
                    fail_elapsed = float(m.group(1))
                    fail_stamp = ts
                    if result["final_state"] != "DONE":
                        result["success"] = 0
                        result["final_state"] = "FAIL"

                if "NAV_RETRY" in text:
                    result["nav_retry_count"] += 1

                if "Failed to get a plan" in text:
                    result["failed_plan_count"] += 1

            elif topic == "/odometry/filtered":
                x = msg.pose.pose.position.x
                y = msg.pose.pose.position.y
                if last_x is not None and last_y is not None:
                    result["path_length_m"] += math.hypot(x - last_x, y - last_y)
                last_x, last_y = x, y

            elif topic == "/front/scan":
                # valid range extraction
                rng_min = getattr(msg, "range_min", 0.0)
                rng_max = getattr(msg, "range_max", float("inf"))
                for r in msg.ranges:
                    if math.isfinite(r) and rng_min < r < rng_max:
                        if result["min_scan_range_m"] is None or r < result["min_scan_range_m"]:
                            result["min_scan_range_m"] = r

    # total_time priority:
    # 1) elapsed from mission logs (most reliable for mission phase)
    # 2) fallback to rosout timestamps
    end_elapsed = done_elapsed if done_elapsed is not None else fail_elapsed
    if scan_enter_elapsed is not None and end_elapsed is not None:
        result["total_time_s"] = max(0.0, end_elapsed - scan_enter_elapsed)
    else:
        end_stamp = done_stamp if done_stamp is not None else fail_stamp
        if scan_enter_stamp is not None and end_stamp is not None:
            result["total_time_s"] = max(0.0, end_stamp - scan_enter_stamp)

    return result


def update_csv(csv_path: str, analysis: Dict[str, object]) -> bool:
    if not os.path.exists(csv_path):
        return False

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    run_id = analysis["run_id"]
    updated = False
    for row in rows:
        if row.get("run_id") == run_id:
            if "success" in row:
                row["success"] = str(analysis["success"])
            if "total_time_s" in row:
                row["total_time_s"] = safe_float(analysis["total_time_s"])
            if "nav_retry_count" in row:
                row["nav_retry_count"] = str(analysis["nav_retry_count"])
            if "failed_plan_count" in row:
                row["failed_plan_count"] = str(analysis["failed_plan_count"])
            if "final_state" in row:
                row["final_state"] = str(analysis["final_state"])
            if "notes" in row:
                extra = (
                    f"path_length_m={analysis['path_length_m']:.3f};"
                    f"min_scan_range_m={safe_float(analysis['min_scan_range_m'])}"
                )
                prev = row.get("notes", "").strip()
                row["notes"] = extra if not prev else f"{prev} | {extra}"
            updated = True
            break

    if not updated:
        return False

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return True


def main():
    parser = argparse.ArgumentParser(description="Analyze one eval rosbag for tuning metrics.")
    parser.add_argument("bag", help="Path to run_XX.bag")
    parser.add_argument(
        "--update-csv",
        default="",
        help="Optional CSV path to auto-fill metrics (e.g., eval_runs/run_metrics.csv)",
    )
    args = parser.parse_args()

    analysis = analyze_bag(args.bag)

    print("=== Run Analysis ===")
    print(f"run_id            : {analysis['run_id']}")
    print(f"success           : {analysis['success']}")
    print(f"final_state       : {analysis['final_state']}")
    print(f"total_time_s      : {safe_float(analysis['total_time_s'])}")
    print(f"nav_retry_count   : {analysis['nav_retry_count']}")
    print(f"failed_plan_count : {analysis['failed_plan_count']}")
    print(f"path_length_m     : {analysis['path_length_m']:.3f}")
    print(f"min_scan_range_m  : {safe_float(analysis['min_scan_range_m'])}")

    if args.update_csv:
        ok = update_csv(args.update_csv, analysis)
        if ok:
            print(f"CSV updated       : {args.update_csv}")
        else:
            print(f"CSV not updated   : {args.update_csv} (run_id not found or file missing)")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""离线 obs 参考计算（rosbag 回放），用于与 planner 输出比对。

与训练侧 env/wrapper.py 以及 C++ ObsBuilder 保持一致的公式:
    obs[0:100] = 按 base_link 系前向±90° 重采样 100 束归一化 lidar
    obs[100:103] = [dist/rmax, cos(a), sin(a)]
    (prev_action 由 planner 内部维护,本脚本只算前 103 维)

用法:
    rosrun td3_nav obs_reference.py --bag scan.bag --goal-x 5 --goal-y 5 \
        [--laser-yaw-deg 180] [--reverse-scan False] [--out obs_ref.csv]
"""
import argparse
import math

import numpy as np

try:
    import rosbag
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan
except ImportError as e:
    raise SystemExit("需要 ROS python 环境: %s" % e)

P = math.pi / 2.0


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def build_lidar(scan, laser_yaw, reverse_scan, range_max, num_beams=100, half=math.pi / 2.0):
    n = len(scan.ranges)
    la = np.array(
        [scan.angle_min + float(n - 1 - i if reverse_scan else i) * scan.angle_increment
         for i in range(n)]
    )
    base = wrap(la + laser_yaw)
    r = np.array(scan.ranges, dtype=np.float64)
    rv = np.where(np.isfinite(r) & (r <= range_max), np.maximum(r, 0.0), range_max)
    rv = np.minimum(rv, range_max) / range_max

    targets = -half + 2.0 * half * np.arange(num_beams) / (num_beams - 1)
    out = np.empty(num_beams)
    for k, t in enumerate(targets):
        d = np.abs(wrap(base - t))
        out[k] = rv[np.argmin(d)]
    return out


def build_goal(gx, gy, rx, ry, ryaw, range_max):
    dx, dy = gx - rx, gy - ry
    dist = math.hypot(dx, dy)
    a = wrap(math.atan2(dy, dx) - ryaw)
    return np.array([dist / range_max, math.cos(a), math.sin(a)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--goal-x", type=float, default=5.0)
    ap.add_argument("--goal-y", type=float, default=5.0)
    ap.add_argument("--laser-yaw-deg", type=float, default=180.0)
    ap.add_argument("--reverse-scan", action="store_true")
    ap.add_argument("--range-max", type=float, default=7.0)
    ap.add_argument("--out", default="obs_ref.csv")
    args = ap.parse_args()

    laser_yaw = math.radians(args.laser_yaw_deg)
    bag = rosbag.Bag(args.bag)
    odom = None
    rows = []
    for topic, msg, t in bag.read_messages(topics=["/scan", "/odom"]):
        if topic == "/odom":
            odom = msg
            continue
        if odom is None:
            continue
        q = odom.pose.pose.orientation
        rx = odom.pose.pose.position.x
        ry = odom.pose.pose.position.y
        ryaw = yaw_from_quat(q)
        lidar = build_lidar(msg, laser_yaw, args.reverse_scan, args.range_max)
        goal = build_goal(args.goal_x, args.goal_y, rx, ry, ryaw, args.range_max)
        row = np.concatenate([lidar, goal])
        rows.append(row)
    bag.close()

    if not rows:
        raise SystemExit("bag 里没有 /scan 或 /odom")
    arr = np.vstack(rows)
    header = ",".join("lidar%d" % i for i in range(100))
    header += ",goal_dist,goal_cos,goal_sin"
    np.savetxt(args.out, arr, delimiter=",", header=header, comments="")
    print("已写 %d 行 -> %s (obs 前 %d 维)" % (arr.shape[0], args.out, arr.shape[1]))
    print("min/max: lidar=[%.3f, %.3f]  goal_dist=[%.3f, %.3f]"
          % (arr[:, :100].min(), arr[:, :100].max(), arr[:, 100].min(), arr[:, 100].max()))


if __name__ == "__main__":
    main()

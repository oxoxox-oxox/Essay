#!/usr/bin/env python3
"""现场校验 /scan 方向与角度对齐（决定 reverse_scan）。

用法: 在机器人正前方约 0.5~1m 放一个障碍物,然后运行:
    rosrun td3_nav verify_scan_order.py
    (或 python3 scripts/verify_scan_order.py)

原理: 分别按「正常序」和「反序(reverse_scan=True)」两种约定,把 /scan 每束角度
换算到 base_link 系;打印最近障碍束在两种约定下的 base_link 角度。
- 若正常序下最近束 base 角≈0° -> reverse_scan 保持 false
- 若反序下最近束 base 角≈0° -> 设 reverse_scan=true
同时打印两种约定下前向±90° 窗口的平均距离,窗口均值更小(更接近障碍)的即正确约定。
"""
import math
import sys

import numpy as np
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

P = math.pi / 2.0


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def make_angles(n, amin, inc, reversed_):
    if reversed_:
        return [amin + float(n - 1 - i) * inc for i in range(n)]
    return [amin + i * inc for i in range(n)]


class OrderChecker:
    def __init__(self):
        self.laser_yaw = rospy.get_param("~laser_yaw_deg", 180.0) * math.pi / 180.0
        self.range_max = rospy.get_param("~range_max_norm", 7.0)

    def on_scan(self, scan):
        n = len(scan.ranges)
        inc = scan.angle_increment
        amin = scan.angle_min
        ranges = np.array(scan.ranges, dtype=np.float64)
        finite = np.isfinite(ranges)
        if not finite.any():
            return
        i_min = int(np.nanargmin(np.where(finite, ranges, np.inf)))

        for rev in (False, True):
            la = np.array(make_angles(n, amin, inc, rev))
            base = wrap(la + self.laser_yaw)  # laser系 -> base_link 系
            front = (np.abs(base) <= P) & finite
            mean_front = ranges[front].mean() if front.any() else float("nan")
            base_at_min = wrap(base[i_min])
            tag = "reverse_scan=True" if rev else "reverse_scan=False"
            print(
                "[%s] 最近束 idx=%d laser=%.1fdeg  base=%.1fdeg  前向±90均值=%.2fm"
                % (tag, i_min, math.degrees(la[i_min]), math.degrees(base_at_min), mean_front)
            )
        print("-> 期望: 正确约定下 base≈0deg 且前向均值更小(贴近障碍物)")

    def on_odom(self, odom):
        q = odom.pose.pose.orientation
        print("[odom] x=%.3f y=%.3f yaw=%.1fdeg" % (
            odom.pose.pose.position.x, odom.pose.pose.position.y,
            math.degrees(yaw_from_quat(q))))


def main():
    rospy.init_node("verify_scan_order")
    c = OrderChecker()
    rospy.Subscriber("/scan", LaserScan, c.on_scan, queue_size=1)
    rospy.Subscriber("/odom", Odometry, c.on_odom, queue_size=1)
    rospy.loginfo("正在监听 /scan /odom ... 请把障碍物放到机器人正前方 (Ctrl+C 退出)")
    rospy.spin()


if __name__ == "__main__":
    sys.exit(main())

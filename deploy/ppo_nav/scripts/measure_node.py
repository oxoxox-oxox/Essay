#!/usr/bin/env python3
"""真机消融测量节点（独立 rospy 脚本，不依赖 catkin 编译）。

记录/统计：
  - scan->决策 总延时（/planner/scan_age_ms，planner 每个推理 tick 发布）
  - 单次推理耗时（/planner/fwd_ms，量化直接作用项）
  - 推理频率（scan_age 消息数 / 时长）
  - 动作下发频率（/cmd_vel_planner 消息数 / 时长）
  - 安全急停计数（/cmd_vel_planner 非零但 /cmd_vel 被 safety 归零）

用法（真机）:
    python3 measure_node.py _out:=/home/wheeltec/measure.csv
  结果写 CSV（每行 t,kind,value_ms），退出时打印汇总。

说明: GPU 占用不在这里测，用 tegrastats / nvtop 单独记录。
"""

from __future__ import annotations

import csv
import statistics
import time

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32


class MeasureNode:
    def __init__(self) -> None:
        self.t0 = time.time()
        self.scan_ages: list[tuple[float, float]] = []
        self.fwds: list[tuple[float, float]] = []
        self.n_infer = 0
        self.n_action = 0
        self.n_stop = 0
        self.planner_active = False
        self.last_planner = Twist()

        rospy.Subscriber("/planner/scan_age_ms", Float32, self.on_scan_age)
        rospy.Subscriber("/planner/fwd_ms", Float32, self.on_fwd)
        rospy.Subscriber("/cmd_vel_planner", Twist, self.on_planner_cmd)
        rospy.Subscriber("/cmd_vel", Twist, self.on_cmd_vel)
        self.out = rospy.get_param("~out", "/tmp/measure.csv")

    def on_scan_age(self, m: Float32) -> None:
        self.scan_ages.append((time.time(), float(m.data)))
        self.n_infer += 1

    def on_fwd(self, m: Float32) -> None:
        self.fwds.append((time.time(), float(m.data)))

    def on_planner_cmd(self, m: Twist) -> None:
        self.n_action += 1
        self.last_planner = m
        self.planner_active = abs(m.linear.x) > 1e-4 or abs(m.angular.z) > 1e-4

    def on_cmd_vel(self, m: Twist) -> None:
        # safety 转发为 0 且 planner 有输出 -> 记一次急停
        if self.planner_active and abs(m.linear.x) < 1e-4 and abs(m.angular.z) < 1e-4:
            self.n_stop += 1
            self.planner_active = False

    @staticmethod
    def _stat(xs: list[tuple[float, float]]):
        if not xs:
            return (0, 0.0, 0.0, 0.0)
        v = [x[1] for x in xs]
        return (len(v), float(sum(v) / len(v)), float(statistics.median(v)), float(max(v)))

    def save(self) -> None:
        dt = max(time.time() - self.t0, 1e-6)
        with open(self.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "kind", "value_ms"])
            for t, v in self.scan_ages:
                w.writerow([f"{t:.3f}", "scan_age_ms", f"{v:.3f}"])
            for t, v in self.fwds:
                w.writerow([f"{t:.3f}", "fwd_ms", f"{v:.3f}"])

        sa = self._stat(self.scan_ages)
        fw = self._stat(self.fwds)
        rospy.loginfo("========== measure summary ==========")
        rospy.loginfo("elapsed=%.1fs", dt)
        rospy.loginfo("inference Hz = %.3f  (n=%d)", self.n_infer / dt, self.n_infer)
        rospy.loginfo("action    Hz = %.3f  (n=%d)", self.n_action / dt, self.n_action)
        rospy.loginfo("safety stops   = %d", self.n_stop)
        rospy.loginfo("scan_age_ms n=%d mean=%.3f median=%.3f max=%.3f", *sa)
        rospy.loginfo("fwd_ms     n=%d mean=%.3f median=%.3f max=%.3f", *fw)
        rospy.loginfo("saved: %s", self.out)


if __name__ == "__main__":
    rospy.init_node("measure_node")
    node = MeasureNode()
    rospy.on_shutdown(node.save)
    rospy.spin()

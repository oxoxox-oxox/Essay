#!/usr/bin/env python3
"""Real-robot ablation measurement node (standalone rospy script, no catkin build needed).

Records / computes:
  - scan->decision total latency (/planner/scan_age_ms, published by the planner on every inference tick)
  - single inference time (/planner/fwd_ms, the item quantization acts on directly)
  - inference frequency (scan_age message count / duration)
  - action publish frequency (/cmd_vel_planner message count / duration)
  - safety emergency-stop count (/cmd_vel_planner nonzero but /cmd_vel zeroed by safety)

Usage (on robot):
    python3 measure_node.py _out:=/home/wheeltec/measure.csv
  Results written to CSV (each row t,kind,value_ms); prints a summary on exit.

Note: GPU usage is not measured here; record it separately with tegrastats / nvtop.
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
        # safety forwarded 0 while the planner has output -> count one emergency-stop
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

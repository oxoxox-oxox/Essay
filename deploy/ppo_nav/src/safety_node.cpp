/// Safety node: fallback between the planner output (/cmd_vel_planner) and the real /cmd_vel.
///
/// Checks (any trigger -> stop):
///   - nearest laser distance < stop_dist
///   - battery < min_voltage
///   - charging/recharge/red flag set
///   - planner output timeout (watchdog_timeout)
/// Slow zone: when stop_dist < nearest distance < slow_dist, scale by slow_factor.
/// The final output is also clamped by max_linear/max_angular.
#include <geometry_msgs/Twist.h>
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Int8.h>
#include <std_msgs/UInt8.h>

#include <cmath>
#include <mutex>

namespace ppo_nav {

class SafetyNode {
 public:
  SafetyNode(ros::NodeHandle& nh) {
    ros::NodeHandle pnh("safety");
    pnh.param("stop_dist", stop_dist_, 0.35);
    pnh.param("slow_dist", slow_dist_, 0.6);
    pnh.param("slow_factor", slow_factor_, 0.4);
    pnh.param("min_voltage", min_voltage_, 10.5);
    pnh.param("watchdog_timeout", watchdog_timeout_, 0.6);
    pnh.param("max_linear", max_linear_, 0.8);
    pnh.param("max_angular", max_angular_, 1.0);
    pnh.param("front_only", front_only_, false);  // when true, only check the front ±90° nearest distance

    scan_sub_ = nh.subscribe("/scan", 1, &SafetyNode::scanCb, this);
    cmd_sub_ = nh.subscribe("/cmd_vel_planner", 1, &SafetyNode::cmdCb, this);
    volt_sub_ = nh.subscribe("/PowerVoltage", 1, &SafetyNode::voltCb, this);
    charge_sub_ = nh.subscribe("/robot_charging_flag", 1, &SafetyNode::chargeCb, this);
    recharge_sub_ = nh.subscribe("/robot_recharge_flag", 1, &SafetyNode::rechargeCb, this);
    red_sub_ = nh.subscribe("/robot_red_flag", 1, &SafetyNode::redCb, this);
    cmd_pub_ = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
    timer_ = nh.createTimer(ros::Duration(0.1), &SafetyNode::tick, this);
  }

 private:
  void scanCb(const sensor_msgs::LaserScanConstPtr& msg) {
    std::lock_guard<std::mutex> lk(mtx_);
    scan_ = msg;
  }

  void cmdCb(const geometry_msgs::TwistConstPtr& msg) {
    std::lock_guard<std::mutex> lk(mtx_);
    last_cmd_ = *msg;
    last_cmd_time_ = ros::Time::now();
  }

  void voltCb(const std_msgs::Float32ConstPtr& msg) {
    std::lock_guard<std::mutex> lk(mtx_);
    voltage_ = msg->data;
  }

  void chargeCb(const std_msgs::BoolConstPtr& msg) {
    std::lock_guard<std::mutex> lk(mtx_);
    charging_ = msg->data;
  }

  void rechargeCb(const std_msgs::Int8ConstPtr& msg) {
    std::lock_guard<std::mutex> lk(mtx_);
    recharge_ = msg->data != 0;
  }

  void redCb(const std_msgs::UInt8ConstPtr& msg) {
    std::lock_guard<std::mutex> lk(mtx_);
    red_flag_ = msg->data != 0;
  }

  bool obstacleNear(double* min_range) const {
    if (!scan_) return false;
    double mn = 1e9;
    const size_t n = scan_->ranges.size();
    for (size_t i = 0; i < n; ++i) {
      const float r = scan_->ranges[i];
      if (!std::isfinite(r)) continue;
      if (front_only_) {
        double a = scan_->angle_min + static_cast<double>(i) * scan_->angle_increment;
        if (std::fabs(a) > M_PI / 2.0) continue;  // only front ±90° (laser frame)
      }
      mn = std::min(mn, static_cast<double>(r));
    }
    *min_range = mn;
    return n > 0;
  }

  void tick(const ros::TimerEvent&) {
    geometry_msgs::Twist out;
    std::string reason = "ok";

    double min_range = 1e9;
    const bool has_scan = obstacleNear(&min_range);

    {
      std::lock_guard<std::mutex> lk(mtx_);
      const bool stale =
          last_cmd_time_.isZero() || (ros::Time::now() - last_cmd_time_).toSec() > watchdog_timeout_;
      const bool low_voltage = voltage_ < min_voltage_;
      const bool blocking = charging_ || recharge_ || red_flag_;

      if (stale) {
        reason = "watchdog";
      } else if (blocking) {
        reason = "charging/flag";
      } else if (low_voltage) {
        reason = "low_voltage";
      } else if (has_scan && min_range < stop_dist_) {
        reason = "near_obstacle";
      } else {
        out = last_cmd_;
        if (has_scan && min_range < slow_dist_) {
          out.linear.x *= slow_factor_;
          out.angular.z *= slow_factor_;
          reason = "slow";
        }
      }
    }

    out.linear.x = std::max(-max_linear_, std::min(max_linear_, out.linear.x));
    out.angular.z = std::max(-max_angular_, std::min(max_angular_, out.angular.z));
    cmd_pub_.publish(out);

    if (reason != "ok") {
      ROS_WARN_THROTTLE(1.0, "[safety] %s (min_range=%.2f) -> cmd=(%.2f, %.2f)",
                        reason.c_str(), min_range, out.linear.x, out.angular.z);
    }
  }

  double stop_dist_, slow_dist_, slow_factor_;
  double min_voltage_, watchdog_timeout_;
  double max_linear_, max_angular_;
  bool front_only_;

  ros::Subscriber scan_sub_, cmd_sub_, volt_sub_, charge_sub_, recharge_sub_, red_sub_;
  ros::Publisher cmd_pub_;
  ros::Timer timer_;

  std::mutex mtx_;
  sensor_msgs::LaserScanConstPtr scan_;
  geometry_msgs::Twist last_cmd_;
  ros::Time last_cmd_time_;
  double voltage_ = 100.0;
  bool charging_ = false;
  bool recharge_ = false;
  bool red_flag_ = false;
};

}  // namespace ppo_nav

int main(int argc, char** argv) {
  ros::init(argc, argv, "safety_node");
  ros::NodeHandle nh;
  ppo_nav::SafetyNode node(nh);
  ros::spin();
  return 0;
}

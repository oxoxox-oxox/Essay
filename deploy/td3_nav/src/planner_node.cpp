/// TD3 (TensorRT) 导航规划节点。
///
/// 数据流:
///   /scan + /odom + rosparam goal -> obs[1, 100+3+N*2] -> TensorRT engine -> /cmd_vel_planner
///
/// action chunking: 一次推理输出 N 步动作开环执行（chunk_size=N），推理频率降到 1/N，
///   控制周期仍为 control_period 秒/步（对齐训练 step_time=0.3s）。
///   N=1 时为单步（每次决策都推理），行为与原 N1 部署一致。
///
/// 话题:
///   订阅: /scan (sensor_msgs/LaserScan), /odom (nav_msgs/Odometry)
///   发布: /cmd_vel_planner (geometry_msgs/Twist)  <- 由 safety_node 校验后转发到真机 /cmd_vel
///
/// 决策节流为 control_period 秒（对齐训练 step_time=0.3s），使用最新一帧 scan。
/// obs 布局与训练侧 env/wrapper.py 完全一致，详见 include/td3_nav/obs_builder.hpp。
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>

#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "td3_nav/obs_builder.hpp"
#include "td3_nav/tensorrt_engine.hpp"

namespace td3 {

class PlannerNode {
 public:
  PlannerNode(ros::NodeHandle& nh) : tf_buffer_(), tf_listener_(tf_buffer_) {
    ros::NodeHandle pnh("planner");

    pnh.param<std::string>("engine_path", engine_path_, "actor.engine");
    pnh.param("goal_x", goal_x_, 5.0);
    pnh.param("goal_y", goal_y_, 5.0);
    pnh.param("control_period", control_period_, 0.3);
    pnh.param("scan_timeout", scan_timeout_, 0.5);
    pnh.param("debug_log", debug_log_, false);

    // 与训练侧对齐的常量
    pnh.param("num_beams", params_.num_beams, 100);
    pnh.param("range_max_norm", params_.range_max_norm, 7.0);
    double half_fov_deg = 90.0;
    pnh.param("half_fov_deg", half_fov_deg, 90.0);
    params_.half_fov_rad = half_fov_deg * M_PI / 180.0;
    pnh.param("reverse_scan", params_.reverse_scan, false);
    pnh.param("laser_yaw_fallback_deg", laser_yaw_fallback_deg_, 180.0);
    params_.laser_yaw_in_base = laser_yaw_fallback_deg_ * M_PI / 180.0;

    // action chunk 长度 N（决定 obs 维度与开环执行步数）
    pnh.param("chunk_size", params_.chunk_size, 1);
    if (params_.chunk_size < 1) params_.chunk_size = 1;

    // 归一化动作 -> 真实速度映射 + 下发限幅
    std::vector<double> vel_min{ -1.0, -1.0 }, vel_max{ 1.0, 1.0 };
    pnh.getParam("vel_min", vel_min);
    pnh.getParam("vel_max", vel_max);
    if (vel_min.size() >= 2 && vel_max.size() >= 2) {
      params_.vel_min[0] = vel_min[0];
      params_.vel_min[1] = vel_min[1];
      params_.vel_max[0] = vel_max[0];
      params_.vel_max[1] = vel_max[1];
    }
    pnh.param("max_linear", max_linear_, params_.vel_max[0]);
    pnh.param("max_angular", max_angular_, params_.vel_max[1]);

    if (engine_.inputDim() == 0 && !engine_.load(engine_path_)) {
      ROS_FATAL("[planner] failed to load engine %s", engine_path_.c_str());
      ros::shutdown();
      return;
    }
    const int expected_in = params_.num_beams + 3 + params_.prev_action_dim();
    if (engine_.inputDim() != expected_in) {
      ROS_WARN("[planner] engine input=%d (expect %d for chunk=%d), 请核对模型与 obs 配置",
               engine_.inputDim(), expected_in, params_.chunk_size);
    }

    // warmup
    std::vector<float> obs(engine_.inputDim(), 0.5f), act;
    for (int i = 0; i < 20; ++i) engine_.forward(obs, act);

    obs_builder_.reset(new ObsBuilder(params_));
    obs_builder_->setGoal(goal_x_, goal_y_);

    scan_sub_ = nh.subscribe("/scan", 1, &PlannerNode::scanCb, this);
    odom_sub_ = nh.subscribe("/odom", 1, &PlannerNode::odomCb, this);
    cmd_pub_ = nh.advertise<geometry_msgs::Twist>("/cmd_vel_planner", 1);
    timer_ = nh.createTimer(ros::Duration(control_period_), &PlannerNode::decide, this);

    ROS_INFO("[planner] ready. goal=(%.2f, %.2f), period=%.2fs", goal_x_, goal_y_, control_period_);
  }

 private:
  void scanCb(const sensor_msgs::LaserScanConstPtr& msg) {
    std::lock_guard<std::mutex> lk(mtx_);
    scan_ = msg;
  }

  void odomCb(const nav_msgs::OdometryConstPtr& msg) {
    std::lock_guard<std::mutex> lk(mtx_);
    odom_ = msg;
  }

  void publishStop() {
    geometry_msgs::Twist t;
    cmd_pub_.publish(t);
  }

  void decide(const ros::TimerEvent&) {
    sensor_msgs::LaserScanConstPtr scan;
    nav_msgs::OdometryConstPtr odom;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      scan = scan_;
      odom = odom_;
    }

    if (!scan || !odom) {
      pending_.clear();
      publishStop();
      return;
    }
    if ((ros::Time::now() - scan->header.stamp).toSec() > scan_timeout_) {
      ROS_WARN_THROTTLE(1.0, "[planner] scan 超时 %.1fs", (ros::Time::now() - scan->header.stamp).toSec());
      pending_.clear();
      publishStop();
      return;
    }

    // 用 tf2 实时查 laser -> base_link 的 yaw（绕 z 旋转）；失败用兜底参数
    try {
      geometry_msgs::TransformStamped ts =
          tf_buffer_.lookupTransform("base_link", scan->header.frame_id, ros::Time(0));
      tf2::Quaternion q;
      tf2::fromMsg(ts.transform.rotation, q);
      double r, p;
      tf2::Matrix3x3(q).getRPY(r, p, params_.laser_yaw_in_base);
    } catch (const tf2::TransformException& e) {
      params_.laser_yaw_in_base = laser_yaw_fallback_deg_ * M_PI / 180.0;
      ROS_WARN_THROTTLE(2.0, "[planner] tf lookup 失败(%.40s)，用 fallback yaw=%.1fdeg",
                        e.what(), laser_yaw_fallback_deg_);
    }

    // chunk 用尽 -> 推理一次拿新 chunk（obs 用当前 prev_chunk = 上一段已执行 chunk）
    if (pending_.empty()) {
      const double x = odom->pose.pose.position.x;
      const double y = odom->pose.pose.position.y;
      const double yaw = ObsBuilder::yawFromQuaternion(
          odom->pose.pose.orientation.x, odom->pose.pose.orientation.y,
          odom->pose.pose.orientation.z, odom->pose.pose.orientation.w);

      std::vector<float> obs;
      obs_builder_->buildObs(*scan, x, y, yaw, obs);

      std::vector<float> chunk;
      if (!engine_.forward(obs, chunk)) {
        ROS_ERROR_THROTTLE(1.0, "[planner] engine forward failed");
        pending_.clear();
        publishStop();
        return;
      }
      // 记录本 chunk 为“上一 chunk 动作”（供下次决策的 obs 使用），随后开环执行
      obs_builder_->setPrevChunk(chunk);
      pending_ = chunk;
    }

    // 取当前步动作（引擎输出按 step 主序：每步 [lin, ang]）
    const int n = 2;
    if (static_cast<int>(pending_.size()) < n) {
      pending_.clear();
      publishStop();
      return;
    }
    float a_lin = pending_[0];
    float a_ang = pending_[1];
    pending_.erase(pending_.begin(), pending_.begin() + n);

    a_lin = std::max(-1.0f, std::min(1.0f, a_lin));
    a_ang = std::max(-1.0f, std::min(1.0f, a_ang));

    geometry_msgs::Twist out;
    out.linear.x = std::max(-max_linear_, std::min(max_linear_,
        ObsBuilder::scaleAction(a_lin, params_.vel_min[0], params_.vel_max[0])));
    out.angular.z = std::max(-max_angular_, std::min(max_angular_,
        ObsBuilder::scaleAction(a_ang, params_.vel_min[1], params_.vel_max[1])));

    cmd_pub_.publish(out);

    if (debug_log_) {
      ROS_INFO("[planner] v=(%.3f, %.3f) chunk_remain=%d",
               out.linear.x, out.angular.z, static_cast<int>(pending_.size() / n));
    }
  }

  std::string engine_path_;
  double goal_x_, goal_y_;
  double control_period_;
  double scan_timeout_;
  double laser_yaw_fallback_deg_;
  double max_linear_, max_angular_;
  bool debug_log_;
  PlannerParams params_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::unique_ptr<ObsBuilder> obs_builder_;
  TrtEngine engine_;

  ros::Subscriber scan_sub_, odom_sub_;
  ros::Publisher cmd_pub_;
  ros::Timer timer_;

  std::mutex mtx_;
  sensor_msgs::LaserScanConstPtr scan_;
  nav_msgs::OdometryConstPtr odom_;
  std::vector<float> pending_;  // 当前 chunk 剩余待执行动作（归一化，step 主序）
};

}  // namespace td3

int main(int argc, char** argv) {
  ros::init(argc, argv, "planner_node");
  ros::NodeHandle nh;
  td3::PlannerNode node(nh);
  ros::spin();
  return 0;
}

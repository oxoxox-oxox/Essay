/// PPO (TensorRT) 导航规划节点。
///
/// 数据流:
///   /scan + /odom + rosparam goal -> obs[1, 100+3+N*2] -> TensorRT engine -> /cmd_vel_planner
///
/// 动作: PPO 单步（chunk_size=1）或 action chunk（chunk_size=5）。引擎输出 μ，
///   长度 = chunk_size*2（每步 [lin,ang]）；节点 clip 到 [-1,1] 后按 scale_action
///   映射到真实速度并限幅下发。
///
/// 触发机制（事件驱动，非固定周期定时器）:
///   收到一帧"比上次已处理更新的" /scan -> 取最新 scan/odom -> 推理 -> 下发 ->
///   立即等下一帧新 scan 再推理，如此往复。模型每帧雷达只串行跑一次，
///   决策频率 = 雷达帧率（LSLIDAR 典型 12Hz，83ms）。
///   chunk_size>1 时：一次推理产出 N 步动作塞入 pending_，其后 N-1 帧只弹出动作、
///   不重新推理（开环），推理频率降为 1/N。
///
/// 话题:
///   订阅: /scan (sensor_msgs/LaserScan), /odom (nav_msgs/Odometry)
///   发布: /cmd_vel_planner (geometry_msgs/Twist)  <- 由 safety_node 校验后转发到真机 /cmd_vel
///         /planner/scan_age_ms (std_msgs/Float32)  <- scan->决策 总延时（测量用）
///         /planner/fwd_ms     (std_msgs/Float32)  <- 单次推理耗时（量化直接作用项）
///
/// obs 布局与训练侧 env/wrapper.py 完全一致（见 include/ppo_nav/obs_builder.hpp）。
/// ⚠️ chunk=1（N=1 模型）训练时"上一动作"通道恒为 0 -> use_prev_action=false；
///    chunk=5（N=5 模型）训练时 obs 含上一 chunk 的 5 步动作 -> use_prev_action=true。
///    两者必须与所用模型严格匹配。
///
/// ⚠️ 时序口径: 训练 step_time=0.083s（1/12，对齐真机 12Hz 雷达帧率）。本节点帧驱动
///    即 ≈12Hz，与训练一致；N=5 的开环窗口 ≈ 5×83ms = 0.42s。
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <std_msgs/Float32.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "ppo_nav/obs_builder.hpp"
#include "ppo_nav/tensorrt_engine.hpp"

namespace ppo_nav {

class PlannerNode {
 public:
  PlannerNode(ros::NodeHandle& nh) : tf_buffer_(), tf_listener_(tf_buffer_) {
    ros::NodeHandle pnh("planner");

    pnh.param<std::string>("engine_path", engine_path_, "actor.engine");
    pnh.param("goal_x", goal_x_, 5.0);
    pnh.param("goal_y", goal_y_, 5.0);
    pnh.param("scan_timeout", scan_timeout_, 0.5);
    pnh.param("min_decision_period", min_decision_period_, 0.0);  // 0=纯雷达帧驱动
    pnh.param("debug_log", debug_log_, false);

    // 与训练侧对齐的常量
    pnh.param("num_beams", params_.num_beams, 100);
    pnh.param("range_max_norm", params_.range_max_norm, 7.0);
    pnh.param("goal_dist_norm", params_.goal_dist_norm, 10.0);
    double half_fov_deg = 90.0;
    pnh.param("half_fov_deg", half_fov_deg, 90.0);
    params_.half_fov_rad = half_fov_deg * M_PI / 180.0;
    pnh.param("reverse_scan", params_.reverse_scan, false);
    pnh.param("laser_yaw_fallback_deg", laser_yaw_fallback_deg_, 180.0);
    params_.laser_yaw_in_base = laser_yaw_fallback_deg_ * M_PI / 180.0;

    // action chunk 长度 N（决定 obs 维度与开环执行步数；PPO 当前固定 1）
    pnh.param("chunk_size", params_.chunk_size, 1);
    if (params_.chunk_size < 1) params_.chunk_size = 1;

    // 上一动作是否写回 obs（当前 PPO 模型训练时恒 0，保持 false）
    pnh.param("use_prev_action", use_prev_action_, false);

    // 归一化动作 -> 真实速度映射 + 下发限幅
    std::vector<double> vel_min{-1.0, -1.0}, vel_max{1.0, 1.0};
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
    latency_pub_ = nh.advertise<std_msgs::Float32>("/planner/scan_age_ms", 10);  // scan->决策 总延时
    fwd_pub_ = nh.advertise<std_msgs::Float32>("/planner/fwd_ms", 10);            // 单次推理耗时

    worker_ = std::thread(&PlannerNode::workerLoop, this);

    ROS_INFO("[planner] ready. goal=(%.2f, %.2f), scan-driven, min_period=%.2fs, "
             "use_prev_action=%d",
             goal_x_, goal_y_, min_decision_period_, use_prev_action_ ? 1 : 0);
  }

  ~PlannerNode() {
    running_.store(false);
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
  }

 private:
  void scanCb(const sensor_msgs::LaserScanConstPtr& msg) {
    {
      std::lock_guard<std::mutex> lk(mtx_);
      scan_ = msg;
    }
    cv_.notify_one();
  }

  void odomCb(const nav_msgs::OdometryConstPtr& msg) {
    {
      std::lock_guard<std::mutex> lk(mtx_);
      odom_ = msg;
    }
    cv_.notify_one();
  }

  void publishStop() {
    geometry_msgs::Twist t;
    cmd_pub_.publish(t);
  }

  /// 主循环：等一帧比上次更新的 /scan（可选 min_decision_period 节流），串行推理一次。
  void workerLoop() {
    while (ros::ok() && running_.load()) {
      sensor_msgs::LaserScanConstPtr scan;
      nav_msgs::OdometryConstPtr odom;
      {
        std::unique_lock<std::mutex> lk(mtx_);
        auto fresh = [this]() {
          if (!running_.load()) return true;
          if (!scan_ || !odom_) return false;
          if (!last_scan_stamp_.isZero() &&
              scan_->header.stamp <= last_scan_stamp_) return false;
          if (min_decision_period_ > 0.0 &&
              (ros::Time::now() - last_decision_time_).toSec() < min_decision_period_)
            return false;
          return true;
        };
        while (!fresh()) {
          if (min_decision_period_ > 0.0) {
            double remain = min_decision_period_ -
                            (ros::Time::now() - last_decision_time_).toSec();
            if (remain > 0.0) {
              cv_.wait_for(lk, std::chrono::duration<double>(remain));
            } else {
              cv_.wait(lk);
            }
          } else {
            cv_.wait(lk);
          }
        }
        if (!running_.load()) break;
        scan = scan_;
        odom = odom_;
        last_scan_stamp_ = scan->header.stamp;
        last_decision_time_ = ros::Time::now();
      }
      processScan(scan, odom);
    }
  }

  /// 对一帧雷达 + 最新 odom 做一次决策（推理或从 chunk 队列弹出），并下发。
  void processScan(const sensor_msgs::LaserScanConstPtr& scan,
                   const nav_msgs::OdometryConstPtr& odom) {
    if ((ros::Time::now() - scan->header.stamp).toSec() > scan_timeout_) {
      ROS_WARN_THROTTLE(1.0, "[planner] scan 超时 %.1fs",
                        (ros::Time::now() - scan->header.stamp).toSec());
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

    // chunk 用尽 -> 推理一次拿新 chunk（推理前把上一 chunk 已执行动作写回 obs 的 prev 通道）
    if (pending_.empty()) {
      if (use_prev_action_) {
        obs_builder_->setPrevChunk(prev_history_);
      }
      const double x = odom->pose.pose.position.x;
      const double y = odom->pose.pose.position.y;
      const double yaw = ObsBuilder::yawFromQuaternion(
          odom->pose.pose.orientation.x, odom->pose.pose.orientation.y,
          odom->pose.pose.orientation.z, odom->pose.pose.orientation.w);

      std::vector<float> obs;
      obs_builder_->buildObs(*scan, x, y, yaw, obs);

      std::vector<float> chunk;
      ros::Time t0 = ros::Time::now();
      const bool fwd_ok = engine_.forward(obs, chunk);
      const double fwd_ms = (ros::Time::now() - t0).toSec() * 1000.0;
      const double scan_age_ms = (ros::Time::now() - scan->header.stamp).toSec() * 1000.0;
      if (!fwd_ok) {
        ROS_ERROR_THROTTLE(1.0, "[planner] engine forward failed");
        pending_.clear();
        publishStop();
        return;
      }
      pending_ = chunk;
      std_msgs::Float32 fmsg, amsg;
      fmsg.data = static_cast<float>(fwd_ms);
      amsg.data = static_cast<float>(scan_age_ms);
      fwd_pub_.publish(fmsg);
      latency_pub_.publish(amsg);
      if (debug_log_) {
        ROS_INFO("[planner] infer done. obs=%d act=%d", static_cast<int>(obs.size()),
                 static_cast<int>(chunk.size()));
      }
    }

    // 取当前步动作（引擎输出按 step 主序：每步 [lin, ang]；PPO N=1 -> 2 个元素）
    const int n = 2;
    if (static_cast<int>(pending_.size()) < n) {
      pending_.clear();
      publishStop();
      return;
    }
    float a_lin = pending_[0];
    float a_ang = pending_[1];
    pending_.erase(pending_.begin(), pending_.begin() + n);

    // PPO 输出为 μ（无 tanh），必须先 clip 到 [-1,1] 再映射真实速度
    a_lin = std::max(-1.0f, std::min(1.0f, a_lin));
    a_ang = std::max(-1.0f, std::min(1.0f, a_ang));

    // 记录本步已执行（clip 后）动作，供下一 chunk 推理时写回 obs 的 prev 通道
    if (use_prev_action_) {
      prev_history_.push_back(a_lin);
      prev_history_.push_back(a_ang);
      const size_t max_hist = static_cast<size_t>(params_.chunk_size) * 2;
      if (prev_history_.size() > max_hist) {
        prev_history_.erase(prev_history_.begin(), prev_history_.begin() + 2);
      }
    }

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
  double scan_timeout_;
  double min_decision_period_;
  double laser_yaw_fallback_deg_;
  double max_linear_, max_angular_;
  bool debug_log_;
  bool use_prev_action_;
  PlannerParams params_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::unique_ptr<ObsBuilder> obs_builder_;
  TrtEngine engine_;

  ros::Subscriber scan_sub_, odom_sub_;
  ros::Publisher cmd_pub_, latency_pub_, fwd_pub_;

  std::thread worker_;
  std::atomic<bool> running_{true};
  std::mutex mtx_;
  std::condition_variable cv_;
  sensor_msgs::LaserScanConstPtr scan_;
  nav_msgs::OdometryConstPtr odom_;
  ros::Time last_scan_stamp_;      // 上次已处理的那帧 scan 的时间戳
  ros::Time last_decision_time_;   // 上次决策的墙钟时间（min_decision_period 节流用）
  std::vector<float> pending_;     // 当前 chunk 剩余待执行动作（归一化，step 主序）
  std::vector<float> prev_history_;  // 最近 chunk_size 步已执行动作（归一化，供 prev 通道）
};

}  // namespace ppo_nav

int main(int argc, char** argv) {
  ros::init(argc, argv, "planner_node");
  ros::NodeHandle nh;
  ppo_nav::PlannerNode node(nh);
  ros::spin();
  return 0;
}

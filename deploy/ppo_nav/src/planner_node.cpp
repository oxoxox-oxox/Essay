/// PPO (TensorRT) navigation planner node.
///
/// Data flow:
///   /scan + /odom + rosparam goal -> obs[1, 100+3+N*2] -> TensorRT engine -> /cmd_vel_planner
///
/// Action: PPO single-step (chunk_size=1) or action chunk (chunk_size=5). The engine outputs μ,
///   length = chunk_size*2 (per step [lin,ang]); the node clips to [-1,1], then maps through scale_action
///   to real velocity and clamps before publishing.
///
/// Trigger mechanism (event-driven, not a fixed-period timer):
///   on a new /scan newer than the last processed one -> take the latest scan/odom -> infer -> publish ->
///   immediately wait for the next new scan and infer again, repeating. The model runs once serially per lidar frame,
///   decision frequency = lidar frame rate (LSLIDAR typically 12Hz, 83ms).
///   with chunk_size>1: one inference produces N steps pushed into pending_, the next N-1 frames only pop actions,
///   without re-inferring (open-loop), cutting inference frequency to 1/N.
///
/// Topics:
///   subscribe: /scan (sensor_msgs/LaserScan), /odom (nav_msgs/Odometry)
///   publish: /cmd_vel_planner (geometry_msgs/Twist)  <- validated by safety_node then forwarded to the real /cmd_vel
///            /planner/scan_age_ms (std_msgs/Float32)  <- total scan->decision latency (for measurement)
///            /planner/fwd_ms     (std_msgs/Float32)  <- single inference time (the item quantization acts on directly)
///
/// The obs layout is exactly the training side's env/wrapper.py (see include/ppo_nav/obs_builder.hpp).
/// ⚠️ with chunk=1 (N=1 model) the "previous action" channel is always 0 during training -> use_prev_action=false;
///    with chunk=5 (N=5 model) obs includes the previous chunk's 5 actions -> use_prev_action=true.
///    Both must strictly match the model in use.
///
/// ⚠️ Timing convention: training step_time=0.083s (1/12, matching the real 12Hz lidar frame rate). This node is frame-driven
///    so it is ≈12Hz, consistent with training; the N=5 open-loop window ≈ 5×83ms = 0.42s.
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
    pnh.param("min_decision_period", min_decision_period_, 0.0);  // 0=pure lidar-frame-driven
    pnh.param("debug_log", debug_log_, false);

    // Constants aligned with the training side
    pnh.param("num_beams", params_.num_beams, 100);
    pnh.param("range_max_norm", params_.range_max_norm, 7.0);
    pnh.param("goal_dist_norm", params_.goal_dist_norm, 10.0);
    double half_fov_deg = 90.0;
    pnh.param("half_fov_deg", half_fov_deg, 90.0);
    params_.half_fov_rad = half_fov_deg * M_PI / 180.0;
    pnh.param("reverse_scan", params_.reverse_scan, false);
    pnh.param("laser_yaw_fallback_deg", laser_yaw_fallback_deg_, 180.0);
    params_.laser_yaw_in_base = laser_yaw_fallback_deg_ * M_PI / 180.0;

    // action chunk length N (determines the obs dim and open-loop execution steps; PPO currently fixed at 1)
    pnh.param("chunk_size", params_.chunk_size, 1);
    if (params_.chunk_size < 1) params_.chunk_size = 1;

    // whether to write the previous action back into obs (the current PPO model has it always 0 during training, keep false)
    pnh.param("use_prev_action", use_prev_action_, false);

    // normalized action -> real velocity mapping + command clamps
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
      ROS_WARN("[planner] engine input=%d (expect %d for chunk=%d), check the model vs obs config",
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
    latency_pub_ = nh.advertise<std_msgs::Float32>("/planner/scan_age_ms", 10);  // total scan->decision latency
    fwd_pub_ = nh.advertise<std_msgs::Float32>("/planner/fwd_ms", 10);            // single inference time

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

  /// Main loop: wait for a /scan newer than the last processed one (optional min_decision_period throttling), run one serial inference.
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

  /// Make one decision on a lidar frame + latest odom (infer or pop from the chunk queue), and publish.
  void processScan(const sensor_msgs::LaserScanConstPtr& scan,
                   const nav_msgs::OdometryConstPtr& odom) {
    if ((ros::Time::now() - scan->header.stamp).toSec() > scan_timeout_) {
      ROS_WARN_THROTTLE(1.0, "[planner] scan timeout %.1fs",
                        (ros::Time::now() - scan->header.stamp).toSec());
      pending_.clear();
      publishStop();
      return;
    }

    // Query the laser -> base_link yaw (rotation about z) live via tf2; use the fallback param on failure
    try {
      geometry_msgs::TransformStamped ts =
          tf_buffer_.lookupTransform("base_link", scan->header.frame_id, ros::Time(0));
      tf2::Quaternion q;
      tf2::fromMsg(ts.transform.rotation, q);
      double r, p;
      tf2::Matrix3x3(q).getRPY(r, p, params_.laser_yaw_in_base);
    } catch (const tf2::TransformException& e) {
      params_.laser_yaw_in_base = laser_yaw_fallback_deg_ * M_PI / 180.0;
      ROS_WARN_THROTTLE(2.0, "[planner] tf lookup failed (%.40s), using fallback yaw=%.1fdeg",
                        e.what(), laser_yaw_fallback_deg_);
    }

    // chunk exhausted -> run one inference for a new chunk (write the previous chunk's executed actions back into obs's prev channel before inferring)
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

    // Take the current step's action (engine output is step-major: each step [lin, ang]; PPO N=1 -> 2 elements)
    const int n = 2;
    if (static_cast<int>(pending_.size()) < n) {
      pending_.clear();
      publishStop();
      return;
    }
    float a_lin = pending_[0];
    float a_ang = pending_[1];
    pending_.erase(pending_.begin(), pending_.begin() + n);

    // PPO output is μ (no tanh); must clip to [-1,1] before mapping to real velocity
    a_lin = std::max(-1.0f, std::min(1.0f, a_lin));
    a_ang = std::max(-1.0f, std::min(1.0f, a_ang));

    // Record this step's executed (clipped) actions, to write back into obs's prev channel at the next chunk inference
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
  ros::Time last_scan_stamp_;      // timestamp of the last processed scan frame
  ros::Time last_decision_time_;   // wall-clock of the last decision (for min_decision_period throttling)
  std::vector<float> pending_;     // remaining actions of the current chunk to execute (normalized, step-major)
  std::vector<float> prev_history_;  // the most recent chunk_size steps of executed actions (normalized, for the prev channel)
};

}  // namespace ppo_nav

int main(int argc, char** argv) {
  ros::init(argc, argv, "planner_node");
  ros::NodeHandle nh;
  ppo_nav::PlannerNode node(nh);
  ros::spin();
  return 0;
}

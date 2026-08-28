#ifndef PPO_NAV_OBS_BUILDER_HPP_
#define PPO_NAV_OBS_BUILDER_HPP_

#include <sensor_msgs/LaserScan.h>

#include <algorithm>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ppo_nav {

struct PlannerParams {
  double range_max_norm = 7.0;    // normalization range (training value, lidar uses clip(r,0,rmax)/rmax)
  int num_beams = 100;            // training lidar beam count
  double half_fov_rad = M_PI / 2.0;  // forward half field of view (training angle_range=pi)
  double goal_dist_norm = 10.0;   // divisor for normalizing goal distance (training obs.goal_dist_norm)
  bool reverse_scan = false;      // whether the /scan array is reversed (with /lslidar_order / on-site verification)
  double laser_yaw_in_base = 0.0; // fallback yaw(laser->base_link), normally queried live via tf2
  double vel_min[2] = {-1.0, -1.0};  // normalized action -> real velocity mapping lower bound
  double vel_max[2] = {1.0, 1.0};    // mapping upper bound
  int chunk_size = 1;             // action chunk length N (N=1 or N=5)
  int action_dim = 2;             // per-step action dims [lin, ang]
  int prev_action_dim() const { return chunk_size * action_dim; }
};

struct GoalPose {
  double x = 0.0;
  double y = 0.0;
  bool valid = false;
};

/// obs builder: exactly consistent with the training side's env/wrapper.py.
/// layout: obs[0:num_beams]=normalized lidar, [num_beams:num_beams+3]=goal polar coordinates,
///       [num_beams+3:]=previous action (N*action_dim, always 0 during PPO training, see setPrevChunk comment).
class ObsBuilder {
 public:
  explicit ObsBuilder(const PlannerParams& params);

  void setGoal(double x, double y) {
    goal_.x = x;
    goal_.y = y;
    goal_.valid = true;
  }

  /// Convert each /scan beam to the base_link-frame angle, and nearest-neighbor resample 100 beams
  /// at the training angles linspace(-pi/2, pi/2, 100), normalized to [0,1]. inf/NaN/>range_max treated as range_max(->1.0).
  void buildLidar(const sensor_msgs::LaserScan& scan, std::vector<float>& out) const;

  /// Generate the goal 3-dim feature [dist/goal_dist_norm, cos(a), sin(a)] from the robot pose (x,y,yaw).
  void buildGoal(double rx, double ry, double ryaw, std::vector<float>& out) const;

  /// Assemble the full obs (lidar + goal + prev), length should be num_beams+3+N*action_dim (N=1: 105, N=5: 113).
  void buildObs(const sensor_msgs::LaserScan& scan, double rx, double ry, double ryaw,
                std::vector<float>& obs) const;

  /// Set the previous action (normalized [-1,1], per-dim training transform [lin*2, (ang+1)/2], zero-padded if short, truncated if long).
  /// ⚠️ The current PPO model is trained through step_single, so the training-side "previous action" channel is always 0 —
  ///    therefore deployment should NOT call this method by default (prev stays all zeros, consistent with the training distribution);
  ///    enable it only for a model retrained with prev feedback (with planner param use_prev_action: true).
  void setPrevChunk(const std::vector<float>& chunk) {
    prev_.assign(p_.prev_action_dim(), 0.f);
    const size_t n = std::min<size_t>(chunk.size(), prev_.size());
    for (size_t i = 0; i < n; ++i) {
      const int step = static_cast<int>(i / p_.action_dim);
      const int dim = static_cast<int>(i % p_.action_dim);
      if (step >= p_.chunk_size) break;
      const float a = chunk[i];
      prev_[step * p_.action_dim + dim] =
          (dim == 0) ? (a * 2.0f) : ((a + 1.0f) / 2.0f);  // [lin*2, (ang+1)/2]
    }
  }
  const float* prevAction() const { return prev_.data(); }

  /// Normalize an angle to [-pi, pi].
  static double wrapAngle(double a) {
    return std::atan2(std::sin(a), std::cos(a));
  }
  /// Quaternion -> yaw (rotation about z).
  static double yawFromQuaternion(double x, double y, double z, double w) {
    return std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
  }
  /// Normalized action -> real velocity (same as env/wrapper.py scale_action).
  static double scaleAction(double a, double vmin, double vmax) {
    return vmin + (a + 1.0) / 2.0 * (vmax - vmin);
  }

 private:
  PlannerParams p_;
  GoalPose goal_;
  std::vector<double> target_angles_;
  std::vector<float> prev_;
};

}  // namespace ppo_nav

#endif  // PPO_NAV_OBS_BUILDER_HPP_

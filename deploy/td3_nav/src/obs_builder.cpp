#include "td3_nav/obs_builder.hpp"

#include <algorithm>
#include <cmath>

namespace td3 {

ObsBuilder::ObsBuilder(const PlannerParams& params) : p_(params) {
  target_angles_.reserve(p_.num_beams);
  for (int k = 0; k < p_.num_beams; ++k) {
    // ir-sim: angle_list = linspace(-pi/2, pi/2, number) -> 步长 2*pi/2/(number-1)
    double t = -p_.half_fov_rad + 2.0 * p_.half_fov_rad * k / (p_.num_beams - 1);
    target_angles_.push_back(t);
  }
  prev_.assign(p_.prev_action_dim(), 0.f);
}

void ObsBuilder::buildLidar(const sensor_msgs::LaserScan& scan, std::vector<float>& out) const {
  out.assign(p_.num_beams, 1.0f);
  const size_t n = scan.ranges.size();
  if (n == 0) return;

  const double inc = scan.angle_increment;
  const double amin = scan.angle_min;
  std::vector<double> ang(n), rng(n);
  for (size_t i = 0; i < n; ++i) {
    // reverse_scan: 驱动输出反序时，物理角度按 (n-1-i) 递增
    double la = amin + (p_.reverse_scan ? static_cast<double>(n - 1 - i)
                                        : static_cast<double>(i)) * inc;
    ang[i] = wrapAngle(la + p_.laser_yaw_in_base);  // 激光系 -> base_link 系

    const float r = scan.ranges[i];
    double rv;
    if (!std::isfinite(r) || r > p_.range_max_norm) {
      rv = p_.range_max_norm;  // 无回波/超量程 -> 视为无障碍(1.0)
    } else {
      rv = std::max(static_cast<double>(r), 0.0);
    }
    rng[i] = std::min(rv, p_.range_max_norm) / p_.range_max_norm;
  }

  // 每个目标角最近邻取原始束
  for (int k = 0; k < p_.num_beams; ++k) {
    const double t = target_angles_[k];
    double best_d = 1e9;
    size_t best_i = 0;
    for (size_t i = 0; i < n; ++i) {
      double d = std::fabs(wrapAngle(ang[i] - t));
      if (d < best_d) {
        best_d = d;
        best_i = i;
      }
    }
    out[k] = static_cast<float>(rng[best_i]);
  }
}

void ObsBuilder::buildGoal(double rx, double ry, double ryaw, std::vector<float>& out) const {
  out.assign(3, 0.f);
  if (!goal_.valid) return;
  const double dx = goal_.x - rx;
  const double dy = goal_.y - ry;
  const double dist = std::hypot(dx, dy);
  const double angle = wrapAngle(std::atan2(dy, dx) - ryaw);
  out[0] = static_cast<float>(dist / p_.range_max_norm);
  out[1] = static_cast<float>(std::cos(angle));
  out[2] = static_cast<float>(std::sin(angle));
}

void ObsBuilder::buildObs(const sensor_msgs::LaserScan& scan, double rx, double ry, double ryaw,
                          std::vector<float>& obs) const {
  std::vector<float> lidar, goal;
  buildLidar(scan, lidar);
  buildGoal(rx, ry, ryaw, goal);
  obs.clear();
  obs.reserve(lidar.size() + goal.size() + prev_.size());
  obs.insert(obs.end(), lidar.begin(), lidar.end());
  obs.insert(obs.end(), goal.begin(), goal.end());
  obs.insert(obs.end(), prev_.begin(), prev_.end());
}

}  // namespace td3

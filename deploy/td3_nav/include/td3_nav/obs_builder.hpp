#ifndef TD3_NAV_OBS_BUILDER_HPP_
#define TD3_NAV_OBS_BUILDER_HPP_

#include <sensor_msgs/LaserScan.h>

#include <algorithm>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace td3 {

struct PlannerParams {
  double range_max_norm = 7.0;  // 归一化量程（训练值，obs 用 clip(r,0,rmax)/rmax）
  int num_beams = 100;          // 训练 lidar 光束数
  double half_fov_rad = M_PI / 2.0;  // 前向半视场（训练 angle_range=pi）
  bool reverse_scan = false;    // /scan 数组是否反序（配合 /lslidar_order / 现场校验）
  double laser_yaw_in_base = 0.0;  // 兜底 yaw(laser->base_link)，正常用 tf2 实时查
  double vel_min[2] = {-1.0, -1.0};  // 归一化动作 -> 真实速度 映射下限
  double vel_max[2] = {1.0, 1.0};    // 映射上限
  int chunk_size = 1;           // action chunk 长度 N（N=1 为单步）
  int action_dim = 2;           // 每步动作维数 [lin, ang]
  int prev_action_dim() const { return chunk_size * action_dim; }
};

struct GoalPose {
  double x = 0.0;
  double y = 0.0;
  bool valid = false;
};

/// obs 构造器：与训练侧 env/wrapper.py 完全一致。
/// 布局: obs[0:num_beams]=归一化lidar, [num_beams:num_beams+3]=goal极坐标,
///       [num_beams+3:]=上一 chunk 归一化动作（N*action_dim，episode 起点全零）。
class ObsBuilder {
 public:
  explicit ObsBuilder(const PlannerParams& params);

  void setGoal(double x, double y) {
    goal_.x = x;
    goal_.y = y;
    goal_.valid = true;
  }

  /// 将 /scan 每束换算到 base_link 系角度，按训练角度 linspace(-pi/2, pi/2, 100)
  /// 最近邻重采样 100 束并归一化到 [0,1]。inf/NaN/>range_max 按 range_max(->1.0) 处理。
  void buildLidar(const sensor_msgs::LaserScan& scan, std::vector<float>& out) const;

  /// 由机器人位姿 (x,y,yaw) 生成 goal 3 维特征 [dist/rmax, cos(a), sin(a)]。
  void buildGoal(double rx, double ry, double ryaw, std::vector<float>& out) const;

  /// 组装完整 obs（lidar + goal + prev_chunk），长度应为 num_beams+3+N*action_dim。
  void buildObs(const sensor_msgs::LaserScan& scan, double rx, double ry, double ryaw,
                std::vector<float>& obs) const;

  /// 设置上一 chunk 动作（全 N 步归一化动作），不足补零、多余截断；episode 起点全零。
  void setPrevChunk(const std::vector<float>& chunk) {
    prev_.assign(p_.prev_action_dim(), 0.f);
    const size_t n = std::min<size_t>(chunk.size(), prev_.size());
    std::copy_n(chunk.begin(), n, prev_.begin());
  }
  const float* prevAction() const { return prev_.data(); }

  /// 归一化到 [-pi, pi]。
  static double wrapAngle(double a) {
    return std::atan2(std::sin(a), std::cos(a));
  }
  /// 四元数 -> yaw（绕 z 轴）。
  static double yawFromQuaternion(double x, double y, double z, double w) {
    return std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
  }
  /// 归一化动作 -> 真实速度（同 env/wrapper.py scale_action）。
  static double scaleAction(double a, double vmin, double vmax) {
    return vmin + (a + 1.0) / 2.0 * (vmax - vmin);
  }

 private:
  PlannerParams p_;
  GoalPose goal_;
  std::vector<double> target_angles_;
  std::vector<float> prev_;
};

}  // namespace td3

#endif  // TD3_NAV_OBS_BUILDER_HPP_

/*
 * MIT License
 *
 * Copyright (c) 2025 Meher V.R. Malladi.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <spdlog/spdlog.h>

#include "deskew.hpp"
#include "error.hpp"
#include "lio.hpp"
#include "preprocess_scan.hpp"
#include "profiler.hpp"
#include "util.hpp"
// other
#include <sophus/se3.hpp>
// tbb
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_arena.h>
// stl
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>

namespace {
using namespace rko_lio::core;

constexpr Scalar NEGLIGIBLE_EXTRINSIC = 1e-6;
constexpr auto EPSILON_TIME = std::chrono::nanoseconds(10);
constexpr auto MAX_SCAN_TO_SCAN_DELTA = std::chrono::seconds(1);
constexpr std::size_t MAX_INTERVAL_GYRO_SAMPLES = 8192;
constexpr std::size_t STILL_INTERVALS_BEFORE_HOLD = 3;
constexpr std::size_t HOLD_CHECK_EVERY = 5;
constexpr std::size_t HOLD_COOLDOWN_SCANS = 10;
constexpr std::size_t HOLD_CHECKS_FAILED_TO_RELEASE = 3;
constexpr Scalar HOLD_FAR_OFF_FACTOR = 3;
constexpr std::size_t HOLD_SCANS_BEFORE_BIAS_UPDATE = 10;
constexpr Scalar MAX_GYRO_MOTION_BIAS = 0.002;

template <typename T>
inline std::string streamed(const T& value) {
  std::ostringstream oss;
  oss << value;
  return oss.str();
}

inline double ms_since(const std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

inline void transform_points(const Sophus::SE3s& T, Vector3sVector& points) {
  std::transform(points.begin(), points.end(), points.begin(), [&](const auto& point) { return T * point; });
}

struct AccelInfo {
  Scalar accel_mag_variance;
  Eigen::Vector3s local_gravity_estimate;
};

struct BodyAccelKF {
  Eigen::Vector3s mean;
  Eigen::Matrix3s covariance;
};

struct AccelFilterStep {
  std::optional<AccelInfo> info;
  BodyAccelKF updated{};
};

AccelFilterStep step_body_accel_filter(const BodyAccelKF& prev,
                                       const IntervalStats& stats,
                                       const Sophus::SO3s& rotation_estimate,
                                       const Nsec dt,
                                       const Scalar max_expected_jerk) {
  if (stats.imu_count <= 1) {
    spdlog::warn("{} IMU message(s) in interval between two lidar scans. Cannot compute acceleration statistics "
                 "for orientation regularisation. Please check your data and its timestamping as likely there "
                 "should not be so few IMU measurements between two LiDAR scans.",
                 stats.imu_count);
    return {.info = std::nullopt, .updated = prev};
  }

  const Eigen::Vector3s avg_imu_accel = stats.imu_acceleration_sum / stats.imu_count;
  const Scalar accel_mag_variance = stats.welford_sum_of_squares / static_cast<Scalar>(stats.imu_count - 1);
  const Eigen::Vector3s body_accel_measurement = avg_imu_accel + rotation_estimate.inverse() * gravity();

  const auto max_acceleration_change = static_cast<Scalar>(max_expected_jerk * to_seconds(dt));
  // assume [j, -j] range for uniform dist. on jerk. variance is (2j)^2 / 12 = j^2/3. multiply by dt^2 for accel
  const Eigen::Matrix3s process_noise = square(max_acceleration_change) / 3 * Eigen::Matrix3s::Identity();
  const Eigen::Matrix3s cov_pred = prev.covariance + process_noise;

  // isotropic accel mag variance
  const Eigen::Matrix3s measurement_noise = accel_mag_variance / 3 * Eigen::Matrix3s::Identity();
  const Eigen::Matrix3s S = cov_pred + measurement_noise;
  const Eigen::Matrix3s kalman_gain = cov_pred * S.inverse();

  const Eigen::Vector3s new_mean = prev.mean + kalman_gain * (body_accel_measurement - prev.mean);
  const Eigen::Matrix3s new_cov = cov_pred - kalman_gain * cov_pred;

  const Eigen::Vector3s local_gravity_estimate = avg_imu_accel - new_mean; // points upwards
  return {
      .info = AccelInfo{.accel_mag_variance = accel_mag_variance, .local_gravity_estimate = local_gravity_estimate},
      .updated = BodyAccelKF{.mean = new_mean, .covariance = new_cov},
  };
}

class IntervalTrajectory {
public:
  IntervalTrajectory(const MotionPrior& motion, const std::vector<GyroSample>& gyro, const Nsec end_time)
      : start_time_(motion.start_time), linear_velocity_(motion.linear_velocity), acceleration_(motion.acceleration) {
    nodes_.reserve(gyro.size() + 2);
    nodes_.push_back({.time = start_time_});
    for (const GyroSample& sample : gyro) {
      if (sample.time <= nodes_.back().time) {
        nodes_.back().rate_after = sample.angular_velocity;
        continue;
      }
      nodes_.back().rate_after = sample.angular_velocity;
      append_node(sample.time);
    }
    if (gyro.empty()) {
      nodes_.back().rate_after = motion.angular_velocity;
    } else {
      nodes_.back().rate_after = gyro.back().angular_velocity;
    }
    if (end_time > nodes_.back().time) {
      const Eigen::Vector3s last_rate = nodes_.back().rate_after;
      append_node(end_time);
      nodes_.back().rate_after = last_rate;
    }
  }

  Sophus::SE3s at(const Nsec time) const {
    const auto next = std::upper_bound(nodes_.cbegin(), nodes_.cend(), time,
                                       [](const Nsec t, const Node& node) { return t < node.time; });
    const Node& node = next == nodes_.cbegin() ? nodes_.front() : *std::prev(next);
    return advance(node, time);
  }

private:
  struct Node {
    Nsec time{0};
    Sophus::SO3s rotation;
    Eigen::Vector3s position = Eigen::Vector3s::Zero();
    Eigen::Vector3s rate_after = Eigen::Vector3s::Zero();
  };

  Sophus::SE3s advance(const Node& node, const Nsec time) const {
    const auto h = to_seconds<Scalar>(time - node.time);
    const auto since_start = to_seconds<Scalar>(node.time - start_time_);
    const Eigen::Vector3s mid_velocity = linear_velocity_ + acceleration_ * (since_start + h / 2);
    const Sophus::SO3s half_turn = node.rotation * Sophus::SO3s::exp(node.rate_after * (h / 2));
    return {node.rotation * Sophus::SO3s::exp(node.rate_after * h), node.position + half_turn * (mid_velocity * h)};
  }

  void append_node(const Nsec time) {
    const Sophus::SE3s pose = advance(nodes_.back(), time);
    nodes_.push_back({.time = time, .rotation = pose.so3(), .position = pose.translation()});
  }

  Nsec start_time_;
  Eigen::Vector3s linear_velocity_;
  Eigen::Vector3s acceleration_;
  std::vector<Node> nodes_;
};

void deskew_scan(Vector3sVector& scan, const Timestamps& timestamps, const IntervalTrajectory& trajectory) {
  using Affine = Eigen::Matrix<Scalar, 3, 4>;
  constexpr double KNOT_SPACING_NS = 2e6;
  constexpr std::size_t MAX_INTERVALS = 512;

  const Sophus::SE3s to_reference = trajectory.at(timestamps.max).inverse();
  const auto as_affine = [&](const Nsec time) {
    const Sophus::SE3s pose = to_reference * trajectory.at(time);
    Affine affine;
    affine.leftCols<3>() = pose.so3().matrix();
    affine.col(3) = pose.translation();
    return affine;
  };

  const double span_ns = static_cast<double>((timestamps.max - timestamps.min).count());
  if (span_ns <= 0) {
    return;
  }
  const std::size_t intervals =
      std::clamp<std::size_t>(static_cast<std::size_t>(std::ceil(span_ns / KNOT_SPACING_NS)), 1, MAX_INTERVALS);
  std::vector<Affine> knots(intervals + 1);
  std::vector<Affine> slopes(intervals);
  for (std::size_t k = 0; k <= intervals; ++k) {
    const auto offset = static_cast<std::int64_t>(std::llround(span_ns * static_cast<double>(k) / intervals));
    knots[k] = as_affine(timestamps.min + Nsec(offset));
  }
  for (std::size_t k = 0; k < intervals; ++k) {
    slopes[k] = knots[k + 1] - knots[k];
  }

  const auto knots_per_ns = static_cast<Scalar>(static_cast<double>(intervals) / span_ns);
  const auto last_interval = static_cast<Scalar>(intervals - 1);
  const std::int64_t min_ns = timestamps.min.count();
  const std::size_t point_count = scan.size();
  std::int64_t stamp_of_transform = std::numeric_limits<std::int64_t>::min();
  Eigen::Matrix3s rotation = Eigen::Matrix3s::Identity();
  Eigen::Vector3s translation = Eigen::Vector3s::Zero();
  for (std::size_t i = 0; i < point_count; ++i) {
    const std::int64_t stamp = timestamps.per_point[i].count();
    if (stamp != stamp_of_transform) {
      const Scalar position = static_cast<Scalar>(stamp - min_ns) * knots_per_ns;
      const Scalar interval = std::clamp(std::floor(position), static_cast<Scalar>(0), last_interval);
      const auto k = static_cast<std::size_t>(interval);
      const Affine transform = knots[k] + (position - interval) * slopes[k];
      rotation = transform.leftCols<3>();
      translation = transform.col(3);
      stamp_of_transform = stamp;
    }
    const Eigen::Vector3s point = scan[i];
    scan[i] = rotation * point + translation;
  }
}

using LinearSystem = std::tuple<Eigen::Matrix6s, Eigen::Vector6s, Scalar>;

struct IcpSums {
  Eigen::Vector3d source = Eigen::Vector3d::Zero();
  Eigen::Matrix3d source_outer = Eigen::Matrix3d::Zero();
  Eigen::Vector3d residual = Eigen::Vector3d::Zero();
  Eigen::Vector3d source_cross_residual = Eigen::Vector3d::Zero();
  double residual_sq = 0;
  std::size_t count = 0;

  void add(const Eigen::Vector3s& s, const Eigen::Vector3s& r) {
    const Eigen::Vector3d sd = s.cast<double>();
    const Eigen::Vector3d rd = r.cast<double>();
    source += sd;
    source_outer.noalias() += sd * sd.transpose();
    residual += rd;
    source_cross_residual += sd.cross(rd);
    residual_sq += rd.squaredNorm();
    ++count;
  }

  IcpSums& operator+=(const IcpSums& other) {
    source += other.source;
    source_outer += other.source_outer;
    residual += other.residual;
    source_cross_residual += other.source_cross_residual;
    residual_sq += other.residual_sq;
    count += other.count;
    return *this;
  }
};

LinearSystem build_icp_linear_system(const Sophus::SE3s& current_pose,
                                     const rko_lio::core::Vector3sVector& keypoints,
                                     const rko_lio::core::VoxelHashMap& voxel_map,
                                     const Scalar& max_correspondence_distance,
                                     const bool single_thread,
                                     std::size_t& correspondences_out) {
  const Eigen::Matrix3s rotation = current_pose.so3().matrix();
  const Eigen::Matrix3s rotation_transpose = rotation.transpose();
  const Eigen::Vector3s translation = current_pose.translation();

  using points_iterator = std::vector<Eigen::Vector3s>::const_iterator;
  const auto accumulate = [&](const points_iterator first, const points_iterator last, IcpSums sums) {
    for (auto it = first; it != last; ++it) {
      const Eigen::Vector3s& point = *it;
      const Eigen::Vector3s transformed_point = rotation * point + translation;
      const auto closest_neighbor = voxel_map.get_closest_neighbor(transformed_point, max_correspondence_distance);
      if (!closest_neighbor.has_value()) {
        continue;
      }
      sums.add(point, rotation_transpose * (transformed_point - *closest_neighbor));
    }
    return sums;
  };

  const IcpSums sums =
      single_thread ? accumulate(keypoints.cbegin(), keypoints.cend(), IcpSums{})
                    : tbb::parallel_reduce(
                          tbb::blocked_range<points_iterator>{keypoints.cbegin(), keypoints.cend()}, IcpSums{},
                          [&](const tbb::blocked_range<points_iterator>& r, const IcpSums& partial) {
                            return accumulate(r.begin(), r.end(), partial);
                          },
                          [](IcpSums lhs, const IcpSums& rhs) {
                            lhs += rhs;
                            return lhs;
                          });

  correspondences_out = sums.count;
  if (sums.count == 0) {
    throw RegistrationError("ICP found no correspondences");
  }

  const double inv_count = 1.0 / static_cast<double>(sums.count);
  const Eigen::Vector3d mean_source = sums.source * inv_count;
  const Eigen::Matrix3d mean_outer = sums.source_outer * inv_count;
  Eigen::Matrix<double, 6, 6> H;
  H.block<3, 3>(0, 0).setIdentity();
  H.block<3, 3>(0, 3) = -Sophus::SO3d::hat(mean_source);
  H.block<3, 3>(3, 0) = Sophus::SO3d::hat(mean_source);
  H.block<3, 3>(3, 3) = mean_outer.trace() * Eigen::Matrix3d::Identity() - mean_outer;
  Eigen::Matrix<double, 6, 1> b;
  b.head<3>() = sums.residual * inv_count;
  b.tail<3>() = sums.source_cross_residual * inv_count;
  return {H.cast<Scalar>(), b.cast<Scalar>(), static_cast<Scalar>(0.5 * sums.residual_sq * inv_count)};
}

LinearSystem build_orientation_linear_system(const Sophus::SE3s& current_pose,
                                             const Eigen::Vector3s& local_gravity_estimate) {
  const Sophus::SO3s& current_rotation = current_pose.so3();
  const Eigen::Vector3s predicted_gravity =
      current_rotation.inverse() * (-1 * gravity()); // points upwards, same as local_gravity_estimate
  const Eigen::Vector3s residual = predicted_gravity - local_gravity_estimate;

  Eigen::Matrix3_6s J_ori = Eigen::Matrix3_6s::Zero();
  J_ori.block<3, 3>(0, 3) = Sophus::SO3s::hat(predicted_gravity);

  return LinearSystem{J_ori.transpose() * J_ori, J_ori.transpose() * residual, 0.5 * residual.squaredNorm()};
}

Sophus::SE3s icp(const Vector3sVector& keypoints,
                 const VoxelHashMap& voxel_map,
                 const Sophus::SE3s& initial_guess,
                 const LIO::Config& config,
                 const std::optional<AccelInfo>& optional_accel_info,
                 const Scalar rotation_prior_weight,
                 LIO::FrameStats& stats) {
  // in case config disables it, or we don't have valid IMU information for this icp loop, beta is -1
  const Scalar beta = (config.min_beta > 0 && optional_accel_info.has_value())
                          ? (config.min_beta * (1 + optional_accel_info->accel_mag_variance))
                          : -1;
  Sophus::SE3s current_pose = initial_guess;
  Scalar previous_chi = -1;

  for (size_t i = 0; i < config.max_iterations; ++i) {
    const auto& [H, b, chi] = std::invoke([&]() -> LinearSystem {
      const auto& [H_icp, b_icp, chi_icp] =
          build_icp_linear_system(current_pose, keypoints, voxel_map, config.max_correspondence_distance,
                                  config.max_num_threads == 1, stats.correspondences);
      if (beta >= 0) {
        const auto& [H_ori, b_ori, chi_ori] =
            build_orientation_linear_system(current_pose, optional_accel_info->local_gravity_estimate);
        // only chi_icp is used for convergence checks. chi_ori is typically small, fine to ignore, and problematically
        // noisy near convergence
        return {H_icp + H_ori / beta, b_icp + b_ori / beta, chi_icp};
      }
      return {H_icp, b_icp, chi_icp};
    });

    Eigen::Matrix6s H_total = H;
    Eigen::Vector6s b_total = b;
    if (rotation_prior_weight > 0) {
      const Eigen::Vector3s rotation_residual = (initial_guess.so3().inverse() * current_pose.so3()).log();
      H_total.block<3, 3>(3, 3).diagonal().array() += rotation_prior_weight;
      b_total.tail<3>() += rotation_prior_weight * rotation_residual;
    }

    const Eigen::Vector6s dx = H_total.ldlt().solve(-b_total);
    if (!dx.allFinite()) {
      throw RegistrationError("ICP normal equations have no finite solution");
    }
    current_pose = current_pose * Sophus::SE3s::exp(dx);
    stats.icp_iterations = i + 1;
    stats.chi = chi;
    stats.last_step_translation = dx.head<3>().norm();
    stats.last_step_rotation = dx.tail<3>().norm();

    // Ceres function_tolerance style convergence check
    const Scalar cost_change = std::abs(previous_chi - chi);
    const bool settled = config.convergence_step > 0 && stats.last_step_translation < config.convergence_step &&
                         stats.last_step_rotation < config.convergence_step / 5;
    const bool converged =
        settled || (previous_chi > 0 && cost_change <= config.convergence_criterion * previous_chi);
    previous_chi = chi;
    if (converged || i == (config.max_iterations - 1)) {
      // TODO: proper debug logging
      // std::cout << "iter " << i << ", beta: " << beta << ", chi: " << chi << ", num_assoc: " <<
      // correspondences.size() << "\n";
      break;
    }
  }
  return current_pose;
}

bool gyro_covers(const std::vector<GyroSample>& gyro, const Nsec start, const Nsec end) {
  constexpr Nsec MAX_HOLE = std::chrono::milliseconds(25);
  if (gyro.empty()) {
    return false;
  }
  Nsec covered_until = start;
  for (const GyroSample& sample : gyro) {
    if (sample.time - covered_until > MAX_HOLE) {
      return false;
    }
    covered_until = std::max(covered_until, sample.time);
  }
  return end - covered_until <= MAX_HOLE;
}

bool imu_reads_still(const std::vector<GyroSample>& gyro,
                     const IntervalStats& stats,
                     const Eigen::Vector3s& gyro_bias,
                     const LIO::Config& config) {
  constexpr int MIN_SAMPLES = 8;
  constexpr Scalar MAX_PLAUSIBLE_GYRO_BIAS = 0.05;
  if (stats.imu_count < MIN_SAMPLES || gyro.size() < static_cast<std::size_t>(MIN_SAMPLES)) {
    return false;
  }
  const Eigen::Vector3s mean_rate = stats.angular_velocity_sum / stats.imu_count;
  if ((mean_rate + gyro_bias).norm() > MAX_PLAUSIBLE_GYRO_BIAS) {
    return false;
  }
  const Scalar accel_mag_variance = stats.welford_sum_of_squares / static_cast<Scalar>(stats.imu_count - 1);
  if (accel_mag_variance > square(config.zupt_accel_threshold)) {
    return false;
  }
  return std::all_of(gyro.cbegin(), gyro.cend(), [&](const GyroSample& sample) {
    return (sample.angular_velocity - mean_rate).norm() < config.zupt_gyro_threshold;
  });
}

Eigen::Vector3s tilt_correction(const IntervalStats& stats, const Scalar interval, const Scalar time_constant) {
  constexpr Scalar MAX_STEP = 1e-3;
  if (stats.imu_count < 2 || time_constant <= 0 || interval <= 0) {
    return Eigen::Vector3s::Zero();
  }
  const Eigen::Vector3s specific_force = stats.odom_acceleration_sum / stats.imu_count;
  const Scalar magnitude = specific_force.norm();
  if (magnitude < static_cast<Scalar>(0.5) * GRAVITY_MAG) {
    return Eigen::Vector3s::Zero();
  }
  const Scalar accel_mag_variance = stats.welford_sum_of_squares / static_cast<Scalar>(stats.imu_count - 1);
  const Scalar calm = 1 / (1 + accel_mag_variance);
  const Scalar one_g = std::max<Scalar>(0, 1 - std::abs(magnitude - GRAVITY_MAG));
  const Eigen::Vector3s tilt = (specific_force / magnitude).cross(Eigen::Vector3s::UnitZ());
  Eigen::Vector3s step = tilt * (std::min<Scalar>(1, interval / time_constant) * calm * one_g);
  const Scalar size = step.norm();
  if (size > MAX_STEP) {
    step *= MAX_STEP / size;
  }
  return step;
}

inline bool within(const Sophus::SE3s& a, const Sophus::SE3s& b, const Scalar translation, const Scalar rotation) {
  const Sophus::SE3s delta = a.inverse() * b;
  return delta.translation().norm() < translation && delta.so3().log().norm() < rotation;
}

inline Sophus::SO3s align_accel_to_z_world(const Eigen::Vector3s& accel) {
  //  unobservable in the gravity direction, and the z in R.log() will always be 0
  const Eigen::Vector3s z_world = {0.0, 0.0, 1.0};
  const Eigen::Quaternions quat_accel = Eigen::Quaternions::FromTwoVectors(accel, z_world);
  return Sophus::SO3s(quat_accel);
}
} // namespace

// ==========================
//   actual LIO class stuff
// ==========================

namespace rko_lio::core {

LIO::LIO(const Config& config_)
    : config(config_),
      map(config_.voxel_size, config_.max_range),
      arena(config_.max_num_threads > 0 ? config_.max_num_threads : tbb::task_arena::automatic) {}

// ==========================
//          private
// ==========================

void LIO::initialize(const Nsec lidar_time) {
  if (interval_stats.imu_count == 0) {
    spdlog::warn("Cannot initialize. No imu measurements received.");
    // lidar_state.time has the time from the previous lidar, which we didn't log if init_phase was on
    poses_with_timestamps.emplace_back(lidar_state.time, lidar_state.pose);
    initialized_ = true;
    return;
  }

  const Eigen::Vector3s avg_accel = interval_stats.imu_acceleration_sum / interval_stats.imu_count;
  const Eigen::Vector3s avg_gyro = interval_stats.angular_velocity_sum / interval_stats.imu_count;

  const Sophus::SO3s initial_rotation = align_accel_to_z_world(avg_accel);
  lidar_state.pose.so3() = initial_rotation;
  imu_state = lidar_state;

  // lidar_state.time has the time from the previous lidar, which we didn't log if init_phase was on
  poses_with_timestamps.emplace_back(lidar_state.time, lidar_state.pose);

  // the pose for the current time gets logged at the end of register_scan in the typical fashion
  lidar_state.time = lidar_time;
  imu_state.time = lidar_time;

  const Eigen::Vector3s local_gravity = imu_state.pose.so3().inverse() * gravity();
  imu_bias.accelerometer = avg_accel + local_gravity;
  imu_bias.gyroscope = avg_gyro;
  gyro_rest_bias = avg_gyro;
  gyro_motion_bias.setZero();

  initialized_ = true;
  spdlog::info("Odometry map frame initialized using {} IMU measurements. Estimated initial rotation [se(3)] is {}",
               interval_stats.imu_count, streamed(imu_state.pose.so3().log().transpose()));
  spdlog::info("Estimated accel bias: {}, gyro bias: {}", streamed(imu_bias.accelerometer.transpose()),
               streamed(imu_bias.gyroscope.transpose()));
}

Vector3sVector LIO::bootstrap_first_scan(const Vector3sVector& scan, const Nsec current_lidar_time) {
  lidar_state.time = current_lidar_time;
  imu_state = lidar_state;
  auto preproc = preprocess_scan(scan, config);
  if (!config.initialization_phase) {
    map.update(config.double_downsample ? preproc.map_points : preproc.keypoints, lidar_state.pose);
    poses_with_timestamps.emplace_back(lidar_state.time, lidar_state.pose);
    spdlog::info("Odometry map frame initialized with first lidar scan.");
  }
  return std::move(preproc.filtered_scan);
}

MotionPrior LIO::motion_prior_from_imu(const Nsec current_lidar_time) {
  const Nsec start = lidar_state.time;
  const Eigen::Vector3s& linear_velocity = lidar_state.linear_velocity;
  if (config.initialization_phase && !initialized_) {
    // assume static and
    initialize(current_lidar_time);
    return {.start_time = start, .linear_velocity = linear_velocity};
  }
  if (interval_stats.imu_count == 0) {
    spdlog::warn("No Imu measurements in interval to average. Assuming constant velocity motion.");
    return {.start_time = start, .linear_velocity = linear_velocity, .angular_velocity = lidar_state.angular_velocity};
  }
  const Eigen::Vector3s avg_body_accel = interval_stats.body_acceleration_sum / interval_stats.imu_count;
  const Eigen::Vector3s avg_ang_vel = interval_stats.angular_velocity_sum / interval_stats.imu_count;
  if (avg_body_accel.norm() > 50.0) {
    spdlog::warn("Erratic body acceleration computed, norm > 50 m/s2. Either IMU data is corrupted, or you should "
                 "report an issue.");
  }
  return {
      .start_time = start,
      .linear_velocity = linear_velocity,
      .acceleration = avg_body_accel,
      .angular_velocity = avg_ang_vel,
  };
}

// ==========================
//          public
// ==========================

// ============================ imu ===============================

void LIO::add_imu_measurement(const ImuControl& base_imu) {
  if (!base_imu.acceleration.allFinite() || !base_imu.angular_velocity.allFinite()) {
    static bool warning_not_finite = false;
    if (!warning_not_finite) {
      spdlog::warn("Skipping IMU message with non-finite values. Ignoring all such messages.");
      warning_not_finite = true;
    }
    return;
  }
  if (lidar_state.time < EPSILON_TIME) {
    static bool warning_skip_till_first_lidar = false;
    if (!warning_skip_till_first_lidar) {
      spdlog::warn("Skipping IMU, waiting for first LiDAR message.");
      warning_skip_till_first_lidar = true;
    }
    last_real_imu_time_ = base_imu.time;
    last_real_base_imu_ang_vel_ = base_imu.angular_velocity;
    return;
  }

  if (imu_state.time < EPSILON_TIME) {
    imu_state = lidar_state;
  }

  const auto dt = to_seconds<Scalar>(base_imu.time - imu_state.time);

  if (dt < 0.0) {
    spdlog::warn("Skipping IMU message from the past.");
    return;
  }

  const Eigen::Vector3s unbiased_ang_vel = base_imu.angular_velocity - imu_bias.gyroscope;
  const Eigen::Vector3s unbiased_accel = base_imu.acceleration - imu_bias.accelerometer;

  const Eigen::Vector3s local_gravity = imu_state.pose.so3().inverse() * gravity();
  const Eigen::Vector3s compensated_accel = unbiased_accel + local_gravity;

  // imu state update
  Eigen::Vector6s tau;
  tau.head<3>() = imu_state.linear_velocity * dt + compensated_accel * square(dt) / 2;
  tau.tail<3>() = unbiased_ang_vel * dt;
  imu_state.pose = imu_state.pose * Sophus::SE3s::exp(tau);
  imu_state.linear_velocity += compensated_accel * dt;
  imu_state.angular_velocity = unbiased_ang_vel;
  imu_state.time = base_imu.time;

  interval_stats.update(unbiased_ang_vel, unbiased_accel, compensated_accel);
  interval_stats.odom_acceleration_sum += imu_state.pose.so3() * unbiased_accel;
  if (interval_gyro.size() >= MAX_INTERVAL_GYRO_SAMPLES) {
    interval_gyro.erase(interval_gyro.begin(), interval_gyro.begin() + MAX_INTERVAL_GYRO_SAMPLES / 2);
  }
  interval_gyro.push_back({.time = base_imu.time, .angular_velocity = unbiased_ang_vel});

  last_real_imu_time_ = base_imu.time;
  last_real_base_imu_ang_vel_ = base_imu.angular_velocity;
}

void LIO::add_imu_measurement(const Sophus::SE3s& extrinsic_imu2base, const ImuControl& raw_imu) {
  if (extrinsic_imu2base.log().norm() < NEGLIGIBLE_EXTRINSIC) {
    add_imu_measurement(raw_imu);
    return;
  }

  if (last_real_imu_time_ < EPSILON_TIME) {
    // skip IMU message as we need a previous imu time for extrinsic compensation
    last_real_imu_time_ = raw_imu.time;
    return;
  }

  // accounting for the transport-rate
  ImuControl base_imu = raw_imu;
  const Sophus::SO3s& extrinsic_rotation = extrinsic_imu2base.so3();
  base_imu.angular_velocity = extrinsic_rotation * raw_imu.angular_velocity;

  const Eigen::Vector3s& lever_arm = -1 * extrinsic_imu2base.translation();
  const Nsec dt = raw_imu.time - last_real_imu_time_;

  const Eigen::Vector3s angular_acceleration = std::invoke([&]() -> Eigen::Vector3s {
    if (std::chrono::abs(dt) < std::chrono::microseconds(200)) {
      // if dt is less than the equivalent of a 5000 Hz imu, assuming zero ang accel,
      // causes numerical issues otherwise
      static bool warning_imu_too_close = false;
      if (!warning_imu_too_close) {
        spdlog::warn("Received IMU message with a very short delta to previous IMU message. Ignoring all such "
                     "messages.");
        warning_imu_too_close = true;
      }
      return Eigen::Vector3s::Zero();
    } else {
      const Eigen::Vector3s angular_acceleration =
          (base_imu.angular_velocity - last_real_base_imu_ang_vel_) / to_seconds(dt);
      return angular_acceleration;
    }
  });

  base_imu.acceleration = extrinsic_rotation * raw_imu.acceleration + angular_acceleration.cross(lever_arm) +
                          base_imu.angular_velocity.cross(base_imu.angular_velocity.cross(lever_arm));

  this->add_imu_measurement(base_imu);
}

// ============================ lidar ===============================

Vector3sVector LIO::register_scan(Vector3sVector scan, const Timestamps& timestamps) {
  if (scan.size() != timestamps.per_point.size()) {
    throw InputError("LiDAR timestamps size (" + std::to_string(timestamps.per_point.size()) +
                     ") does not match scan (" + std::to_string(scan.size()) + ").");
  }
  const Nsec current_lidar_time = timestamps.max;
  const auto total_start = std::chrono::steady_clock::now();
  frame_stats = {};
  frame_stats.input_points = scan.size();
  frame_stats.imu_count = interval_stats.imu_count;

  if (lidar_state.time < EPSILON_TIME) {
    return bootstrap_first_scan(scan, current_lidar_time);
  }

  if (current_lidar_time <= lidar_state.time) {
    throw InputError("Scan ends at " + std::to_string(to_seconds(current_lidar_time)) +
                     " s, not after the previous one at " + std::to_string(to_seconds(lidar_state.time)) + " s.");
  }

  if (std::chrono::abs(current_lidar_time - lidar_state.time) > MAX_SCAN_TO_SCAN_DELTA) {
    // the imu prior below extrapolates across the gap, and falls back to constant velocity if the imu dropped out too
    spdlog::warn("{} seconds delta to the previous scan. Extrapolating the motion prior across it.",
                 to_seconds(current_lidar_time - lidar_state.time));
  }

  const MotionPrior motion = motion_prior_from_imu(current_lidar_time);

  const bool have_gyro = gyro_covers(interval_gyro, lidar_state.time, current_lidar_time);
  const std::optional<IntervalTrajectory> trajectory =
      config.imu_rate_motion ? std::optional<IntervalTrajectory>(std::in_place, motion, interval_gyro, current_lidar_time)
                             : std::nullopt;
  Sophus::SE3s initial_guess =
      lidar_state.pose *
      (trajectory.has_value() ? trajectory->at(current_lidar_time) : relative_pose_at_time(motion, current_lidar_time));
  const Eigen::Vector3s gravity_step = tilt_correction(
      interval_stats, to_seconds<Scalar>(current_lidar_time - lidar_state.time), config.gravity_time_constant);
  initial_guess.so3() = Sophus::SO3s::exp(gravity_step) * initial_guess.so3();

  // body acceleration filter
  const auto kf_step = step_body_accel_filter(
      {.mean = mean_body_acceleration, .covariance = body_acceleration_covariance}, interval_stats, initial_guess.so3(),
      current_lidar_time - lidar_state.time, config.max_expected_jerk);
  mean_body_acceleration = kf_step.updated.mean;
  body_acceleration_covariance = kf_step.updated.covariance;

  frame_stats.accel_mag_variance = kf_step.info.has_value() ? kf_step.info->accel_mag_variance : -1;

  const bool imu_still = config.zero_velocity_update && !map.empty() && have_gyro &&
                         imu_reads_still(interval_gyro, interval_stats, imu_bias.gyroscope, config);
  if (!imu_still) {
    still_intervals_ = 0;
  } else {
    if (still_intervals_ == 0) {
      still_anchor_ = lidar_state.pose;
    }
    ++still_intervals_;
  }
  if (hold_cooldown_ > 0) {
    --hold_cooldown_;
  }
  bool hold = imu_still && hold_cooldown_ == 0 && held_scans_ > 0;
  if (imu_still && hold_cooldown_ == 0 && held_scans_ == 0 && still_intervals_ >= STILL_INTERVALS_BEFORE_HOLD) {
    hold = within(still_anchor_, lidar_state.pose, config.zupt_max_translation, config.zupt_max_rotation);
    if (!hold) {
      still_intervals_ = 0;
    }
  }

  const auto deskew_start = std::chrono::steady_clock::now();
  if (config.deskew && !hold) {
    SCOPED_PROFILER("Deskew");
    if (trajectory.has_value()) {
      deskew_scan(scan, timestamps, *trajectory);
    } else {
      const auto gap_to_scan_start = to_seconds<Scalar>(timestamps.min - motion.start_time);
      const MotionPrior scan_motion{
          .start_time = timestamps.min,
          .linear_velocity = motion.linear_velocity + motion.acceleration * gap_to_scan_start,
          .acceleration = motion.acceleration,
          .angular_velocity = motion.angular_velocity,
      };
      deskew_scan(scan, timestamps, scan_motion);
    }
  }
  frame_stats.deskew_ms = ms_since(deskew_start);
  const std::size_t input_scan_size = scan.size();
  const auto preprocess_start = std::chrono::steady_clock::now();
  auto preproc_result = preprocess_scan(std::move(scan), config);
  frame_stats.preprocess_ms = ms_since(preprocess_start);
  frame_stats.filtered_points = preproc_result.filtered_scan.size();
  frame_stats.keypoints = preproc_result.keypoints.size();

  if (preproc_result.keypoints.size() < 10) {
    const std::string error_msg = "Too few ICP keypoints: " + std::to_string(preproc_result.keypoints.size()) +
                                  " (input scan size: " + std::to_string(input_scan_size) +
                                  ", voxel_size: " + std::to_string(config.voxel_size) +
                                  "). Scan empty/corrupt, or downsampling is too aggressive.";
    throw InputError(error_msg);
  }

  const Vector3sVector& map_input = config.double_downsample ? preproc_result.map_points : preproc_result.keypoints;

  bool held = false;
  std::optional<Sophus::SE3s> checked_pose;
  if (hold) {
    held = true;
    if (held_scans_ % HOLD_CHECK_EVERY == 0) {
      const auto icp_start = std::chrono::steady_clock::now();
      try {
        checked_pose = arena.execute([&] {
          return icp(preproc_result.keypoints, map, lidar_state.pose, config, kf_step.info, 0, frame_stats);
        });
        const bool enough = 4 * frame_stats.correspondences >= preproc_result.keypoints.size();
        if (!enough ||
            within(lidar_state.pose, *checked_pose, config.zupt_max_translation, config.zupt_max_rotation)) {
          hold_checks_failed_ = 0;
        } else {
          ++hold_checks_failed_;
          const bool far_off = !within(lidar_state.pose, *checked_pose, HOLD_FAR_OFF_FACTOR * config.zupt_max_translation,
                                       HOLD_FAR_OFF_FACTOR * config.zupt_max_rotation);
          held = !far_off && hold_checks_failed_ < HOLD_CHECKS_FAILED_TO_RELEASE;
        }
      } catch (const RegistrationError&) {
        checked_pose.reset();
      }
      frame_stats.icp_ms = ms_since(icp_start);
      if (!held) {
        hold_cooldown_ = HOLD_COOLDOWN_SCANS;
        still_intervals_ = 0;
        hold_checks_failed_ = 0;
      }
    }
  }

  if (held) {
    ++held_scans_;
    if (held_scans_ >= HOLD_SCANS_BEFORE_BIAS_UPDATE && config.zupt_gyro_bias_gain > 0) {
      const Eigen::Vector3s raw_mean = interval_stats.angular_velocity_sum / interval_stats.imu_count + imu_bias.gyroscope;
      gyro_rest_bias += config.zupt_gyro_bias_gain * (raw_mean - gyro_rest_bias);
      imu_bias.gyroscope = gyro_rest_bias;
    }
    lidar_state.linear_velocity.setZero();
    lidar_state.angular_velocity.setZero();
    lidar_state.linear_acceleration.setZero();
    frame_stats.stationary = true;
  } else if (!map.empty()) {
    if (held_scans_ > 0) {
      imu_bias.gyroscope = gyro_rest_bias + gyro_motion_bias;
    }
    held_scans_ = 0;
    hold_checks_failed_ = 0;
    SCOPED_PROFILER("ICP");
    const auto icp_start = std::chrono::steady_clock::now();
    const auto dt = to_seconds<Scalar>(current_lidar_time - lidar_state.time);
    const Scalar rotation_prior_weight = have_gyro ? config.rotation_prior_weight : 0;
    const Sophus::SE3s registered_pose = checked_pose.has_value() ? *checked_pose : arena.execute([&] {
      return icp(preproc_result.keypoints, map, initial_guess, config, kf_step.info, rotation_prior_weight, frame_stats);
    });
    frame_stats.icp_ms = ms_since(icp_start);
    const bool finite = registered_pose.matrix3x4().allFinite();
    const bool plausible =
        finite && (!config.reject_implausible_registration ||
                   (static_cast<Scalar>(frame_stats.correspondences) >=
                        config.min_matched_ratio * static_cast<Scalar>(preproc_result.keypoints.size()) &&
                    within(initial_guess, registered_pose, config.max_correction_translation,
                           config.max_correction_rotation)));
    if (plausible) {
      rejected_scans_ = 0;
    } else {
      ++rejected_scans_;
      frame_stats.rejected = true;
      if (!finite || rejected_scans_ > config.max_rejected_scans) {
        rejected_scans_ = 0;
        throw RegistrationError(finite ? "registration kept disagreeing with the motion prediction"
                                       : "registration produced a non-finite pose");
      }
    }
    const Sophus::SE3s optimized_pose = plausible ? registered_pose : initial_guess;
    const Sophus::SE3s prior_correction = initial_guess.inverse() * optimized_pose;
    frame_stats.prior_correction_translation = prior_correction.translation().norm();
    frame_stats.prior_correction_rotation = prior_correction.so3().log().norm();
    if (plausible && rotation_prior_weight > 0 && config.gyro_bias_time_constant > 0) {
      const Eigen::Vector3s corrected = prior_correction.so3().log() + initial_guess.so3().inverse() * gravity_step;
      gyro_motion_bias -= corrected / config.gyro_bias_time_constant;
      gyro_motion_bias = gyro_motion_bias.cwiseMax(-MAX_GYRO_MOTION_BIAS).cwiseMin(MAX_GYRO_MOTION_BIAS);
      imu_bias.gyroscope = gyro_rest_bias + gyro_motion_bias;
    }

    // estimate velocities and accelerations from the new pose
    const Sophus::SE3s pose_delta = lidar_state.pose.inverse() * optimized_pose;
    const Eigen::Vector6s local_velocity = pose_delta.log() / dt;
    const Eigen::Vector3s local_linear_acceleration =
        (local_velocity.head<3>() - pose_delta.so3().inverse() * lidar_state.linear_velocity) / dt;

    // update
    lidar_state.pose = optimized_pose;
    lidar_state.linear_velocity = local_velocity.head<3>();
    lidar_state.angular_velocity = local_velocity.tail<3>();
    lidar_state.linear_acceleration = local_linear_acceleration;
  }
  // even if map is empty, time should still update
  lidar_state.time = current_lidar_time;
  // reset imu to last know lidar state
  imu_state = lidar_state;
  // reset imu averages
  interval_stats.reset();
  interval_gyro.clear();

  const auto map_start = std::chrono::steady_clock::now();
  if (!held && !frame_stats.rejected) {
    map.update(map_input, lidar_state.pose);
  }
  frame_stats.map_ms = ms_since(map_start);

  poses_with_timestamps.emplace_back(lidar_state.time, lidar_state.pose);
  frame_stats.total_ms = ms_since(total_start);

  return std::move(preproc_result.filtered_scan);
}

Vector3sVector
LIO::register_scan(const Sophus::SE3s& extrinsic_lidar2base, Vector3sVector scan, const Timestamps& timestamps) {
  if (extrinsic_lidar2base.log().norm() >= NEGLIGIBLE_EXTRINSIC) {
    transform_points(extrinsic_lidar2base, scan);
  }
  return register_scan(std::move(scan), timestamps);
}

void LIO::restart() {
  ++reset_count;
  config.initialization_phase = false;
  map.clear();
  lidar_state = {};
  imu_state = {};
  imu_bias = {};
  mean_body_acceleration.setZero();
  body_acceleration_covariance = Eigen::Matrix3s::Identity();
  interval_stats.reset();
  interval_gyro.clear();
  still_intervals_ = 0;
  held_scans_ = 0;
  hold_cooldown_ = 0;
  hold_checks_failed_ = 0;
  rejected_scans_ = 0;
  initialized_ = false;
}
} // namespace rko_lio::core

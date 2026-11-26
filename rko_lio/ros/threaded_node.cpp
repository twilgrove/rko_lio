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

#include "threaded_node.hpp"
#include "rko_lio/core/error.hpp"
#include "rko_lio/core/profiler.hpp"
#include "rko_lio/ros/utils/utils.hpp"

#include <ctime>

namespace {
using namespace std::literals;

inline double ms_between(const std::chrono::steady_clock::time_point start,
                         const std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

inline double thread_cpu_ms() {
  timespec ts{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return static_cast<double>(ts.tv_sec) * 1e3 + static_cast<double>(ts.tv_nsec) * 1e-6;
}
} // namespace

namespace rko_lio::ros {

ThreadedNode::ThreadedNode(const std::string& node_name, const rclcpp::NodeOptions& options)
    : BaseNode(node_name, options) {
  max_lidar_buffer_size = static_cast<size_t>(
      node->declare_parameter<int>("async.max_lidar_buffer_size", static_cast<int>(max_lidar_buffer_size)));
  max_imu_wait = std::chrono::duration_cast<core::Nsec>(std::chrono::duration<double>(
      node->declare_parameter<double>("async.max_imu_wait_s", core::to_seconds(max_imu_wait))));
  registration_thread = std::jthread([this] { registration_loop(); });
}

bool ThreadedNode::ready_to_register() const {
  if (lidar_buffer.empty()) {
    return false;
  }
  const core::Nsec scan_end = lidar_buffer.front().timestamps.max;
  if (!imu_buffer.empty() && imu_buffer.back().time > scan_end) {
    return true;
  }
  return max_imu_wait > core::Nsec::zero() && lidar_buffer.back().timestamps.max - scan_end > max_imu_wait;
}

void ThreadedNode::imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr& imu_msg) {
  if (!ensure_frame_and_extrinsics(imu_frame, imu_msg->header.frame_id, "IMU")) {
    return;
  }
  {
    const std::scoped_lock lock(buffer_mutex);
    imu_buffer.emplace(imu_msg_to_imu_data(*imu_msg, imu_gyro_scale, imu_accel_scale));
    atomic_can_process = ready_to_register();
  }
  if (atomic_can_process) {
    sync_condition_variable.notify_one();
  }
}

void ThreadedNode::lidar_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg) {
  if (!ensure_frame_and_extrinsics(lidar_frame, lidar_msg->header.frame_id, "LiDAR")) {
    return;
  }
  try {
    const auto arrival = std::chrono::steady_clock::now();
    LidarScan scan = process_lidar_msg(lidar_msg);
    scan.arrival = arrival;
    scan.raw_points = static_cast<std::size_t>(lidar_msg->height) * lidar_msg->width;
    scan.parse_ms = ms_between(arrival, std::chrono::steady_clock::now());
    {
      const std::scoped_lock lock(buffer_mutex);
      if (lidar_buffer.size() >= max_lidar_buffer_size) {
        RCLCPP_WARN_STREAM(node->get_logger(),
                           "Registration lidar buffer limit reached. Dropping oldest buffered frame.");
        lidar_buffer.pop();
      }
      lidar_buffer.push(std::move(scan));
      atomic_can_process = ready_to_register();
    }
    if (atomic_can_process) {
      sync_condition_variable.notify_one();
    }
  } catch (const core::InputError& ex) {
    RCLCPP_ERROR_STREAM(node->get_logger(), "Dropping scan: " << ex.what());
  }
}

void ThreadedNode::registration_loop() {
  while (rclcpp::ok() && atomic_node_running) {
    std::unique_lock buffer_lock(buffer_mutex);
    sync_condition_variable.wait(buffer_lock, [this] { return !atomic_node_running || atomic_can_process; });
    SCOPED_PROFILER("ROS Registration Loop");
    if (!atomic_node_running) {
      // node could have been killed after waiting on the cv
      break;
    }
    const auto loop_start = std::chrono::steady_clock::now();
    const double loop_cpu_start = thread_cpu_ms();
    LoopStats loop_stats;
    LidarScan scan = std::move(lidar_buffer.front());
    lidar_buffer.pop();
    loop_stats.queue_length = lidar_buffer.size();
    loop_stats.parse_ms = scan.parse_ms;
    loop_stats.raw_points = scan.raw_points;
    const auto arrival = scan.arrival;
    registration_busy = true;
    const core::Nsec end_stamp = scan.timestamps.max;
    const bool imu_reached_scan_end = !imu_buffer.empty() && imu_buffer.back().time > end_stamp;
    const core::Nsec last_registered = lio->lidar_state.time;
    for (; !imu_buffer.empty() && imu_buffer.front().time < end_stamp; imu_buffer.pop()) {
      const core::ImuControl& imu_data = imu_buffer.front();
      if (imu_data.time < last_registered) {
        continue;
      }
      lio->add_imu_measurement(extrinsic_imu2base, imu_data);
    }
    // check if there are more messages buffered already
    atomic_can_process = ready_to_register();
    buffer_lock.unlock(); // we dont touch the buffers anymore
    if (!imu_reached_scan_end) {
      RCLCPP_WARN_THROTTLE(node->get_logger(), *node->get_clock(), 2000,
                           "IMU is more than %.2f s behind the lidar. Registering without waiting for it.",
                           core::to_seconds(max_imu_wait));
    }
    const auto imu_done = std::chrono::steady_clock::now();
    loop_stats.imu_ms = ms_between(loop_start, imu_done);

    try {
      const core::Vector3sVector deskewed_scan = register_scan_locked(std::move(scan.points), scan.timestamps);
      const auto register_done = std::chrono::steady_clock::now();
      loop_stats.register_ms = ms_between(imu_done, register_done);
      if (!deskewed_scan.empty()) {
        // TODO: first scan is skipped and an empty scan is returned. improve how we handle this
        publish_tf(lio->lidar_state);
        publish_lidar_outputs(deskewed_scan);
      }
      const auto publish_done = std::chrono::steady_clock::now();
      loop_stats.publish_ms = ms_between(register_done, publish_done);
      loop_stats.loop_ms = ms_between(loop_start, publish_done);
      loop_stats.latency_ms = ms_between(arrival, publish_done);
      loop_stats.loop_cpu_ms = thread_cpu_ms() - loop_cpu_start;
      write_stats(loop_stats);
    } catch (const core::InputError& ex) {
      RCLCPP_ERROR_STREAM(node->get_logger(), "Dropping scan: " << ex.what());
    } catch (const core::RegistrationError& ex) {
      RCLCPP_ERROR_STREAM(node->get_logger(), "Registration failed (" << ex.what() << ").");
      if (reset_on_registration_error) {
        reset_odometry();
      } else {
        RCLCPP_WARN(node->get_logger(), "Set reset_on_registration_error:=true to reset odometry and continue.");
        throw;
      }
    } catch (const std::exception& ex) {
      RCLCPP_ERROR_STREAM(node->get_logger(), "Unexpected error while registering a scan (" << ex.what() << ").");
      if (reset_on_registration_error) {
        reset_odometry();
      } else {
        throw;
      }
    }
    registration_busy = false;
  }
  atomic_node_running = false;
}

ThreadedNode::~ThreadedNode() {
  atomic_node_running = false;
  sync_condition_variable.notify_all();
}

} // namespace rko_lio::ros

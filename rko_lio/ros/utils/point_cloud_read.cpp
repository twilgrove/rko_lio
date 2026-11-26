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

#include "point_cloud_read.hpp"
#include "rko_lio/core/error.hpp"
#include "rosbag.hpp"
#include <point_cloud_interfaces/msg/compressed_point_cloud2.hpp>
#include <rclcpp/utilities.hpp>
#include <rclcpp/version.h>
#include "time.hpp"
#include <spdlog/spdlog.h>
// stl
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace rko_lio::ros::utils {
using point_cloud_interfaces::msg::CompressedPointCloud2;
using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;

core::Vector3sVector point_cloud2_to_eigen(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) {
  const size_t point_count = static_cast<size_t>(msg->height) * msg->width;
  core::Vector3sVector points;
  points.reserve(point_count);
  sensor_msgs::PointCloud2ConstIterator<float> msg_x(*msg, "x");
  sensor_msgs::PointCloud2ConstIterator<float> msg_y(*msg, "y");
  sensor_msgs::PointCloud2ConstIterator<float> msg_z(*msg, "z");
  for (size_t i = 0; i < point_count; ++i, ++msg_x, ++msg_y, ++msg_z) {
    points.emplace_back(*msg_x, *msg_y, *msg_z);
  }
  return points;
}

RawScan point_cloud2_to_eigen_with_timestamps(const PointCloud2::ConstSharedPtr& msg) {
  using sensor_msgs::PointCloud2ConstIterator;
  // getting points and time in a single cycle loop
  const size_t point_count = static_cast<size_t>(msg->height) * msg->width;
  RawScan scan;
  scan.points.reserve(point_count);
  scan.timestamps.reserve(point_count);
  PointCloud2ConstIterator<float> msg_x(*msg, "x");
  PointCloud2ConstIterator<float> msg_y(*msg, "y");
  PointCloud2ConstIterator<float> msg_z(*msg, "z");

  const auto& timestamp_field = std::invoke([&msg]() -> PointField {
    for (const PointField& field : msg->fields) {
      if ((field.name == "t" || field.name == "time" || field.name == "timestamp" || field.name == "timestamps" ||
           field.name == "stamps") &&
          field.count != 0U) {
        return field;
      }
    }
    throw core::InputError(
        "No per-point timestamp field (t/time/timestamp/timestamps/stamps). Disable deskew, or add timestamps.");
  });

  // templated lambda (auto) ftw
  const auto extract_points_and_timestamps = [&](auto&& time_iter) {
    for (size_t i = 0; i < point_count; ++i, ++msg_x, ++msg_y, ++msg_z, ++time_iter) {
      scan.points.emplace_back(*msg_x, *msg_y, *msg_z);
      scan.timestamps.emplace_back(static_cast<double>(*time_iter));
    }
  };

  switch (timestamp_field.datatype) {
  case PointField::UINT32: {
    PointCloud2ConstIterator<uint32_t> msg_time(*msg, timestamp_field.name);
    extract_points_and_timestamps(msg_time);
    break;
  }
  case PointField::FLOAT32: {
    PointCloud2ConstIterator<float> msg_time(*msg, timestamp_field.name);
    extract_points_and_timestamps(msg_time);
    break;
  }
  case PointField::FLOAT64: {
    PointCloud2ConstIterator<double> msg_time(*msg, timestamp_field.name);
    extract_points_and_timestamps(msg_time);
    break;
  }
  default:
    throw core::InputError("Unsupported timestamp field datatype " + std::to_string(timestamp_field.datatype) +
                           ". Please open an issue.");
  }

  return scan;
}

LidarDeserializer::LidarDeserializer(const std::string_view type) {
  if (type == rosidl_generator_traits::name<CompressedPointCloud2>()) {
    codec.emplace();
  }
}

PointCloud2::ConstSharedPtr LidarDeserializer::operator()(const std::shared_ptr<rclcpp::SerializedMessage>& msg) const {
  if (!codec) {
    return deserialize<PointCloud2>(*msg);
  }
  if (!decoder) {
    const std::string format = deserialize<CompressedPointCloud2>(*msg)->format;
    decoder = codec->getDecoderByName(format);
    if (!decoder) {
      throw core::InputError("No point_cloud_transport plugin named '" + format + "' is installed.");
    }
  }
  const auto decoded = decoder->decode(msg);
  if (!decoded) {
    spdlog::warn("Dropping scan: could not decode it: {}", decoded.error());
    return nullptr;
  }
  return decoded->value_or(nullptr);
}

rclcpp::SubscriptionBase::SharedPtr
create_lidar_subscription(const rclcpp::Node::SharedPtr& node,
                          const std::string& topic,
                          const rclcpp::QoS& qos,
                          const std::function<void(const PointCloud2::ConstSharedPtr&)>& callback) {
  auto publishers = node->get_publishers_info_by_topic(topic);
  if (publishers.empty()) {
    spdlog::info("Waiting for a publisher on {}.", topic);
  }
  while (publishers.empty()) {
    if (!rclcpp::sleep_for(std::chrono::milliseconds(100), node->get_node_base_interface()->get_context())) {
      return nullptr;
    }
    publishers = node->get_publishers_info_by_topic(topic);
  }
  const std::string& type = publishers.front().topic_type();
  if (type == rosidl_generator_traits::name<CompressedPointCloud2>()) {
    return node->create_generic_subscription(
        topic, type, qos,
#if RCLCPP_VERSION_MAJOR >= 28
        [deserialize_lidar = LidarDeserializer(type), callback](std::unique_ptr<rclcpp::SerializedMessage> msg) {
          if (const auto cloud = deserialize_lidar(std::move(msg))) {
#else
        [deserialize_lidar = LidarDeserializer(type), callback](std::shared_ptr<rclcpp::SerializedMessage> msg) {
          if (const auto cloud = deserialize_lidar(msg)) {
#endif
            callback(cloud);
          }
        });
  }
  return node->create_subscription<PointCloud2>(topic, qos, callback);
}

namespace {
template <typename T>
inline T load(const std::uint8_t* address) {
  T value;
  std::memcpy(&value, address, sizeof(T));
  return value;
}

const PointField& find_field(const PointCloud2& msg, const std::string& name) {
  for (const PointField& field : msg.fields) {
    if (field.name == name && field.count != 0U) {
      return field;
    }
  }
  throw core::InputError("PointCloud2 has no '" + name + "' field.");
}

const PointField* find_time_field(const PointCloud2& msg) {
  for (const PointField& field : msg.fields) {
    if ((field.name == "t" || field.name == "time" || field.name == "timestamp" || field.name == "timestamps" ||
         field.name == "stamps") &&
        field.count != 0U) {
      return &field;
    }
  }
  return nullptr;
}

template <typename StampType>
FilteredScan read_scan_typed(const PointCloud2& msg,
                             const ScanFilter& filter,
                             const core::TimestampProcessingConfig& timestamp_config,
                             const std::size_t x_offset,
                             const std::size_t y_offset,
                             const std::size_t z_offset,
                             const std::size_t stamp_offset) {
  const std::size_t point_count = static_cast<std::size_t>(msg.height) * msg.width;
  const std::size_t step = msg.point_step;
  const std::uint8_t* const data = msg.data.data();

  FilteredScan scan;
  scan.message_points = point_count;
  scan.points.resize(point_count);
  thread_local std::vector<StampType> raw_stamps;
  raw_stamps.resize(point_count);

  const auto min_range_sq = static_cast<float>(filter.min_range * filter.min_range);
  const auto max_range_sq = static_cast<float>(std::min(filter.max_range * filter.max_range, 1e30));
  const Eigen::Vector3f box_min = filter.box_min;
  const Eigen::Vector3f box_max = filter.box_max;
  const bool drop_box = filter.drop_box;
  Eigen::Vector3s* const points = scan.points.data();
  StampType* const kept_stamps = raw_stamps.data();
  StampType raw_min = load<StampType>(data + stamp_offset);
  StampType raw_max = raw_min;
  std::size_t kept = 0;
  for (std::size_t i = 0; i < point_count; ++i) {
    const std::uint8_t* const point = data + i * step;
    const StampType stamp = load<StampType>(point + stamp_offset);
    raw_min = std::min(raw_min, stamp);
    raw_max = std::max(raw_max, stamp);
    const float x = load<float>(point + x_offset);
    const float y = load<float>(point + y_offset);
    const float z = load<float>(point + z_offset);
    const float range_sq = x * x + y * y + z * z;
    if (!(range_sq > min_range_sq && range_sq < max_range_sq)) {
      continue;
    }
    if (drop_box && x >= box_min.x() && x <= box_max.x() && y >= box_min.y() && y <= box_max.y() &&
        z >= box_min.z() && z <= box_max.z()) {
      continue;
    }
    points[kept] = Eigen::Vector3s(static_cast<core::Scalar>(x), static_cast<core::Scalar>(y),
                                   static_cast<core::Scalar>(z));
    kept_stamps[kept] = stamp;
    ++kept;
  }
  scan.points.resize(kept);

  const core::Nsec header_stamp = to_ns(msg.header.stamp);
  const core::TimestampConversion conversion = core::classify_timestamps(
      static_cast<double>(raw_min), static_cast<double>(raw_max), header_stamp, timestamp_config);
  scan.timestamps.min = conversion(static_cast<double>(raw_min));
  scan.timestamps.max = conversion(static_cast<double>(raw_max));
  scan.timestamps.per_point.resize(kept);
  core::Nsec* const stamps = scan.timestamps.per_point.data();
  for (std::size_t k = 0; k < kept; ++k) {
    stamps[k] = conversion(static_cast<double>(kept_stamps[k]));
  }
  return scan;
}
}

FilteredScan read_scan(const PointCloud2& msg,
                       const ScanFilter& filter,
                       const core::TimestampProcessingConfig& timestamp_config,
                       const bool use_point_stamps) {
  const PointField& x_field = find_field(msg, "x");
  const PointField& y_field = find_field(msg, "y");
  const PointField& z_field = find_field(msg, "z");
  if (x_field.datatype != PointField::FLOAT32 || y_field.datatype != PointField::FLOAT32 ||
      z_field.datatype != PointField::FLOAT32) {
    throw core::InputError("Only float32 x/y/z fields are supported.");
  }
  if (msg.is_bigendian) {
    throw core::InputError("Big endian point clouds are not supported.");
  }
  const std::size_t point_count = static_cast<std::size_t>(msg.height) * msg.width;
  if (msg.point_step == 0 || msg.data.size() < point_count * msg.point_step) {
    throw core::InputError("PointCloud2 data is shorter than height * width * point_step.");
  }
  if (point_count == 0) {
    throw core::InputError("LiDAR scan is empty.");
  }

  if (use_point_stamps) {
    const PointField* stamp_field = find_time_field(msg);
    if (stamp_field == nullptr) {
      throw core::InputError(
          "No per-point timestamp field (t/time/timestamp/timestamps/stamps). Disable deskew, or add timestamps.");
    }
    switch (stamp_field->datatype) {
    case PointField::UINT32:
      return read_scan_typed<std::uint32_t>(msg, filter, timestamp_config, x_field.offset, y_field.offset,
                                            z_field.offset, stamp_field->offset);
    case PointField::FLOAT32:
      return read_scan_typed<float>(msg, filter, timestamp_config, x_field.offset, y_field.offset, z_field.offset,
                                    stamp_field->offset);
    case PointField::FLOAT64:
      return read_scan_typed<double>(msg, filter, timestamp_config, x_field.offset, y_field.offset, z_field.offset,
                                     stamp_field->offset);
    default:
      throw core::InputError("Unsupported timestamp field datatype " + std::to_string(stamp_field->datatype) +
                             ". Please open an issue.");
    }
  }

  const core::Nsec header_stamp = to_ns(msg.header.stamp);
  core::TimestampProcessingConfig forced = timestamp_config;
  forced.force_absolute = true;
  FilteredScan scan =
      read_scan_typed<float>(msg, filter, forced, x_field.offset, y_field.offset, z_field.offset, x_field.offset);
  scan.timestamps.min = header_stamp;
  scan.timestamps.max = header_stamp;
  std::fill(scan.timestamps.per_point.begin(), scan.timestamps.per_point.end(), header_stamp);
  return scan;
}
} // namespace rko_lio::ros::utils

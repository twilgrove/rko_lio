#include "preprocess_scan.hpp"
#include "voxel_down_sample.hpp"

#include <cmath>

namespace rko_lio::core {

PreprocessingResult preprocess_scan(Vector3sVector scan, const LIO::Config& config) {
  if (!config.scan_is_range_filtered) {
    const Scalar min_range_sq = square(config.min_range);
    const Scalar max_range_sq = square(config.max_range);
    std::erase_if(scan, [&](const auto& point) {
      const Scalar range_sq = point.squaredNorm();
      return !(range_sq > min_range_sq && range_sq < max_range_sq);
    });
  }

  if (config.double_downsample) {
    Vector3sVector downsampled = voxel_down_sample(scan, config.voxel_size * static_cast<Scalar>(0.5));
    Vector3sVector keypoints = voxel_down_sample(downsampled, config.voxel_size * static_cast<Scalar>(1.5));
    return {.filtered_scan = std::move(scan), .keypoints = std::move(keypoints), .map_points = std::move(downsampled)};
  }
  Vector3sVector keypoints = voxel_down_sample(scan, config.voxel_size);
  return {.filtered_scan = std::move(scan), .keypoints = std::move(keypoints), .map_points = {}};
}

} // namespace rko_lio::core

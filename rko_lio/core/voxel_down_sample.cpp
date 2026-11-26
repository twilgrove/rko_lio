// MIT License
//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
// Stachniss.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "voxel_down_sample.hpp"
#include <Eigen/Core>
#include <algorithm>
#include <cstdint>
#include <sophus/se3.hpp>
#include <vector>

namespace {
using rko_lio::core::Scalar;

constexpr std::uint64_t EMPTY = ~std::uint64_t{0};
constexpr std::size_t MIN_SLOTS = 1U << 12U;

inline std::uint64_t pack(const Eigen::Vector3i& voxel) {
  constexpr std::uint64_t MASK = (std::uint64_t{1} << 21U) - 1;
  return ((static_cast<std::uint64_t>(voxel.x()) & MASK) << 42U) |
         ((static_cast<std::uint64_t>(voxel.y()) & MASK) << 21U) | (static_cast<std::uint64_t>(voxel.z()) & MASK);
}

class VoxelSet {
public:
  void reset() {
    if (slots_.empty()) {
      slots_.assign(MIN_SLOTS, EMPTY);
      shift_ = 64U - 12U;
    } else {
      std::fill(slots_.begin(), slots_.end(), EMPTY);
    }
    size_ = 0;
  }

  bool insert(const std::uint64_t key) {
    if (2 * (size_ + 1) > slots_.size()) {
      grow();
    }
    return place(key);
  }

private:
  bool place(const std::uint64_t key) {
    const std::size_t mask = slots_.size() - 1;
    std::size_t slot = static_cast<std::size_t>((key * 0x9E3779B97F4A7C15ULL) >> shift_);
    while (true) {
      const std::uint64_t held = slots_[slot];
      if (held == key) {
        return false;
      }
      if (held == EMPTY) {
        slots_[slot] = key;
        ++size_;
        return true;
      }
      slot = (slot + 1) & mask;
    }
  }

  void grow() {
    std::vector<std::uint64_t> old(2 * slots_.size(), EMPTY);
    old.swap(slots_);
    --shift_;
    size_ = 0;
    for (const std::uint64_t key : old) {
      if (key != EMPTY) {
        place(key);
      }
    }
  }

  std::vector<std::uint64_t> slots_;
  std::size_t size_ = 0;
  unsigned shift_ = 64U - 12U;
};
}

namespace rko_lio::core {
// if you need even better runtime-performance, consider using Luca Lobefaro's version of one cycle downsampling here:
// https://github.com/PRBonn/kiss-icp/pull/347
// although it does lead to worse odometry performance in certain situations

std::vector<Eigen::Vector3s> voxel_down_sample(const std::vector<Eigen::Vector3s>& points, const Scalar voxel_size) {
  const auto inv_voxel_size = static_cast<Scalar>(1.0 / voxel_size);
  thread_local VoxelSet seen;
  seen.reset();
  std::vector<Eigen::Vector3s> downsampled;
  downsampled.reserve(std::min<std::size_t>(points.size(), 1U << 14U));
  std::uint64_t previous_key = EMPTY;
  for (const Eigen::Vector3s& point : points) {
    const std::uint64_t key = pack(point_to_voxel(point, inv_voxel_size));
    if (key == previous_key) {
      continue;
    }
    previous_key = key;
    if (seen.insert(key)) {
      downsampled.push_back(point);
    }
  }
  return downsampled;
}

} // namespace rko_lio::core

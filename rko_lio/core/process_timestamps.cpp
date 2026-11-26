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

#include "process_timestamps.hpp"
#include "error.hpp"
#include <spdlog/spdlog.h>
// stl
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>

namespace {
using rko_lio::core::Nsec;
using rko_lio::core::TimestampConversion;
using rko_lio::core::TimestampProcessingConfig;
using rko_lio::core::Timestamps;
using rko_lio::core::TimestampVector;
using namespace std::chrono_literals;

// The timestamps are either in seconds or in nanoseconds, otherwise the user needs to specify the multiplier
TimestampConversion units_from_range(const double min_stamp, const double max_stamp, const double multiplier_to_seconds) {
  TimestampConversion conversion;
  if (multiplier_to_seconds < 1e-12) {
    const double scan_duration = std::abs(max_stamp - min_stamp);
    // 100 seconds is far more than the duration of any normal scan
    conversion.source_is_ns = (scan_duration > 100.0);
    conversion.multiplier_to_ns = conversion.source_is_ns ? 1.0 : 1e9;
  } else {
    conversion.multiplier_to_ns = multiplier_to_seconds * 1e9;
  }
  return conversion;
}
} // namespace

namespace rko_lio::core {
TimestampConversion classify_timestamps(const double raw_min,
                                        const double raw_max,
                                        const Nsec header_stamp,
                                        const TimestampProcessingConfig& config) {
  TimestampConversion conversion = units_from_range(raw_min, raw_max, config.multiplier_to_seconds);
  const Nsec min = conversion(raw_min);
  const Nsec max = conversion(raw_max);

  const bool absolute_stamps = config.force_absolute || (std::chrono::abs(header_stamp - min) < 1ms) ||
                               (std::chrono::abs(header_stamp - max) < 10ms);
  if (absolute_stamps) {
    return conversion;
  }

  const bool relative_stamps =
      config.force_relative || (std::chrono::abs(min) < 1ms) || (std::chrono::abs(max) < 10ms);
  if (relative_stamps) {
    conversion.offset = header_stamp;
    return conversion;
  }

  // Error-out for unique/unsupported cases
  const double header_s = to_seconds(header_stamp);
  const double min_s = to_seconds(min);
  const double max_s = to_seconds(max);
  spdlog::error("header: {:.6f} s\n"
                "points: {:.6f} .. {:.6f} s  (span {:.4f} s)\n"
                "|header-min| = {:.4f} s  (absolute if < 0.001 s)\n"
                "|header-max| = {:.4f} s  (absolute if < 0.010 s)\n"
                "|min| = {:.4f} s  (relative if < 0.001 s)\n"
                "|max| = {:.4f} s  (relative if < 0.010 s)\n"
                "multiplier_to_seconds: {}  (0 = autodetect)\n"
                "Set force_absolute if point times share the header clock; "
                "force_relative if they are offsets within the scan.",
                header_s, min_s, max_s, max_s - min_s, std::abs(header_s - min_s), std::abs(header_s - max_s),
                std::abs(min_s), std::abs(max_s), config.multiplier_to_seconds);
  throw InputError("Cannot classify LiDAR timestamps as absolute or relative.");
}

Timestamps process_timestamps(const std::vector<double>& raw_timestamps,
                              const Nsec header_stamp,
                              const TimestampProcessingConfig& config) {
  if (raw_timestamps.empty()) {
    throw InputError("LiDAR timestamps are empty.");
  }
  const auto& [min_it, max_it] = std::minmax_element(raw_timestamps.begin(), raw_timestamps.end());
  const TimestampConversion conversion = classify_timestamps(*min_it, *max_it, header_stamp, config);
  Timestamps timestamps{
      .min = conversion(*min_it),
      .max = conversion(*max_it),
      .per_point = TimestampVector(raw_timestamps.size()),
  };
  std::transform(raw_timestamps.cbegin(), raw_timestamps.cend(), timestamps.per_point.begin(), conversion);
  return timestamps;
}
} // namespace rko_lio::core

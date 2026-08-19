// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>

#include <cstdio>
#include <cstdlib>

#include "segment_anything_decoder_node.hpp"
#include "rclcpp/rclcpp.hpp"

// Objective: to validate construction of SegmentAnythingDecoderNode with
// default and custom parameters


TEST(segment_anything_decoder_node_test, test_default_construction)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  EXPECT_NO_THROW(
  {
    nvidia::isaac_ros::segment_anything::SegmentAnythingDecoderNode
    segment_anything_decoder_node(options);
  });
  rclcpp::shutdown();
}

TEST(segment_anything_decoder_node_test, test_custom_parameters)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("mask_width", static_cast<int16_t>(1920));
  options.append_parameter_override("mask_height", static_cast<int16_t>(1080));
  options.append_parameter_override("max_batch_size", static_cast<int16_t>(10));
  options.append_parameter_override("tensor_name", "output_mask");
  EXPECT_NO_THROW(
  {
    nvidia::isaac_ros::segment_anything::SegmentAnythingDecoderNode
    segment_anything_decoder_node(options);
  });
  rclcpp::shutdown();
}

TEST(segment_anything_decoder_node_test, test_empty_tensor_name)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("tensor_name", "");
  EXPECT_NO_THROW(
  {
    nvidia::isaac_ros::segment_anything::SegmentAnythingDecoderNode
    segment_anything_decoder_node(options);
  });
  rclcpp::shutdown();
}


int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  const int test_result = RUN_ALL_TESTS();
  // Workaround for EGL libraries included in VPI not tearing down
  // (tests pass, then a double-free corrupts the heap during static destruction).
  // Flush output and _Exit to skip global destructors and bypass the teardown abort.
  std::fflush(nullptr);
  std::_Exit(test_result);
}

// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "segment_anything_data_encoder_node.hpp"
#include "rclcpp/rclcpp.hpp"

// Objective: to cover code lines where exceptions are thrown and validate
// successful construction with valid parameters
// Approach: send Invalid Arguments for node parameters to trigger the exception,
// and verify valid parameters produce a working node


TEST(segment_anything_data_encoder_node_test, test_invalid_input_prompt_type)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("prompt_input_type", "");
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::segment_anything::SegmentAnythingDataEncoderNode
      segment_anything_data_encoder_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("Received invalid input prompt type"));
      throw;
    } catch (const rclcpp::exceptions::InvalidParameterValueException & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("No parameter value set"));
      throw;
    }
  }, std::invalid_argument);
  rclcpp::shutdown();
}

TEST(segment_anything_data_encoder_node_test, test_invalid_unsupported_prompt_type)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("prompt_input_type", "mask");
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::segment_anything::SegmentAnythingDataEncoderNode
      segment_anything_data_encoder_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("Received invalid input prompt type"));
      throw;
    } catch (const rclcpp::exceptions::InvalidParameterValueException & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("No parameter value set"));
      throw;
    }
  }, std::invalid_argument);
  rclcpp::shutdown();
}

TEST(segment_anything_data_encoder_node_test, test_valid_bbox_prompt_type)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("prompt_input_type", "bbox");
  EXPECT_NO_THROW(
  {
    nvidia::isaac_ros::segment_anything::SegmentAnythingDataEncoderNode
    segment_anything_data_encoder_node(options);
  });
  rclcpp::shutdown();
}

TEST(segment_anything_data_encoder_node_test, test_valid_point_prompt_type)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("prompt_input_type", "point");
  EXPECT_NO_THROW(
  {
    nvidia::isaac_ros::segment_anything::SegmentAnythingDataEncoderNode
    segment_anything_data_encoder_node(options);
  });
  rclcpp::shutdown();
}

TEST(segment_anything_data_encoder_node_test, test_default_parameters)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  EXPECT_NO_THROW(
  {
    nvidia::isaac_ros::segment_anything::SegmentAnythingDataEncoderNode
    segment_anything_data_encoder_node(options);
  });
  rclcpp::shutdown();
}

TEST(segment_anything_data_encoder_node_test, test_custom_parameters)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("prompt_input_type", "bbox");
  options.append_parameter_override("max_batch_size", 10);
  options.append_parameter_override("has_input_mask", true);
  options.append_parameter_override("orig_img_dims", std::vector<int64_t>{480, 640});
  EXPECT_NO_THROW(
  {
    nvidia::isaac_ros::segment_anything::SegmentAnythingDataEncoderNode
    segment_anything_data_encoder_node(options);
  });
  rclcpp::shutdown();
}


int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

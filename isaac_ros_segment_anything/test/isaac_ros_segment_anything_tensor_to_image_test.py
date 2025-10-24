# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import time

from isaac_ros_tensor_list_interfaces.msg import Tensor, TensorList, TensorShape
from isaac_ros_test import IsaacROSBaseTest
import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import launch_testing
import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import Image


@pytest.mark.rostest
def generate_test_description():
    """Generate test description for testing tensor to image node."""
    tensor_to_image_node = ComposableNode(
        name='tensor_to_image',
        package='isaac_ros_segment_anything',
        plugin='nvidia::isaac_ros::segment_anything::TensorToImageNode',
    )

    container = ComposableNodeContainer(
        name='segment_anything_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            tensor_to_image_node
        ],
        output='screen'
    )

    return launch.LaunchDescription([
        container,
        launch_testing.actions.ReadyToTest()
    ])


class TensorToImageTest(IsaacROSBaseTest):
    """Test for converting tensor to binary mask image."""

    def test_tensor_to_image_conversion(self):
        """Test tensor to image conversion with a sample tensor."""
        self.node = rclpy.create_node('tensor_to_image_test')

        # Create publishers and subscribers
        self.tensor_pub = self.node.create_publisher(
            TensorList, 'segmentation_tensor', 10)

        received_messages = []

        def image_callback(msg):
            self.node.get_logger().info('Received image message')
            received_messages.append(msg)

        self.image_sub = self.node.create_subscription(
            Image, 'binary_mask', image_callback, 10)

        # Create a sample tensor (simulating a segmentation mask)
        height, width = 480, 640
        test_tensor = np.zeros((1, 1, height, width), dtype=np.uint8)
        # Create a simple pattern (e.g., a rectangle) in the middle
        test_tensor[0, 0, height//4:3*height//4, width//4:3*width//4] = 255.0

        # Create TensorList message
        tensor_msg = TensorList()
        tensor_msg.tensors = []

        tensor = Tensor()
        tensor_shape = TensorShape()

        tensor_shape.rank = 4  # [batch, channel, height, width]
        tensor_shape.dims = [1, 1, height, width]

        tensor.shape = tensor_shape
        tensor.name = 'segmentation'
        tensor.data_type = 2  # UINT8
        tensor.strides = []  # Let pynitros handle strides
        tensor.data = test_tensor.tobytes()

        tensor_msg.tensors.append(tensor)

        # Wait for subscriptions to be ready
        time.sleep(2)

        # Publish tensor
        self.tensor_pub.publish(tensor_msg)

        # Wait for results (timeout after 10 seconds)
        end_time = time.time() + 10
        while time.time() < end_time and not received_messages:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        # Verify results
        self.assertTrue(len(received_messages) > 0, 'No image message received')

        received_image = received_messages[0]
        self.assertEqual(received_image.height, height)
        self.assertEqual(received_image.width, width)
        self.assertEqual(received_image.encoding, 'mono8')

        # Convert received image data to numpy array for verification
        received_data = np.frombuffer(received_image.data, dtype=np.uint8).reshape(height, width)

        # Verify the pattern is preserved (allowing for some tolerance due to thresholding)
        expected_mask = ((test_tensor[0] > 0.5) * 255).reshape(height, width)

        np.testing.assert_array_equal(received_data, expected_mask,
                                      'Received image does not match expected binary mask')

        self.node.get_logger().info('Test completed successfully')

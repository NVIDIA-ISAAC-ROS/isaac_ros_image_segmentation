# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import pathlib
import time

from isaac_ros_tensor_list_interfaces.msg import TensorList
from isaac_ros_test import IsaacROSBaseTest
import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

import pytest
import rclpy


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with all nodes to test Segment Anything package."""
    # Loads and runs segment_anything
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_dir = dir_path + '/model'

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    namespace = IsaacROSSegmentAnythingTest.generate_namespace()

    resize_node = ComposableNode(
        name='sam_resize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        namespace=namespace,
        parameters=[{
            'output_width': 1024,
            'output_height': 1024,
            'keep_aspect_ratio': True,
            'disable_padding': True,
            'input_width': 1200,
            'input_height': 632
        }],
        remappings=[('resize/image', 'resized_image')]
    )

    pad_node = ComposableNode(
        name='sam_pad_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::PadNode',
        namespace=namespace,
        parameters=[{
            'output_image_width': 1024,
            'output_image_height': 1024,
            'padding_type': 'BOTTOM_RIGHT'
        }],
        remappings=[('image', 'resized_image'),
                    ('padded_image', 'padded_image')]
    )

    image_format_converter_node = ComposableNode(
        name='image_format_converter_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        namespace=namespace,
        parameters=[{
            'image_width': 1024,
            'image_height': 1024,
            'encoding_desired': 'rgb8',
        }],
        remappings=[
            ('image_raw', 'padded_image'),
            ('image', 'color_converted_image'),
        ],
    )

    image_to_tensor_node = ComposableNode(
        name='image_to_tensor_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        namespace=namespace,
        parameters=[{
            'scale': True,
            'tensor_name': 'image',
        }],
        remappings=[
            ('image', 'color_converted_image'),
            ('tensor', 'tensor_proc/tensor')
        ]
    )

    normalize_node = ComposableNode(
        name='sam_normalize_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageTensorNormalizeNode',
        namespace=namespace,
        parameters=[{
            'mean': [0.5, 0.5, 0.5],
            'stddev': [0.5, 0.5, 0.5],
            'input_tensor_name': 'image',
            'output_tensor_name': 'image'
        }],
        remappings=[
            ('tensor', 'tensor_proc/tensor'),
            ('normalized_tensor', 'normalized_tensor'),
        ],
    )

    interleaved_to_planar_node = ComposableNode(
        name='sam_interleaved_to_planar_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        namespace=namespace,
        parameters=[{
            'input_tensor_shape': [1024, 1024, 3]
        }],
        remappings=[
            ('interleaved_tensor', 'normalized_tensor'),
            ('planar_tensor', 'planar_tensor'),
        ]
    )

    reshaper_node = ComposableNode(
        name='reshape_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        namespace=namespace,
        parameters=[{
            'output_tensor_name': 'input_tensor',
            'input_tensor_shape': [3, 1024, 1024],
            'output_tensor_shape': [1, 3, 1024, 1024]
        }],
        remappings=[
            ('tensor', 'planar_tensor'),
            ('reshaped_tensor', 'tensor_pub'),
        ],
    )

    dummy_mask_pub_node = ComposableNode(
        name='dummy_mask_pub',
        package='isaac_ros_segment_anything',
        plugin='nvidia::isaac_ros::segment_anything::DummyMaskPublisher',
        namespace=namespace
    )

    data_preprocessor_node = ComposableNode(
        name='sam_data_encoder_node',
        package='isaac_ros_segment_anything',
        plugin='nvidia::isaac_ros::segment_anything::SegmentAnythingDataEncoderNode',
        namespace=namespace,
        parameters=[{
            'prompt_input_type': 'point'
        }]
    )

    triton_node = ComposableNode(
        name='mask_decoder_triton_node',
        package='isaac_ros_triton',
        plugin='nvidia::isaac_ros::dnn_inference::TritonNode',
        namespace=namespace,
        parameters=[{
            'model_name': 'segment_anything',
            'model_repository_paths': [model_dir],
            'max_batch_size': 1,
            'input_tensor_names': ['input_tensor', 'points', 'labels', 'input_mask',
                                   'has_input_mask', 'orig_img_dims'],
            'input_binding_names': ['images', 'point_coords', 'point_labels', 'mask_input',
                                    'has_mask_input', 'orig_im_size'],
            'input_tensor_formats': ['nitros_tensor_list_nchw_rgb_f32'],
            'output_tensor_names': ['masks', 'iou', 'low_res_mask'],
            'output_binding_names': ['masks', 'iou_predictions', 'low_res_masks'],
            'output_tensor_formats': ['nitros_tensor_list_nchw_rgb_f32'],
        }],
        remappings=[('tensor_pub', 'tensor')])

    sam_decoder_node = ComposableNode(
        name='semgnet_anything_decoder_node',
        package='isaac_ros_segment_anything',
        plugin='nvidia::isaac_ros::segment_anything::SegmentAnythingDecoderNode',
        namespace=namespace,
        parameters=[{
            'mask_width': 1200,
            'mask_height': 632,
        }])

    rosbag_play = launch.actions.ExecuteProcess(
        cmd=['ros2', 'bag', 'play', '-l', os.path.dirname(__file__) +
             '/../../resources/rosbags/segment_anything_sample_data',
             '--remap',
             '/detectnet/detections:=' +
             IsaacROSSegmentAnythingTest.generate_namespace() + '/prompts',
             '/image:=' +
             IsaacROSSegmentAnythingTest.generate_namespace() + '/image',
             '/camera_info:=' +
             IsaacROSSegmentAnythingTest.generate_namespace() + '/camera_info'],
        output='screen'
    )

    nodes = [resize_node, pad_node, image_to_tensor_node, normalize_node,
             interleaved_to_planar_node, reshaper_node, dummy_mask_pub_node,
             image_format_converter_node, data_preprocessor_node, triton_node, sam_decoder_node]

    return IsaacROSSegmentAnythingTest.generate_test_description([
        ComposableNodeContainer(
            name='sam_container',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=nodes,
            namespace='segment_anything',
            output='screen',
            arguments=['--ros-args', '--log-level', 'info'],
        ),
        rosbag_play
    ])


class IsaacROSSegmentAnythingTest(IsaacROSBaseTest):
    """
    Proof-of-Life Test for Isaac ROS Segment Anything pipeline.

    1. Sets up ROS subscriber to listen to output channel of SAM decoder.
    2. Verify received tensors are the correct dimensions and size.
    """

    # Using default ROS-GXF Bridge output tensor channel configured in 'run_triton_inference' exe
    SUBSCRIBER_CHANNEL = 'segment_anything/raw_segmentation_mask'
    # The amount of seconds to allow Triton node to run before verifying received tensors
    # Will depend on time taken for Triton engine generation
    TEST_DURATION = 200.0

    DATA_TYPE = 2
    DIMENSIONS = [1, 1, 632, 1200]
    RANK = 4
    STRIDES = [1 * 632 * 1200 * 1, 632 * 1200 * 1, 1200 * 1, 1]
    DATA_LENGTH = STRIDES[0] * DIMENSIONS[0]

    filepath = pathlib.Path(os.path.dirname(__file__) +
                            '/../../resources/rosbags/segment_anything_sample_data')

    def test_segment_anything(self) -> None:
        self.node._logger.info('Starting Isaac ROS SAM POL Test')

        received_messages = {}

        subscriber_topic_namespace = self.generate_namespace(self.SUBSCRIBER_CHANNEL)
        test_subscribers = [
            (subscriber_topic_namespace, TensorList)
        ]

        subs = self.create_logging_subscribers(
            subscription_requests=test_subscribers,
            received_messages=received_messages,
            use_namespace_lookup=False,
            accept_multiple_messages=True,
            add_received_message_timestamps=True
        )

        try:

            end_time = time.time() + self.TEST_DURATION
            while time.time() < end_time:
                time.sleep(1)
                rclpy.spin_once(self.node, timeout_sec=0.1)
                if len(received_messages[subscriber_topic_namespace]) > 0:
                    break

            # Verify received tensors and log total number of tensors received
            num_tensors_received = len(received_messages[subscriber_topic_namespace])
            self.assertGreater(num_tensors_received, 0)

            for tensor_list, _ in received_messages[subscriber_topic_namespace]:
                tensor = tensor_list.tensors[0]
                self.assertEqual(
                    tensor.data_type, self.DATA_TYPE,
                    f'Unexpected tensor data type, expected: {self.DATA_TYPE} '
                    f'received: {tensor.data_type}'
                )
                self.assertEqual(
                    tensor.strides.tolist(), self.STRIDES,
                    f'Unexpected tensor strides, expected: {self.STRIDES} '
                    f'received: {tensor.strides}'
                )
                self.assertEqual(
                    len(tensor.data.tolist()), self.DATA_LENGTH,
                    f'Unexpected tensor length, expected: {self.DATA_LENGTH} '
                    f'received: {len(tensor.data)}'
                )

                shape = tensor.shape

                self.assertEqual(
                    shape.rank, self.RANK,
                    f'Unexpected tensor rank, expected: {self.RANK} received: {shape.rank}'
                )
                self.assertEqual(
                    shape.dims.tolist(), self.DIMENSIONS,
                    f'Unexpected tensor dimensions, expected: {self.DIMENSIONS} '
                    f'received: {shape.dims}'
                )

            # Log properties of last received tensor
            tensor_list, _ = received_messages[subscriber_topic_namespace][-1]
            tensor = tensor_list.tensors[0]
            shape = tensor.shape
            length = len(tensor.data.tolist())
            strides = tensor.strides.tolist()
            dimensions = shape.dims.tolist()

            self.node._logger.info(
                f'Received Tensor Properties:\n'
                f'Name: {tensor.name}\n'
                f'Data Type: {tensor.data_type}\n'
                f'Strides: {strides}\n'
                f'Byte Length: {length}\n'
                f'Rank: {shape.rank}\n'
                f'Dimensions: {dimensions}'
            )

            self.node._logger.info('Finished Isaac ROS Segment Anything Node POL Test')
        finally:
            [self.node.destroy_subscription(sub) for sub in subs]

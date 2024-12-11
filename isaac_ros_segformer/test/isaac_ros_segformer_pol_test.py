# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Proof-Of-Life test for the Isaac ROS Segformer package.

    1. Sets up DnnImageEncoderNode, TensorRTNode, UNetDecoderNode
    2. Loads a sample image and publishes it
    3. Subscribes to the relevant topics, waiting for an output from UNetDecodeNode
    4. Verifies that the received output sizes and encodings are correct (based on dummy model)

    Note: the data is not verified because the model is initialized with random weights
"""

import errno
import os
import pathlib
import time

from ament_index_python.packages import get_package_share_directory
from isaac_ros_test import IsaacROSBaseTest, JSONConversion, MockModelGenerator
import launch
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions.composable_node_container import ComposableNodeContainer
from launch_ros.descriptions.composable_node import ComposableNode
import launch_testing
import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import CameraInfo, Image
import torch

_TEST_CASE_NAMESPACE = 'segformer_node_test'


def generate_random_color_palette(num_classes):
    np.random.seed(0)
    L = []
    for i in range(num_classes):
        r = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        b = np.random.randint(0, 256)
        color = r << 16 | g << 8 | b
        L.append(color)
    return L


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description for testing relevant nodes."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_file_path = dir_path + '/dummy_model/model.dummy.onnx'
    try:
        os.remove('/tmp/trt_engine.plan')
        print('Deleted existing /tmp/trt_engine.plan')
    except OSError as e:
        if e.errno != errno.ENOENT:
            print('File exists but error deleting /tmp/trt_engine.plan')

    # Generate a mock model with SegFormer-like I/O
    MockModelGenerator.generate(
        input_bindings=[
            MockModelGenerator.Binding('input', [-1, 3, 512, 512], torch.float32)
        ],
        output_bindings=[
            MockModelGenerator.Binding('output', [-1, 1, 512, 512], torch.int64),
        ],
        output_onnx_path=model_file_path
    )

    encoder_dir = get_package_share_directory('isaac_ros_dnn_image_encoder')
    encoder_node_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(encoder_dir, 'launch', 'dnn_image_encoder.launch.py')]
        ),
        launch_arguments={
            'input_image_width': '1200',
            'input_image_height': '632',
            'network_image_width': '512',
            'network_image_height': '512',
            'enable_padding': 'True',
            'final_tensor_name': 'input_tensor',
            'tensor_output_topic': 'tensor_pub',
            'attach_to_shared_component_container': 'True',
            'component_container_name': 'unet_container',
            'dnn_image_encoder_namespace': IsaacROSSegformerPipelineTest.generate_namespace(
                _TEST_CASE_NAMESPACE),
        }.items(),
    )

    tensorrt_node = ComposableNode(
        name='TensorRTNode',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        namespace=IsaacROSSegformerPipelineTest.generate_namespace(_TEST_CASE_NAMESPACE),
        parameters=[{
            'model_file_path': model_file_path,
            'engine_file_path': '/tmp/trt_engine.plan',
            'input_tensor_names': ['input_tensor'],
            'input_binding_names': ['input'],
            'output_tensor_names': ['output_tensor'],
            'output_binding_names': ['output'],
            'verbose': False,
            'force_engine_update': False,
            'max_workspace_size': 104857600,
            'output_tensor_formats': ['nitros_tensor_list_nchw_rgb_f32'],
            'input_tensor_formats': ['nitros_tensor_list_nchw_rgb_f32']
        }])

    segformer_decoder_node = ComposableNode(
        name='SegformerDecoderNode',
        package='isaac_ros_unet',
        plugin='nvidia::isaac_ros::unet::UNetDecoderNode',
        namespace=IsaacROSSegformerPipelineTest.generate_namespace(_TEST_CASE_NAMESPACE),
        parameters=[{
            'network_output_type': 'argmax',
            'color_segmentation_mask_encoding': 'rgb8',
            'color_palette': generate_random_color_palette(20),  # 20 classes
        }])

    container = ComposableNodeContainer(
        name='unet_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[tensorrt_node, segformer_decoder_node],
        output='screen'
    )

    return IsaacROSSegformerPipelineTest.generate_test_description([
        container, encoder_node_launch,
        launch.actions.TimerAction(
            period=2.5, actions=[launch_testing.actions.ReadyToTest()])
    ])


class IsaacROSSegformerPipelineTest(IsaacROSBaseTest):
    """Validates a U-Net model with randomized weights with a sample output from Python."""

    filepath = pathlib.Path(os.path.dirname(__file__))
    MODEL_GENERATION_TIMEOUT_SEC = 300
    INIT_WAIT_SEC = 10
    MODEL_PATH = '/tmp/trt_engine.plan'

    @IsaacROSBaseTest.for_each_test_case()
    def test_image_segmentation(self, test_folder):
        self.node._logger.info(f'Generating model (timeout={self.MODEL_GENERATION_TIMEOUT_SEC}s)')
        start_time = time.time()
        ten_second_count = 1
        while not os.path.isfile(self.MODEL_PATH):
            time_diff = time.time() - start_time
            if time_diff > self.MODEL_GENERATION_TIMEOUT_SEC:
                self.fail('Model generation timed out')
            if time_diff > ten_second_count*10:
                self.node._logger.info(
                    f'Waiting for model generation to finish... ({int(time_diff)}s passed)')
                ten_second_count += 1
            time.sleep(1)

        # Wait for TensorRT engine
        time.sleep(self.INIT_WAIT_SEC)

        self.node._logger.info(
            f'Model generation was finished (took {(time.time() - start_time)}s)')

        """Expect the node to segment an image."""
        self.generate_namespace_lookup(
            ['image', 'camera_info', 'unet/raw_segmentation_mask',
             'unet/colored_segmentation_mask'], _TEST_CASE_NAMESPACE)
        image_pub = self.node.create_publisher(Image, self.namespaces['image'], self.DEFAULT_QOS)
        camera_info_pub = self.node.create_publisher(
            CameraInfo, self.namespaces['camera_info'], self.DEFAULT_QOS
        )
        received_messages = {}
        segmentation_mask_sub, color_segmentation_mask_sub = self.create_logging_subscribers(
            [('unet/raw_segmentation_mask', Image),
             ('unet/colored_segmentation_mask', Image)
             ], received_messages, accept_multiple_messages=True)

        EXPECTED_HEIGHT = 512
        EXPECTED_WIDTH = 512

        try:
            image = JSONConversion.load_image_from_json(test_folder / 'image.json')
            camera_info = CameraInfo()
            camera_info.distortion_model = 'plumb_bob'
            TIMEOUT = 60
            end_time = time.time() + TIMEOUT
            done = False
            while time.time() < end_time:
                image_pub.publish(image)
                camera_info_pub.publish(camera_info)
                rclpy.spin_once(self.node, timeout_sec=0.1)

                if len(received_messages['unet/raw_segmentation_mask']) > 0 and \
                   len(received_messages['unet/colored_segmentation_mask']) > 0:
                    done = True
                    break

            self.assertTrue(done, "Didn't receive output on unet/raw_segmentation_mask topic!")

            unet_mask = received_messages['unet/raw_segmentation_mask'][0]

            self.assertEqual(unet_mask.width, EXPECTED_WIDTH, 'Received incorrect width')
            self.assertEqual(unet_mask.height, EXPECTED_HEIGHT, 'Received incorrect height')
            self.assertEqual(unet_mask.encoding, 'mono8', 'Received incorrect encoding')

            unet_color_mask = received_messages['unet/colored_segmentation_mask'][0]
            self.assertEqual(unet_color_mask.width, EXPECTED_WIDTH,
                             'Received incorrect width for colored mask!')
            self.assertEqual(unet_color_mask.height, EXPECTED_HEIGHT,
                             'Received incorrect height for colored mask!')
            self.assertEqual(unet_color_mask.encoding, 'rgb8', 'Received incorrect encoding!')
        finally:
            self.assertTrue(self.node.destroy_subscription(segmentation_mask_sub))
            self.assertTrue(self.node.destroy_subscription(color_segmentation_mask_sub))
            self.assertTrue(self.node.destroy_publisher(image_pub))

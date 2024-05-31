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

import os

from ament_index_python.packages import get_package_share_directory
import launch
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Launch the DNN Image encoder, TensorRT node and UNet decoder node."""
    launch_args = [
        DeclareLaunchArgument(
            'input_image_width',
            default_value='1920',
            description='The input image width'),
        DeclareLaunchArgument(
            'input_image_height',
            default_value='1080',
            description='The input image height'),
        DeclareLaunchArgument(
            'network_image_width',
            default_value='512',
            description='The input image width that the network expects'),
        DeclareLaunchArgument(
            'network_image_height',
            default_value='512',
            description='The input image height that the network expects'),
        DeclareLaunchArgument(
            'encoder_image_mean',
            default_value='[0.5, 0.5, 0.5]',
            description='The mean for image normalization'),
        DeclareLaunchArgument(
            'encoder_image_stddev',
            default_value='[0.5, 0.5, 0.5]',
            description='The standard deviation for image normalization'),
        DeclareLaunchArgument(
            'model_file_path',
            default_value='',
            description='The absolute file path to the ONNX file'),
        DeclareLaunchArgument(
            'engine_file_path',
            default_value='',
            description='The absolute file path to the TensorRT engine file'),
        DeclareLaunchArgument(
            'input_tensor_names',
            default_value='["input_tensor"]',
            description='A list of tensor names to bound to the specified input binding names'),
        DeclareLaunchArgument(
            'input_binding_names',
            default_value='["input"]',
            description='A list of input tensor binding names (specified by model)'),
        DeclareLaunchArgument(
            'input_tensor_formats',
            default_value='["nitros_tensor_list_nchw_rgb_f32"]',
            description='The nitros format of the input tensors'),
        DeclareLaunchArgument(
            'output_tensor_names',
            default_value='["output_tensor"]',
            description='A list of tensor names to bound to the specified output binding names'),
        DeclareLaunchArgument(
            'output_binding_names',
            default_value='["output"]',
            description='A  list of output tensor binding names (specified by model)'),
        DeclareLaunchArgument(
            'output_tensor_formats',
            default_value='["nitros_tensor_list_nchw_rgb_f32"]',
            description='The nitros format of the output tensors'),
        DeclareLaunchArgument(
            'tensorrt_verbose',
            default_value='False',
            description='Whether TensorRT should verbosely log or not'),
        DeclareLaunchArgument(
            'force_engine_update',
            default_value='False',
            description='Whether TensorRT should update the TensorRT engine file or not'),
        DeclareLaunchArgument(
            'network_output_type',
            default_value='argmax',
            description='The output type that the network provides (softmax, sigmoid or argmax)'),
        DeclareLaunchArgument(
            'color_segmentation_mask_encoding',
            default_value='rgb8',
            description='The image encoding of the colored segmentation mask (rgb8 or bgr8)'),
        DeclareLaunchArgument(
            'mask_width',
            default_value='512',
            description='The width of the segmentation mask'),
        DeclareLaunchArgument(
            'mask_height',
            default_value='512',
            description='The height of the segmentation mask'),
        DeclareLaunchArgument(
            'max_workspace_size',
            default_value='104857600',
            description='Max TensorRT workspace size in bytes'),
    ]

    # DNN Image Encoder parameters
    input_image_width = LaunchConfiguration('input_image_width')
    input_image_height = LaunchConfiguration('input_image_height')
    network_image_width = LaunchConfiguration('network_image_width')
    network_image_height = LaunchConfiguration('network_image_height')
    encoder_image_mean = LaunchConfiguration('encoder_image_mean')
    encoder_image_stddev = LaunchConfiguration('encoder_image_stddev')

    # TensorRT parameters
    model_file_path = LaunchConfiguration('model_file_path')
    engine_file_path = LaunchConfiguration('engine_file_path')
    input_tensor_names = LaunchConfiguration('input_tensor_names')
    input_binding_names = LaunchConfiguration('input_binding_names')
    input_tensor_formats = LaunchConfiguration('input_tensor_formats')
    output_tensor_names = LaunchConfiguration('output_tensor_names')
    output_binding_names = LaunchConfiguration('output_binding_names')
    output_tensor_formats = LaunchConfiguration('output_tensor_formats')
    tensorrt_verbose = LaunchConfiguration('tensorrt_verbose')
    force_engine_update = LaunchConfiguration('force_engine_update')
    max_workspace_size = LaunchConfiguration('max_workspace_size')

    # U-Net Decoder parameters
    network_output_type = LaunchConfiguration('network_output_type')
    color_segmentation_mask_encoding = LaunchConfiguration('color_segmentation_mask_encoding')
    mask_width = LaunchConfiguration('mask_width')
    mask_height = LaunchConfiguration('mask_height')

    container_name = 'segformer_container'

    # Parameters preconfigured for PeopleSemSegNet.
    encoder_dir = get_package_share_directory('isaac_ros_dnn_image_encoder')
    encoder_node_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(encoder_dir, 'launch', 'dnn_image_encoder.launch.py')]
        ),
        launch_arguments={
            'input_image_width': input_image_width,
            'input_image_height': input_image_height,
            'network_image_width': network_image_width,
            'network_image_height':  network_image_height,
            'image_mean': encoder_image_mean,
            'image_stddev': encoder_image_stddev,
            'enable_padding': 'True',
            'image_input_topic': '/image',
            'camera_info_input_topic': '/camera_info',
            'tensor_output_topic': '/tensor_pub',
            'attach_to_shared_component_container': 'True',
            'component_container_name': container_name,
        }.items(),
    )

    tensorrt_node = ComposableNode(
        name='tensor_rt_node',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'model_file_path': model_file_path,
            'engine_file_path': engine_file_path,
            'input_tensor_names': input_tensor_names,
            'input_binding_names': input_binding_names,
            'input_tensor_formats': input_tensor_formats,
            'output_tensor_names': output_tensor_names,
            'output_binding_names': output_binding_names,
            'output_tensor_formats': output_tensor_formats,
            'verbose': tensorrt_verbose,
            'force_engine_update': force_engine_update,
            'max_workspace_size': max_workspace_size
        }])

    segformer_decoder_node = ComposableNode(
        name='segformer_decoder_node',
        package='isaac_ros_unet',
        plugin='nvidia::isaac_ros::unet::UNetDecoderNode',
        parameters=[{
            'network_output_type': network_output_type,
            'color_segmentation_mask_encoding': color_segmentation_mask_encoding,
            'mask_width': mask_width,
            'mask_height': mask_height,
            'color_palette': [0x556B2F, 0x800000, 0x008080, 0x000080, 0x9ACD32, 0xFF0000, 0xFF8C00,
                              0xFFD700, 0x00FF00, 0xBA55D3, 0x00FA9A, 0x00FFFF, 0x0000FF, 0xF08080,
                              0xFF00FF, 0x1E90FF, 0xDDA0DD, 0xFF1493, 0x87CEFA, 0xFFDEAD],
        }],
        remappings=[
            ('unet/colored_segmentation_mask', 'segformer/colored_segmentation_mask'),
            ('unet/raw_segmentation_mask', 'segformer/raw_segmentation_mask')
        ])

    container = ComposableNodeContainer(
        name=container_name,
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[tensorrt_node, segformer_decoder_node],
        output='screen'
    )

    final_launch_description = launch_args + [container, encoder_node_launch]
    return launch.LaunchDescription(final_launch_description)

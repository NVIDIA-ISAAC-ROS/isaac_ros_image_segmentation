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
from typing import Any, Dict

from ament_index_python.packages import get_package_share_directory
from isaac_ros_examples import IsaacROSLaunchFragment
import launch
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.descriptions import ComposableNode


class IsaacROSUNetLaunchFragment(IsaacROSLaunchFragment):

    @staticmethod
    def get_composable_nodes(interface_specs: Dict[str, Any]) -> Dict[str, ComposableNode]:
        # Tensor RT parameters
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

        # U-Net Decoder parameters
        network_output_type = LaunchConfiguration('network_output_type')
        color_segmentation_mask_encoding = LaunchConfiguration('color_segmentation_mask_encoding')
        mask_width = LaunchConfiguration('mask_width')
        mask_height = LaunchConfiguration('mask_height')

        # Alpha Blend parameters
        alpha = LaunchConfiguration('alpha')

        return {
            'unet_inference_node': ComposableNode(
                name='unet_inference',
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
                    'force_engine_update': force_engine_update
                }]
            ),
            'unet_decoder_node': ComposableNode(
                name='unet_decoder',
                package='isaac_ros_unet',
                plugin='nvidia::isaac_ros::unet::UNetDecoderNode',
                parameters=[{
                    'network_output_type': network_output_type,
                    'color_segmentation_mask_encoding': color_segmentation_mask_encoding,
                    'mask_width': mask_width,
                    'mask_height': mask_height,
                    'color_palette': [0x556B2F, 0x800000, 0x008080, 0x000080, 0x9ACD32,
                                      0xFF0000, 0xFF8C00, 0xFFD700, 0x00FF00, 0xBA55D3,
                                      0x00FA9A, 0x00FFFF, 0x0000FF, 0xF08080, 0xFF00FF,
                                      0x1E90FF, 0xDDA0DD, 0xFF1493, 0x87CEFA, 0xFFDEAD],
                }],
            ),
            'alpha_blend_node': ComposableNode(
                name='alpha_blend',
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::AlphaBlendNode',
                parameters=[{
                    'alpha': alpha,
                    'mask_queue_size': 50,
                    'image_queue_size': 50,
                    'sync_queue_size': 50,
                }],
                remappings=[
                    ('mask_input', '/unet/colored_segmentation_mask'),
                    ('image_input', '/unet_encoder/converted/image'),
                    ('blended_image', '/segmentation_image_overlay')
                ],
            )
        }

    @staticmethod
    def get_launch_actions(interface_specs: Dict[str, Any]) -> \
            Dict[str, launch.actions.OpaqueFunction]:

        # DNN Image Encoder parameters
        input_qos = LaunchConfiguration('input_qos')
        network_image_width = LaunchConfiguration('network_image_width')
        network_image_height = LaunchConfiguration('network_image_height')
        encoder_image_mean = LaunchConfiguration('encoder_image_mean')
        encoder_image_stddev = LaunchConfiguration('encoder_image_stddev')
        use_planar_input = LaunchConfiguration('use_planar_input')

        encoder_dir = get_package_share_directory('isaac_ros_unet')
        return {
            'input_qos': DeclareLaunchArgument(
                'input_qos',
                default_value='DEFAULT',
                description='The QoS profile of the resize node subscriber'),
            'network_image_width': DeclareLaunchArgument(
                'network_image_width',
                default_value='960',
                description='The input image width that the network expects'),
            'network_image_height': DeclareLaunchArgument(
                'network_image_height',
                default_value='544',
                description='The input image height that the network expects'),
            'encoder_image_mean': DeclareLaunchArgument(
                'encoder_image_mean',
                default_value='[0.5, 0.5, 0.5]',
                description='The mean for image normalization'),
            'encoder_image_stddev': DeclareLaunchArgument(
                'encoder_image_stddev',
                default_value='[0.5, 0.5, 0.5]',
                description='The standard deviation for image normalization'),
            'use_planar_input': DeclareLaunchArgument(
                'use_planar_input',
                default_value='True',
                description='Whether the input image should be in planar format or not'),
            'model_file_path': DeclareLaunchArgument(
                'model_file_path',
                default_value='',
                description='The absolute file path to the ONNX file'),
            'engine_file_path': DeclareLaunchArgument(
                'engine_file_path',
                default_value='',
                description='The absolute file path to the TensorRT engine file'),
            'input_tensor_names': DeclareLaunchArgument(
                'input_tensor_names',
                default_value='["input_tensor"]',
                description='A list of tensor names to bound to the specified input binding names'
            ),
            'input_binding_names': DeclareLaunchArgument(
                'input_binding_names',
                default_value='["input_2:0"]',
                description='A list of input tensor binding names (specified by model)'),
            'input_tensor_formats': DeclareLaunchArgument(
                'input_tensor_formats',
                default_value='["nitros_tensor_list_nchw_bgr_f32"]',
                description='The nitros format of the input tensors'),
            'output_tensor_names': DeclareLaunchArgument(
                'output_tensor_names',
                default_value='["output_tensor"]',
                description='A list of tensor names to bound to the specified output binding names'
            ),
            'output_binding_names': DeclareLaunchArgument(
                'output_binding_names',
                default_value='["argmax_1"]',
                description='A  list of output tensor binding names (specified by model)'),
            'output_tensor_formats': DeclareLaunchArgument(
                'output_tensor_formats',
                default_value='["nitros_tensor_list_nhwc_bgr_f32"]',
                description='The nitros format of the output tensors'),
            'tensorrt_verbose': DeclareLaunchArgument(
                'tensorrt_verbose',
                default_value='False',
                description='Whether TensorRT should verbosely log or not'),
            'force_engine_update': DeclareLaunchArgument(
                'force_engine_update',
                default_value='False',
                description='Whether TensorRT should update the TensorRT engine file or not'),
            'object_name': DeclareLaunchArgument(
                'object_name',
                default_value='Ketchup',
                description='The object class that the DOPE network is detecting'),
            'network_output_type': DeclareLaunchArgument(
                'network_output_type',
                default_value='argmax',
                choices=['softmax', 'sigmoid', 'argmax'],
                description='The output type that the network provides'),
            'color_segmentation_mask_encoding': DeclareLaunchArgument(
                'color_segmentation_mask_encoding',
                default_value='rgb8',
                description='The image encoding of the colored segmentation mask (rgb8 or bgr8)'),
            'mask_width': DeclareLaunchArgument(
                'mask_width',
                default_value='960',
                description='The width of the segmentation mask'),
            'mask_height': DeclareLaunchArgument(
                'mask_height',
                default_value='544',
                description='The height of the segmentation mask'),
            'alpha': DeclareLaunchArgument(
                'alpha',
                default_value='0.5',
                description='The alpha value for alpha blending.',
            ),
            'dope_encoder_launch': IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [os.path.join(encoder_dir, 'launch', 'isaac_ros_unet_encoder.launch.py')]
                ),
                launch_arguments={
                    'input_qos': input_qos,
                    'input_image_width': str(interface_specs['camera_resolution']['width']),
                    'input_image_height': str(interface_specs['camera_resolution']['height']),
                    'network_image_width': network_image_width,
                    'network_image_height': network_image_height,
                    'image_mean': encoder_image_mean,
                    'image_stddev': encoder_image_stddev,
                    'enable_padding': 'True',
                    'use_planar_input': use_planar_input,
                    'attach_to_shared_component_container': 'True',
                    'component_container_name': '/isaac_ros_examples/container',
                    'dnn_image_encoder_namespace': 'unet_encoder',
                    'image_input_topic': '/image_rect',
                    'camera_info_input_topic': '/camera_info_rect',
                    'tensor_output_topic': '/tensor_pub',
                }.items(),
            ),
        }

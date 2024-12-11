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

from typing import Any, Dict

from isaac_ros_examples import IsaacROSLaunchFragment
import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.descriptions import ComposableNode


SAM_MODEL_INPUT_SIZE = 1024  # SAM model accept 1024x1024 image


class IsaacROSSegmentAnythingLaunchFragment(IsaacROSLaunchFragment):

    @staticmethod
    def get_composable_nodes(interface_specs: Dict[str, Any]) -> Dict[str, ComposableNode]:
        # DNN Image Encoder parameters
        sam_encoder_image_mean = LaunchConfiguration('sam_encoder_image_mean')
        sam_encoder_image_stddev = LaunchConfiguration('sam_encoder_image_stddev')

        # Triton parameters
        sam_model_name = LaunchConfiguration('sam_model_name')
        sam_model_repository_paths = LaunchConfiguration('sam_model_repository_paths')
        sam_max_batch_size = LaunchConfiguration('sam_max_batch_size')
        sam_input_tensor_names = LaunchConfiguration('sam_input_tensor_names')
        sam_input_binding_names = LaunchConfiguration('sam_input_binding_names')
        sam_input_tensor_formats = LaunchConfiguration('sam_input_tensor_formats')
        sam_output_tensor_names = LaunchConfiguration('sam_output_tensor_names')
        sam_output_binding_names = LaunchConfiguration('sam_output_binding_names')
        sam_output_tensor_formats = LaunchConfiguration('sam_output_tensor_formats')

        prompt_input_type = LaunchConfiguration('prompt_input_type')
        has_input_mask = LaunchConfiguration('has_input_mask')
        orig_img_dims = [interface_specs['input_image']['height'],
                         interface_specs['input_image']['width']]
        input_image_encoding = LaunchConfiguration('input_image_encoding')

        return {

            'sam_resize_node': ComposableNode(
                name='sam_resize_node',
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::ResizeNode',
                parameters=[{
                    'output_width': SAM_MODEL_INPUT_SIZE,
                    'output_height': SAM_MODEL_INPUT_SIZE,
                    'keep_aspect_ratio': True,
                    'disable_padding': True,
                    'encoding_desired': input_image_encoding,
                    'input_width': interface_specs['input_image']['width'],
                    'input_height': interface_specs['input_image']['height']
                }],
                remappings=[('/image', interface_specs['subscribed_topics']['image']),
                            ('/camera_info', interface_specs['subscribed_topics']['camera_info']),
                            ('resize/image', '/segment_anything/resized_image')]
            ),

            'sam_pad_node': ComposableNode(
                name='sam_pad_node',
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::PadNode',
                parameters=[{
                    'output_image_width': SAM_MODEL_INPUT_SIZE,
                    'output_image_height': SAM_MODEL_INPUT_SIZE,
                    'padding_type': 'BOTTOM_RIGHT'
                }],
                remappings=[('/image', '/segment_anything/resized_image'),
                            ('/padded_image', '/segment_anything/padded_image')]
            ),

            'sam_image_format_converter_node': ComposableNode(
                name='sam_image_format_converter_node',
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
                parameters=[{
                    'image_width': SAM_MODEL_INPUT_SIZE,
                    'image_height': SAM_MODEL_INPUT_SIZE,
                    'encoding_desired': 'rgb8',
                }],
                remappings=[
                    ('image_raw', '/segment_anything/padded_image'),
                    ('image', '/segment_anything/color_converted_image'),
                ],
            ),

            'sam_image_to_tensor_node': ComposableNode(
                name='sam_image_to_tensor_node',
                package='isaac_ros_tensor_proc',
                plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
                parameters=[{
                    'scale': True,
                    'tensor_name': 'image',
                }],
                remappings=[
                    ('image', '/segment_anything/color_converted_image'),
                    ('tensor', '/segment_anything/image_tensor')
                ]
            ),

            'sam_normalize_node': ComposableNode(
                name='sam_normalize_node',
                package='isaac_ros_tensor_proc',
                plugin='nvidia::isaac_ros::dnn_inference::ImageTensorNormalizeNode',
                parameters=[{
                    'mean': sam_encoder_image_mean,
                    'stddev': sam_encoder_image_stddev,
                    'input_tensor_name': 'image',
                    'output_tensor_name': 'image'
                }],
                remappings=[
                    ('tensor', '/segment_anything/image_tensor'),
                    ('normalized_tensor', '/segment_anything/normalized_tensor'),
                ],
            ),

            'sam_interleaved_to_planar_node': ComposableNode(
                name='sam_interleaved_to_planar_node',
                package='isaac_ros_tensor_proc',
                plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
                parameters=[{
                    'input_tensor_shape': [SAM_MODEL_INPUT_SIZE, SAM_MODEL_INPUT_SIZE, 3]
                }],
                remappings=[
                    ('interleaved_tensor', '/segment_anything/normalized_tensor'),
                    ('planar_tensor', '/segment_anything/planar_tensor'),
                ]
            ),

            'sam_reshaper_node': ComposableNode(
                name='reshape_node',
                package='isaac_ros_tensor_proc',
                plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
                parameters=[{
                    'output_tensor_name': 'input_tensor',
                    'input_tensor_shape': [3, SAM_MODEL_INPUT_SIZE, SAM_MODEL_INPUT_SIZE],
                    'output_tensor_shape': [1, 3, SAM_MODEL_INPUT_SIZE, SAM_MODEL_INPUT_SIZE]
                }],
                remappings=[
                    ('tensor', '/segment_anything/planar_tensor'),
                    ('reshaped_tensor', '/segment_anything/tensor_pub'),
                ],
            ),

            'sam_dummy_mask_pub_node': ComposableNode(
                name='sam_dummy_mask_pub_node',
                package='isaac_ros_segment_anything',
                plugin='nvidia::isaac_ros::segment_anything::DummyMaskPublisher',
                remappings=[('tensor_pub', '/segment_anything/tensor_pub')]
            ),

            'sam_data_preprocessor_node': ComposableNode(
                name='sam_data_encoder_node',
                package='isaac_ros_segment_anything',
                plugin='nvidia::isaac_ros::segment_anything::SegmentAnythingDataEncoderNode',
                parameters=[{
                    'prompt_input_type': prompt_input_type,
                    'has_input_mask': has_input_mask,
                    'max_batch_size': sam_max_batch_size,
                    'orig_img_dims': orig_img_dims
                }],
                remappings=[('prompts', interface_specs['subscribed_topics']['prompt']),
                            ('tensor_pub', '/segment_anything/tensor_pub'),
                            ('tensor', '/segment_anything/encoded_data')]
            ),

            'sam_triton_node': ComposableNode(
                name='sam_triton_node',
                package='isaac_ros_triton',
                plugin='nvidia::isaac_ros::dnn_inference::TritonNode',
                parameters=[{
                    'model_name': sam_model_name,
                    'model_repository_paths': sam_model_repository_paths,
                    'max_batch_size': 1,
                    'input_tensor_names': sam_input_tensor_names,
                    'input_binding_names': sam_input_binding_names,
                    'input_tensor_formats': sam_input_tensor_formats,
                    'output_tensor_names': sam_output_tensor_names,
                    'output_binding_names': sam_output_binding_names,
                    'output_tensor_formats': sam_output_tensor_formats,
                }],
                remappings=[('tensor_pub', '/segment_anything/encoded_data'),
                            ('tensor_sub', '/segment_anything/tensor_sub')]
            ),

            'sam_decoder_node': ComposableNode(
                name='semgnet_anything_decoder_node',
                package='isaac_ros_segment_anything',
                plugin='nvidia::isaac_ros::segment_anything::SegmentAnythingDecoderNode',
                parameters=[{
                    'mask_width': interface_specs['input_image']['width'],
                    'mask_height': interface_specs['input_image']['height'],
                    'max_batch_size': sam_max_batch_size
                }],
                remappings=[('tensor_sub', '/segment_anything/tensor_sub')])

        }

    @staticmethod
    def get_launch_actions(interface_specs: Dict[str, Any]) -> \
            Dict[str, launch.actions.OpaqueFunction]:

        return {
            'sam_encoder_image_mean': DeclareLaunchArgument(
                'sam_encoder_image_mean',
                default_value='[0.485, 0.456, 0.406]',
                description='The mean for image normalization'),
            'sam_encoder_image_stddev': DeclareLaunchArgument(
                'sam_encoder_image_stddev',
                default_value='[0.229, 0.224, 0.225]',
                description='The standard deviation for image normalization'),
            'sam_model_name': DeclareLaunchArgument(
                'sam_model_name',
                default_value='segment_anything',
                description='The name of the model'),
            'sam_model_repository_paths': DeclareLaunchArgument(
                'sam_model_repository_paths',
                default_value='["/tmp/models"]',
                description='The absolute path to the repository of models'),
            'sam_max_batch_size': DeclareLaunchArgument(
                'sam_max_batch_size',
                default_value='20',
                description='The maximum allowed batch size of the model'),
            'sam_input_tensor_names': DeclareLaunchArgument(
                'sam_input_tensor_names',
                default_value='["input_tensor","points","labels","input_mask", \
                "has_input_mask","orig_img_dims"]',
                description='A list of tensor names to bound to the specified input binding name'),
            'sam_input_binding_names': DeclareLaunchArgument(
                'sam_input_binding_names',
                default_value='["images","point_coords","point_labels","mask_input", \
                "has_mask_input","orig_im_size"]',
                description='A list of input tensor binding names (specified by model)'),
            'sam_input_tensor_formats': DeclareLaunchArgument(
                'sam_input_tensor_formats',
                default_value='["nitros_tensor_list_nchw_rgb_f32"]',
                description='The nitros format of the input tensors'),
            'sam_output_tensor_names': DeclareLaunchArgument(
                'sam_output_tensor_names',
                default_value='["masks","iou","low_res_mask"]',
                description='A list of tensor name to bound to the specified output binding name'),
            'sam_output_binding_names': DeclareLaunchArgument(
                'sam_output_binding_names',
                default_value='["masks","iou_predictions","low_res_masks"]',
                description='A  list of output tensor binding names (specified by model)'),
            'sam_output_tensor_formats': DeclareLaunchArgument(
                'sam_output_tensor_formats',
                default_value='["nitros_tensor_list_nchw_rgb_f32"]',
                description='The nitros format of the output tensors'),
            'color_segmentation_mask_encoding': DeclareLaunchArgument(
                'color_segmentation_mask_encoding',
                default_value='rgb8',
                description='The image encoding of the colored segmentation mask (rgb8 or bgr8)'),
            'input_image_encoding': DeclareLaunchArgument(
                'input_image_encoding',
                default_value='rgb8',
                description='The image encoding of the input image \
                (rgb8 or bgr8 or bgra8 or rgba8)'),
            'prompt_input_type': DeclareLaunchArgument(
                'prompt_input_type',
                default_value='bbox',
                description='Type of the prompt input (bbox or point)'),
            'has_input_mask': DeclareLaunchArgument(
                'has_input_mask',
                default_value='False',
                description='Whether input mask is valid or not.')
        }

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

from typing import Any, Dict

from isaac_ros_examples import IsaacROSLaunchFragment
import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.descriptions import ComposableNode


SAM2_MODEL_INPUT_SIZE = 1024  # SAM model accept 1024x1024 image


class IsaacROSSegmentAnything2LaunchFragment(IsaacROSLaunchFragment):

    @staticmethod
    def get_composable_nodes(interface_specs: Dict[str, Any]) -> Dict[str, ComposableNode]:
        # DNN Image Encoder parameters
        sam2_encoder_image_mean = LaunchConfiguration('sam2_encoder_image_mean')
        sam2_encoder_image_stddev = LaunchConfiguration('sam2_encoder_image_stddev')

        # Triton parameters
        sam2_model_name = LaunchConfiguration('sam2_model_name')
        sam2_model_repository_paths = LaunchConfiguration('sam2_model_repository_paths')
        sam2_max_num_objects = LaunchConfiguration('sam2_max_num_objects')
        sam2_input_tensor_names = LaunchConfiguration('sam2_input_tensor_names')
        sam2_input_binding_names = LaunchConfiguration('sam2_input_binding_names')
        sam2_input_tensor_formats = LaunchConfiguration('sam2_input_tensor_formats')
        sam2_output_tensor_names = LaunchConfiguration('sam2_output_tensor_names')
        sam2_output_binding_names = LaunchConfiguration('sam2_output_binding_names')
        sam2_output_tensor_formats = LaunchConfiguration('sam2_output_tensor_formats')

        orig_img_dims = [interface_specs['camera_resolution']['height'],
                         interface_specs['camera_resolution']['width']]
        img_topic = interface_specs.get('subscribed_topics', {}).get('image', 'image_rect')
        camera_info_topic = interface_specs.get(
            'subscribed_topics', {}).get('camera_info', 'camera_info_rect')
        input_image_encoding = LaunchConfiguration('input_image_encoding')

        return {

            'sam2_resize_node': ComposableNode(
                name='sam2_resize_node',
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::ResizeNode',
                parameters=[{
                    'output_width': SAM2_MODEL_INPUT_SIZE,
                    'output_height': SAM2_MODEL_INPUT_SIZE,
                    'keep_aspect_ratio': True,
                    'disable_padding': True,
                    'encoding_desired': input_image_encoding,
                    'input_width': interface_specs['camera_resolution']['width'],
                    'input_height': interface_specs['camera_resolution']['height']
                }],
                remappings=[('/image', img_topic),
                            ('/camera_info', camera_info_topic),
                            ('resize/image', '/segment_anything2/resized_image')]
            ),

            'sam2_pad_node': ComposableNode(
                name='sam2_pad_node',
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::PadNode',
                parameters=[{
                    'output_image_width': SAM2_MODEL_INPUT_SIZE,
                    'output_image_height': SAM2_MODEL_INPUT_SIZE,
                    'padding_type': 'BOTTOM_RIGHT'
                }],
                remappings=[('/image', '/segment_anything2/resized_image'),
                            ('/padded_image', '/segment_anything2/padded_image')]
            ),

            'sam2_image_format_converter_node': ComposableNode(
                name='sam2_image_format_converter_node',
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
                parameters=[{
                    'image_width': SAM2_MODEL_INPUT_SIZE,
                    'image_height': SAM2_MODEL_INPUT_SIZE,
                    'encoding_desired': 'rgb8',
                }],
                remappings=[
                    ('image_raw', '/segment_anything2/padded_image'),
                    ('image', '/segment_anything2/color_converted_image'),
                ],
            ),

            'sam2_image_to_tensor_node': ComposableNode(
                name='sam2_image_to_tensor_node',
                package='isaac_ros_tensor_proc',
                plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
                parameters=[{
                    'scale': True,
                    'tensor_name': 'image',
                }],
                remappings=[
                    ('image', '/segment_anything2/color_converted_image'),
                    ('tensor', '/segment_anything2/image_tensor')
                ]
            ),

            'sam2_normalize_node': ComposableNode(
                name='sam2_normalize_node',
                package='isaac_ros_tensor_proc',
                plugin='nvidia::isaac_ros::dnn_inference::ImageTensorNormalizeNode',
                parameters=[{
                    'mean': sam2_encoder_image_mean,
                    'stddev': sam2_encoder_image_stddev,
                    'input_tensor_name': 'image',
                    'output_tensor_name': 'image'
                }],
                remappings=[
                    ('tensor', '/segment_anything2/image_tensor'),
                    ('normalized_tensor', '/segment_anything2/normalized_tensor'),
                ],
            ),

            'sam2_interleaved_to_planar_node': ComposableNode(
                name='sam2_interleaved_to_planar_node',
                package='isaac_ros_tensor_proc',
                plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
                parameters=[{
                    'input_tensor_shape': [SAM2_MODEL_INPUT_SIZE, SAM2_MODEL_INPUT_SIZE, 3]
                }],
                remappings=[
                    ('interleaved_tensor', '/segment_anything2/normalized_tensor'),
                    ('planar_tensor', '/segment_anything2/planar_tensor'),
                ]
            ),

            'sam2_reshaper_node': ComposableNode(
                name='sam2_reshaper_node',
                package='isaac_ros_tensor_proc',
                plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
                parameters=[{
                    'output_tensor_name': 'input_tensor',
                    'input_tensor_shape': [3, SAM2_MODEL_INPUT_SIZE, SAM2_MODEL_INPUT_SIZE],
                    'output_tensor_shape': [1, 3, SAM2_MODEL_INPUT_SIZE, SAM2_MODEL_INPUT_SIZE]
                }],
                remappings=[
                    ('tensor', '/segment_anything2/planar_tensor'),
                    ('reshaped_tensor', '/segment_anything2/tensor_pub'),
                ],
            ),

            'sam2_data_preprocessor_node': ComposableNode(
                name='sam2_data_encoder_node',
                package='isaac_ros_segment_anything2',
                plugin='nvidia::isaac_ros::segment_anything2::SegmentAnything2DataEncoderNode',
                parameters=[{
                    'max_num_objects': sam2_max_num_objects,
                    'orig_img_dims': orig_img_dims
                }],
                remappings=[('encoded_data', '/segment_anything2/encoded_data'),
                            ('image', '/segment_anything2/tensor_pub'),
                            ('memory', '/segment_anything2/tensor_sub')]
            ),

            'sam2_triton_node': ComposableNode(
                name='sam2_triton_node',
                package='isaac_ros_triton',
                plugin='nvidia::isaac_ros::dnn_inference::TritonNode',
                parameters=[{
                    'model_name': sam2_model_name,
                    'model_repository_paths': sam2_model_repository_paths,
                    'max_batch_size': 1,
                    'input_tensor_names': sam2_input_tensor_names,
                    'input_binding_names': sam2_input_binding_names,
                    'input_tensor_formats': sam2_input_tensor_formats,
                    'output_tensor_names': sam2_output_tensor_names,
                    'output_binding_names': sam2_output_binding_names,
                    'output_tensor_formats': sam2_output_tensor_formats,
                }],
                remappings=[('tensor_pub', '/segment_anything2/encoded_data'),
                            ('tensor_sub', '/segment_anything2/tensor_sub')]
            ),

            'sam2_decoder_node': ComposableNode(
                name='segment_anything2_decoder_node',
                package='isaac_ros_segment_anything',
                plugin='nvidia::isaac_ros::segment_anything::SegmentAnythingDecoderNode',
                parameters=[{
                    'mask_width': interface_specs['camera_resolution']['width'],
                    'mask_height': interface_specs['camera_resolution']['height'],
                    'max_batch_size': sam2_max_num_objects,
                    'tensor_name': 'high_res_masks'
                }],
                remappings=[('tensor_sub', '/segment_anything2/tensor_sub'),
                            ('/segment_anything/raw_segmentation_mask',
                             '/segment_anything2/raw_segmentation_mask')])

        }

    @staticmethod
    def get_launch_actions(interface_specs: Dict[str, Any]) -> \
            Dict[str, launch.actions.OpaqueFunction]:

        return {
            'sam2_encoder_image_mean': DeclareLaunchArgument(
                'sam2_encoder_image_mean',
                default_value='[0.485, 0.456, 0.406]',
                description='The mean for image normalization'),
            'sam2_encoder_image_stddev': DeclareLaunchArgument(
                'sam2_encoder_image_stddev',
                default_value='[0.229, 0.224, 0.225]',
                description='The standard deviation for image normalization'),
            'sam2_model_name': DeclareLaunchArgument(
                'sam2_model_name',
                default_value='segment_anything2',
                description='The name of the model'),
            'sam2_model_repository_paths': DeclareLaunchArgument(
                'sam2_model_repository_paths',
                default_value='["/tmp/models"]',
                description='The absolute path to the repository of models'),
            'sam2_max_num_objects': DeclareLaunchArgument(
                'sam2_max_num_objects',
                default_value='5',
                description='The maximum allowed batch size of the model'),
            'sam2_input_tensor_names': DeclareLaunchArgument(
                'sam2_input_tensor_names',
                default_value='["image","bbox_coords","point_coords","point_labels", \
            "mask_memory","obj_ptr_memory","original_size","permutation"]',
                description='A list of tensor names to bound to the specified input binding name'),
            'sam2_input_binding_names': DeclareLaunchArgument(
                'sam2_input_binding_names',
                default_value='["image","bbox_coords","point_coords","point_labels", \
            "mask_memory","obj_ptr_memory","original_size","permutation"]',
                description='A list of input tensor binding names (specified by model)'),
            'sam2_input_tensor_formats': DeclareLaunchArgument(
                'sam2_input_tensor_formats',
                default_value='["nitros_tensor_list_nchw_rgb_f32"]',
                description='The nitros format of the input tensors'),
            'sam2_output_tensor_names': DeclareLaunchArgument(
                'sam2_output_tensor_names',
                default_value='["high_res_masks","object_score_logits", \
            "maskmem_features","maskmem_pos_enc","obj_ptr_features"]',
                description='A list of tensor name to bound to the specified output binding name'),
            'sam2_output_binding_names': DeclareLaunchArgument(
                'sam2_output_binding_names',
                default_value='["high_res_masks","object_score_logits", \
            "maskmem_features","maskmem_pos_enc","obj_ptr_features"]',
                description='A list of output tensor binding names (specified by model)'),
            'sam2_output_tensor_formats': DeclareLaunchArgument(
                'sam2_output_tensor_formats',
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
                (rgb8 or bgr8 or bgra8 or rgba8)')
        }

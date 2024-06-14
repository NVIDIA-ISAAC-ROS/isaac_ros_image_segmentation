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

from ament_index_python.packages import get_package_share_directory
import launch
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

YOLO_MODEL_INPUT_SIZE = 640  # YOLO model accept 640x640 image


def generate_launch_description():
    """Launch the DNN Image encoder, Triton node and Segformer decoder node."""
    launch_args = [
        DeclareLaunchArgument(
            'input_image_width',
            default_value='1920',
            description='The input image width'),
        DeclareLaunchArgument(
            'input_image_height',
            default_value='1200',
            description='The input image height'),
        DeclareLaunchArgument(
            'network_image_width',
            default_value='1024',
            description='The input image width that the network expects'),
        DeclareLaunchArgument(
            'network_image_height',
            default_value='1024',
            description='The input image height that the network expects'),
        DeclareLaunchArgument(
            'encoder_image_mean',
            default_value='[0.485, 0.456, 0.406]',
            description='The mean for image normalization'),
        DeclareLaunchArgument(
            'encoder_image_stddev',
            default_value='[0.229, 0.224, 0.225]',
            description='The standard deviation for image normalization'),
        DeclareLaunchArgument(
            'model_name',
            default_value='segment_anything',
            description='The name of the model'),
        DeclareLaunchArgument(
            'model_repository_paths',
            default_value='["/tmp/models"]',
            description='The absolute path to the repository of models'),
        DeclareLaunchArgument(
            'max_batch_size',
            default_value='20',
            description='The maximum allowed batch size of the model'),
        DeclareLaunchArgument(
            'input_tensor_names',
            default_value='["input_tensor","points","labels","input_mask", \
            "has_input_mask","orig_img_dims"]',
            description='A list of tensor names to bound to the specified input binding names'),
        DeclareLaunchArgument(
            'input_binding_names',
            default_value='["images","point_coords","point_labels","mask_input", \
            "has_mask_input","orig_im_size"]',
            description='A list of input tensor binding names (specified by model)'),
        DeclareLaunchArgument(
            'input_tensor_formats',
            default_value='["nitros_tensor_list_nchw_rgb_f32"]',
            description='The nitros format of the input tensors'),
        DeclareLaunchArgument(
            'output_tensor_names',
            default_value='["masks","iou","low_res_mask"]',
            description='A list of tensor names to bound to the specified output binding names'),
        DeclareLaunchArgument(
            'output_binding_names',
            default_value='["masks","iou_predictions","low_res_masks"]',
            description='A  list of output tensor binding names (specified by model)'),
        DeclareLaunchArgument(
            'output_tensor_formats',
            default_value='["nitros_tensor_list_nchw_rgb_f32"]',
            description='The nitros format of the output tensors'),
        DeclareLaunchArgument(
            'network_output_type',
            default_value='argmax',
            description='The output type that the network provides (softmax, sigmoid or argmax)'),
        DeclareLaunchArgument(
            'color_segmentation_mask_encoding',
            default_value='rgb8',
            description='The image encoding of the colored segmentation mask (rgb8 or bgr8)'),
        DeclareLaunchArgument(
            'prompt_input_type',
            default_value='bbox',
            description='Type of the prompt input (bbox or point)'),
        DeclareLaunchArgument(
            'has_input_mask',
            default_value='False',
            description='Whether input mask is valid or not.'),
        DeclareLaunchArgument(
            'orig_img_dims',
            default_value='[640, 640]',
            description='Whether input mask is valid or not.'),
        DeclareLaunchArgument(
            'model_file_path',
            default_value='',
            description='The absolute file path to the ONNX file'),
        DeclareLaunchArgument(
            'engine_file_path',
            default_value='',
            description='The absolute file path to the TensorRT engine file'),
    ]

    # DNN Image Encoder parameters
    input_image_width = LaunchConfiguration('input_image_width')
    input_image_height = LaunchConfiguration('input_image_height')
    network_image_width = LaunchConfiguration('network_image_width')
    network_image_height = LaunchConfiguration('network_image_height')
    encoder_image_mean = LaunchConfiguration('encoder_image_mean')
    encoder_image_stddev = LaunchConfiguration('encoder_image_stddev')

    # Triton parameters
    model_name = LaunchConfiguration('model_name')
    model_repository_paths = LaunchConfiguration('model_repository_paths')
    max_batch_size = LaunchConfiguration('max_batch_size')
    input_tensor_names = LaunchConfiguration('input_tensor_names')
    input_binding_names = LaunchConfiguration('input_binding_names')
    input_tensor_formats = LaunchConfiguration('input_tensor_formats')
    output_tensor_names = LaunchConfiguration('output_tensor_names')
    output_binding_names = LaunchConfiguration('output_binding_names')
    output_tensor_formats = LaunchConfiguration('output_tensor_formats')

    prompt_input_type = LaunchConfiguration('prompt_input_type')
    has_input_mask = LaunchConfiguration('has_input_mask')
    orig_img_dims = LaunchConfiguration('orig_img_dims')

    # YOLO nodes params
    model_file_path = LaunchConfiguration('model_file_path')
    engine_file_path = LaunchConfiguration('engine_file_path')
    confidence_threshold = LaunchConfiguration('confidence_threshold')
    nms_threshold = LaunchConfiguration('nms_threshold')

    # YOLO related nodes
    encoder_dir = get_package_share_directory('isaac_ros_dnn_image_encoder')
    yolov8_encoder_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(encoder_dir, 'launch',
                          'dnn_image_encoder.launch.py')]
        ),
        launch_arguments={
            'input_image_width': input_image_width,
            'input_image_height': input_image_height,
            'network_image_width': str(YOLO_MODEL_INPUT_SIZE),
            'network_image_height': str(YOLO_MODEL_INPUT_SIZE),
            'image_mean': '[0.0, 0.0, 0.0]',
            'image_stddev': '[1.0, 1.0, 1.0]',
            'attach_to_shared_component_container': 'True',
            'component_container_name': 'segment_anything_container',
            'dnn_image_encoder_namespace': 'yolov8_encoder',
            'image_input_topic': '/front_stereo_camera/left/image_rect_color',
            'camera_info_input_topic': '/front_stereo_camera/left/camera_info',
            'tensor_output_topic': '/tensor_pub',
        }.items(),
    )

    yolo_tensor_rt_node = ComposableNode(
        name='tensor_rt',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'model_file_path': model_file_path,
            'engine_file_path': engine_file_path,
            'output_binding_names': ['output0'],
            'output_tensor_names': ['output_tensor'],
            'input_tensor_names': ['input_tensor'],
            'input_binding_names': ['images'],
            'verbose': False,
            'force_engine_update': False
        }]
    )

    yolov8_decoder_node = ComposableNode(
        name='yolov8_decoder_node',
        package='isaac_ros_yolov8',
        plugin='nvidia::isaac_ros::yolov8::YoloV8DecoderNode',
        parameters=[{
            'confidence_threshold': confidence_threshold,
            'nms_threshold': nms_threshold,
        }]
    )
    # End of YOLO releated nodes

    # Consider YOLO's 640x640 as input image because predicted bboxes correspond
    # to the resized image.
    resize_node = ComposableNode(
        name='sam_resize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'output_width': network_image_width,
            'output_height': network_image_height,
            'keep_aspect_ratio': True,
            'disable_padding': True,
            'input_width': YOLO_MODEL_INPUT_SIZE,
            'input_height': YOLO_MODEL_INPUT_SIZE
        }],
        remappings=[('/image', '/yolov8_encoder/resize/image'),
                    ('/camera_info', '/yolov8_encoder/resize/camera_info'),
                    ('resize/image', '/segment_anything/resized_image')]
    )

    pad_node = ComposableNode(
        name='sam_pad_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::PadNode',
        parameters=[{
            'output_image_width': network_image_width,
            'output_image_height': network_image_height,
            'padding_type': 'BOTTOM_RIGHT'
        }],
        remappings=[('/image', '/segment_anything/resized_image'),
                    ('/padded_image', '/segment_anything/padded_image')]
    )

    image_format_converter_node = ComposableNode(
        name='image_format_converter_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        parameters=[{
            'image_width': network_image_width,
            'image_height': network_image_height,
            'encoding_desired': 'rgb8',
        }],
        remappings=[
            ('image_raw', '/segment_anything/padded_image'),
            ('image', '/segment_anything/color_converted_image'),
        ],
    )

    image_to_tensor_node = ComposableNode(
        name='image_to_tensor_node',
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
    )

    normalize_node = ComposableNode(
        name='sam_normalize_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageTensorNormalizeNode',
        parameters=[{
            'mean': encoder_image_mean,
            'stddev': encoder_image_stddev,
            'input_tensor_name': 'image',
            'output_tensor_name': 'image'
        }],
        remappings=[
            ('tensor', '/segment_anything/image_tensor'),
            ('normalized_tensor', '/segment_anything/normalized_tensor'),
        ],
    )

    interleaved_to_planar_node = ComposableNode(
        name='sam_interleaved_to_planar_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        parameters=[{
            'input_tensor_shape': [network_image_height, network_image_width, 3]
        }],
        remappings=[
            ('interleaved_tensor', '/segment_anything/normalized_tensor'),
            ('planar_tensor', '/segment_anything/planar_tensor'),
        ]
    )

    reshaper_node = ComposableNode(
        name='reshape_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        parameters=[{
            'output_tensor_name': 'input_tensor',
            'input_tensor_shape': [3, network_image_height, network_image_width],
            'output_tensor_shape': [1, 3, network_image_height, network_image_width]
        }],
        remappings=[
            ('tensor', '/segment_anything/planar_tensor'),
            ('reshaped_tensor', '/segment_anything/tensor_pub'),
        ],
    )

    dummy_mask_pub_node = ComposableNode(
        name='dummy_mask_pub',
        package='isaac_ros_segment_anything',
        plugin='nvidia::isaac_ros::segment_anything::DummyMaskPublisher',
        remappings=[('tensor_pub', '/segment_anything/tensor_pub')]
    )

    data_preprocessor_node = ComposableNode(
        name='sam_data_encoder_node',
        package='isaac_ros_segment_anything',
        plugin='nvidia::isaac_ros::segment_anything::SegmentAnythingDataEncoderNode',
        parameters=[{
            'prompt_input_type': prompt_input_type,
            'has_input_mask': has_input_mask,
            'max_batch_size': max_batch_size,
            'orig_img_dims': orig_img_dims
        }],
        remappings=[('prompts', '/detections_output'),
                    ('tensor_pub', '/segment_anything/tensor_pub'),
                    ('tensor', '/segment_anything/encoded_data')]
    )

    sam_triton_node = ComposableNode(
        name='sam_triton_node',
        package='isaac_ros_triton',
        plugin='nvidia::isaac_ros::dnn_inference::TritonNode',
        parameters=[{
            'model_name': model_name,
            'model_repository_paths': model_repository_paths,
            'max_batch_size': 1,
            'input_tensor_names': input_tensor_names,
            'input_binding_names': input_binding_names,
            'input_tensor_formats': input_tensor_formats,
            'output_tensor_names': output_tensor_names,
            'output_binding_names': output_binding_names,
            'output_tensor_formats': output_tensor_formats,
        }],
        remappings=[('tensor_pub', '/segment_anything/encoded_data'),
                    ('tensor_sub', '/segment_anything/tensor_sub')]
    )

    sam_decoder_node = ComposableNode(
        name='semgnet_anything_decoder_node',
        package='isaac_ros_segment_anything',
        plugin='nvidia::isaac_ros::segment_anything::SegmentAnythingDecoderNode',
        parameters=[{
            'mask_width': YOLO_MODEL_INPUT_SIZE,
            'mask_height': YOLO_MODEL_INPUT_SIZE,
            'max_batch_size': max_batch_size
        }],
        remappings=[('tensor_sub', '/segment_anything/tensor_sub')])

    container = ComposableNodeContainer(
        name='segment_anything_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[yolo_tensor_rt_node, yolov8_decoder_node,
                                      resize_node, pad_node, image_format_converter_node,
                                      image_to_tensor_node, normalize_node,
                                      interleaved_to_planar_node, reshaper_node,
                                      data_preprocessor_node,
                                      sam_triton_node, sam_decoder_node, dummy_mask_pub_node],
        output='screen', arguments=['--ros-args', '--log-level', 'WARN'])

    final_launch_description = launch_args + [container, yolov8_encoder_launch]
    return launch.LaunchDescription(final_launch_description)

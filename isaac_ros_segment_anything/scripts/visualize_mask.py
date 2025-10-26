#!/usr/bin/env python3

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import cv2
import cv_bridge
from isaac_ros_tensor_list_interfaces.msg import TensorList
from message_filters import Subscriber, TimeSynchronizer
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class SegmentAnythingVisualization(Node):

    def __init__(self):
        super().__init__('segment_anything_visualizer')

        self._bridge = cv_bridge.CvBridge()
        self._mask_subscriber = Subscriber(
            self, TensorList, '/segment_anything/raw_segmentation_mask')

        self._image_subscriber = Subscriber(
            self, Image, '/yolov8_encoder/resize/image')

        self._processed_image_pub = self.create_publisher(
            Image, '/segment_anything/colored_segmentation_mask', 10)

        # Pre-convert color palette from hex to RGB for better performance
        hex_colors = ['800000', '556B2F', '008080', '000080', '9ACD32', 'FF0000',
                      'FF8C00', 'FFD700', '00FF00', 'BA55D3', '00FA9A', '00FFFF',
                      '0000FF', 'F08080', 'FF00FF', '1E90FF', 'DDA0DD', 'FF1493',
                      '87CEFA', 'FFDEAD']

        self._color_palette = np.array([
            [int(color[i:i+2], 16) for i in (0, 2, 4)]
            for color in hex_colors
        ], dtype=np.uint8)

        self.sync = TimeSynchronizer([self._image_subscriber, self._mask_subscriber], 50)
        self.sync.registerCallback(self.callback)

    def callback(self, img, masks):
        input_image = self._bridge.imgmsg_to_cv2(img, 'rgb8')
        tensor = masks.tensors[0]

        # Extract dimensions directly from shape
        dimensions = tensor.shape.dims.tolist()
        num_masks = dimensions[0]

        # Reshape data more efficiently
        data = np.array(tensor.data.tolist(), dtype=np.uint8).reshape(dimensions)

        # Start with the original image instead of an empty array
        result_image = input_image.copy()
        # Apply mask colors directly to the result image
        for n in range(num_masks):
            pallet_idx = min(n, len(self._color_palette) - 1)
            mask = data[n, 0, :, :] > 0
            if np.any(mask):  # Only process if mask has positive values
                # For pixels where mask exists, blend with 50% transparency
                result_image[mask] = cv2.addWeighted(
                    input_image[mask], 0.4,
                    np.full((np.count_nonzero(mask), 3),
                            self._color_palette[pallet_idx], dtype=np.uint8),
                    0.6, 0
                )

        # Publish the processed image
        processed_img = self._bridge.cv2_to_imgmsg(result_image, encoding='rgb8')
        processed_img.header = img.header  # Maintain timestamp and frame_id
        self._processed_image_pub.publish(processed_img)


def main(args=None):
    rclpy.init(args=args)
    node = SegmentAnythingVisualization()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

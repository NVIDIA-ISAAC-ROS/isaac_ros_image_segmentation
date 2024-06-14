#!/usr/bin/env python3

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

import cv_bridge
from isaac_ros_tensor_list_interfaces.msg import TensorList
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class SegmentAnythingVisualization(Node):

    def __init__(self):
        super().__init__('segment_anything_visualizer')
        self._bridge = cv_bridge.CvBridge()
        self._mask_subscriber = self.create_subscription(
            TensorList, 'segment_anything/raw_segmentation_mask', self.callback, 10)

        self._processed_image_pub = self.create_publisher(
            Image, 'segment_anything/colored_segmentation_mask', 10)

        self._color_palette = ['556B2F', '800000', '008080', '000080', '9ACD32', 'FF0000',
                               'FF8C00', 'FFD700', '00FF00', 'BA55D3', '00FA9A', '00FFFF',
                               '0000FF', 'F08080', 'FF00FF', '1E90FF', 'DDA0DD', 'FF1493',
                               '87CEFA', 'FFDEAD']

    def callback(self, masks):
        tensor = masks.tensors[0]
        shape = tensor.shape

        dimensions = shape.dims.tolist()

        data = np.array(tensor.data.tolist())
        data = data.reshape(dimensions[:])

        rgb_image = np.zeros((dimensions[2], dimensions[3], 3), dtype=np.uint8)

        for n in range(dimensions[0]):
            palette_idx = n

            if(n >= len(self._color_palette)):
                palette_idx = len(self._color_palette) - 1

            rgb_val = [int(self._color_palette[palette_idx][i:i+2], 16) for i in (0, 2, 4)]
            rgb_image[:, :, :][np.where(data[n, 0, :, :] > 0)] = np.array(rgb_val)

        processed_img = self._bridge.cv2_to_imgmsg(
            rgb_image, encoding='rgb8')
        self._processed_image_pub.publish(processed_img)


def main(args=None):
    rclpy.init(args=args)
    node = SegmentAnythingVisualization()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

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

import cv2
import cv_bridge
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class ColoredMaskConverterNode(Node):

    def __init__(self):
        super().__init__('colored_mask_converter_node')

        self.declare_parameter('color', [255, 0, 0])

        self.bridge = cv_bridge.CvBridge()
        self.mask = None

        self.publisher = self.create_publisher(Image, 'colored_segmentation_mask', 10)

        self.create_subscription(
            Image, 'image', self.image_callback,
            QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10))

        self.create_subscription(Image, 'binary_segmentation_mask', self.mask_callback, 10)

    def mask_callback(self, msg):
        self.mask = msg

    def image_callback(self, msg):
        if self.mask is not None:
            src = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
            binary_mask = self.bridge.imgmsg_to_cv2(self.mask, 'mono8')
            colored_mask = np.zeros(src.shape, dtype=np.uint8)
            color = self.get_parameter('color').get_parameter_value().integer_array_value
            colored_mask[np.where(binary_mask[:, :] > 0)] = color
            dst = cv2.addWeighted(src, 1.0, colored_mask, 1.0, 0)
            image = self.bridge.cv2_to_imgmsg(dst, 'rgb8')
            self.publisher.publish(image)
        else:
            self.publisher.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(ColoredMaskConverterNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()

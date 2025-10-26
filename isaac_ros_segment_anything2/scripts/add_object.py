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

import argparse
import rclpy
from rclpy.node import Node
from vision_msgs.msg import BoundingBox2D, Point2D
from isaac_ros_segment_anything2_interfaces.srv import AddObjects
from std_msgs.msg import Header
import sys


class SingleObjectAdder(Node):
    def __init__(self):
        super().__init__('single_object_adder')

        # Create service client for AddObjects
        self.add_objects_client = self.create_client(AddObjects, 'add_objects')

        # Wait for service to be available
        timeout_count = 0
        while not self.add_objects_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for add_objects service...')
            timeout_count += 1
            if timeout_count > 10:
                self.get_logger().error('Service not available after 10 seconds. Exiting.')
                return

        self.get_logger().info('Connected to add_objects service!')

    def add_bbox_object(self, object_id, x_center, y_center, width, height,
                        frame_id='camera_frame'):
        """Add a single bounding box object"""

        # Create bounding box
        bbox = BoundingBox2D()
        bbox.center.position.x = float(x_center)
        bbox.center.position.y = float(y_center)
        bbox.size_x = float(width)
        bbox.size_y = float(height)

        # Create service request
        request = AddObjects.Request()
        request.request_header = Header()
        request.request_header.frame_id = frame_id

        request.bbox_object_ids = [object_id]
        request.bbox_coords = [bbox]
        request.point_object_ids = []
        request.point_coords = []
        request.point_labels = []

        return self._call_service(request)

    def add_point_object(self, object_id, x, y, point_label=1,
                         frame_id='camera_frame'):
        """Add a single point object"""

        # Create point
        point = Point2D()
        point.x = float(x)
        point.y = float(y)

        # Create service request
        request = AddObjects.Request()
        request.request_header = Header()
        request.request_header.frame_id = frame_id

        request.bbox_object_ids = []
        request.bbox_coords = []
        request.point_object_ids = [object_id]
        request.point_coords = [point]
        request.point_labels = [int(point_label)]

        return self._call_service(request)

    def _call_service(self, request):
        """Call the AddObjects service"""
        try:
            self.get_logger().info('Calling add_objects service...')
            future = self.add_objects_client.call_async(request)

            # Wait for response with 5-second timeout
            self.get_logger().info('Waiting for service response (5 second timeout)...')
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

            if future.done():
                self.get_logger().info('Service call completed')
                try:
                    response = future.result()

                    if response.success:
                        self.get_logger().info(f'Service call succeeded: {response.message}')
                        self.get_logger().info(f'Object IDs: {response.object_ids}')
                        self.get_logger().info(f'Object Indices: {response.object_indices}')
                        return True
                    else:
                        self.get_logger().error(f'Service call failed: {response.message}')
                        if (len(response.not_added_object_ids) > 0):
                            self.get_logger().error(
                                f'Not added object ids: {response.not_added_object_ids}'
                            )
                        return False
                except Exception as result_error:
                    self.get_logger().error(f'Error getting service result: {str(result_error)}')
                    return False
            else:
                # Timeout occurred
                self.get_logger().error(
                    'Service call timed out after 5 seconds - no response received'
                )
                # Cancel the future to clean up
                future.cancel()
                return False

        except Exception as e:
            self.get_logger().error(f'Service call error: {str(e)}')
            return False


def main():
    parser = argparse.ArgumentParser(description='Utility to add a single object.')

    # Common arguments
    parser.add_argument('--object-id', '-i', required=True, type=str,
                        help='Object ID/name (e.g., "bottle", "person1", "box")')
    parser.add_argument('--frame-id', '-f', default='camera_frame', type=str,
                        help='Frame ID for the object (default: camera_frame)')

    # Create subparsers for bbox and point
    subparsers = parser.add_subparsers(dest='type', help='Object type')

    # Bounding box subparser
    bbox_parser = subparsers.add_parser('bbox', help='Add bounding box object')
    bbox_parser.add_argument('--x-center', '-x', required=True, type=float,
                             help='X coordinate of bounding box center')
    bbox_parser.add_argument('--y-center', '-y', required=True, type=float,
                             help='Y coordinate of bounding box center')
    bbox_parser.add_argument('--width', '-w', required=True, type=float,
                             help='Width of bounding box')
    bbox_parser.add_argument('--height', '-H', required=True, type=float,
                             help='Height of bounding box')

    # Point subparser
    point_parser = subparsers.add_parser('point', help='Add point object')
    point_parser.add_argument('--x', '-x', required=True, type=float,
                              help='X coordinate of point')
    point_parser.add_argument('--y', '-y', required=True, type=float,
                              help='Y coordinate of point')
    point_parser.add_argument('--label', '-l', default=1, type=int,
                              help='Point label (1=foreground, 0=background, default: 1)')

    # Parse arguments
    args = parser.parse_args()

    if args.type is None:
        parser.print_help()
        print('\nError: Must specify either bbox or point subcommand')
        return 1

    # Initialize ROS2
    rclpy.init()

    try:
        # Create node
        adder = SingleObjectAdder()

        success = False
        if args.type == 'bbox':
            print(f'Adding bounding box object {args.object_id} at '
                  f'({args.x_center}, {args.y_center}) with size '
                  f'{args.width}x{args.height}')
            success = adder.add_bbox_object(
                args.object_id,
                args.x_center,
                args.y_center,
                args.width,
                args.height,
                args.frame_id
            )
        elif args.type == 'point':
            print(f'Adding point object {args.object_id} at ({args.x}, {args.y}) '
                  f'with label {args.label}')
            success = adder.add_point_object(
                args.object_id,
                args.x,
                args.y,
                args.label,
                args.frame_id
            )

        if success:
            print('Object added successfully!')
            return 0
        else:
            print('Failed to add object (check logs for details - may be timeout '
                  'or service error)')
            return 1

    except KeyboardInterrupt:
        print('\nInterrupted by user')
        return 1
    except Exception as e:
        print(f'Error: {e}')
        return 1
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())

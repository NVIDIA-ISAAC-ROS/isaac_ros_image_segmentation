# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.22.1)
project(isaac_ros_peoplesemseg_models_install)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Download and install models to the asset folder (outside the build)
install_isaac_ros_asset(install_peoplesemsegnet_vanilla)
install_isaac_ros_asset(install_peoplesemsegnet_shuffleseg)

# Install the installation scripts such that the models can be installed when not building from source
install(PROGRAMS asset_scripts/install_peoplesemsegnet_shuffleseg.sh DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS asset_scripts/install_peoplesemsegnet_vanilla.sh DESTINATION lib/${PROJECT_NAME})

ament_auto_package(INSTALL_TO_SHARE)

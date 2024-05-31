#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Download and tao-convert ESS models.
# * Models will be stored in the isaac_ros_assets dir
# * The script must be called with the --eula argument prior to downloading.

set -e

ASSET_NAME="deployable_quantized_vanilla_unet_v2.0"
EULA_URL="https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplesemsegnet"
ASSET_DIR="${ISAAC_ROS_WS}/isaac_ros_assets/models/peoplesemsegnet/${ASSET_NAME}"
ASSET_INSTALL_PATHS="${ASSET_DIR}/1/model.plan"
MODEL_URL="https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesemsegnet/versions/deployable_quantized_vanilla_unet_v2.0/files/peoplesemsegnet_vanilla_unet_dynamic_etlt_int8_fp16.etlt"

source "isaac_ros_asset_eula.sh"

mkdir -p $(dirname "$ASSET_INSTALL_PATHS")

wget "${MODEL_URL}" -O "${ASSET_DIR}/model.etlt"

echo "Converting PeopleSemSegnet etlt file to plan file."
/opt/nvidia/tao/tao-converter \
    -k tlt_encode \
    -d 3,544,960 \
    -p input_1:0,1x3x544x960,1x3x544x960,1x3x544x960 \
    -t fp16 \
    -e "${ASSET_INSTALL_PATHS}" \
    -o argmax_1 \
    "${ASSET_DIR}/model.etlt"

# Create config.pbtxt
config_file_text=$(
  cat <<EOF
name: "$(basename "$model_directory")"
platform: "tensorrt_plan"
max_batch_size: 0
input [
  {
    name: "input_1:0"
    data_type: TYPE_FP32
    dims: [ 1, 3, 544, 960 ]
  }
]
output [
  {
    name: "argmax_1"
    data_type: TYPE_INT32
    dims: [ 1, 544, 960, 1 ]
  }
]
version_policy: {
  specific {
    versions: [ 1 ]
  }
}
EOF
)
echo "$config_file_text" >${ASSET_DIR}/config.pbtxt


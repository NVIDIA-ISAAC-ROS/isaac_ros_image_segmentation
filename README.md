# Isaac ROS Image Segmentation

<div align="center"><img alt="Isaac ROS Image Segmentation Sample Output" src="resources/isaac_ros_people_segmentation.gif" width="400px"/></div>

## Overview
This repository provides NVIDIA GPU-accelerated packages for semantic image segmentation. Using a deep learned [U-Net](https://en.wikipedia.org/wiki/U-Net) model, such as [`PeopleSemSegnet`](https://ngc.nvidia.com/catalog/models/nvidia:tao:peoplesemsegnet), and a monocular camera, the `isaac_ros_unet` package can generate an image mask segmenting out objects of interest.

Packages in this repository rely on accelerated DNN model inference using [Triton](https://github.com/triton-inference-server/server) or [TensorRT](https://developer.nvidia.com/tensorrt) from [Isaac ROS DNN Inference](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/tree/main/isaac_ros_dnn_inference) and a pretrained model from  [NVIDIA GPU Cloud (NGC)](https://docs.nvidia.com/ngc/) or elsewhere.

### Isaac ROS NITROS Acceleration
This package is powered by [NVIDIA Isaac Transport for ROS (NITROS)](https://developer.nvidia.com/blog/improve-perception-performance-for-ros-2-applications-with-nvidia-isaac-transport-for-ros/), which leverages type adaptation and negotiation to optimize message formats and dramatically accelerate communication between participating nodes. 

## Performance
The following are the benchmark performance results of the prepared pipelines in this package, by supported platform:

| Pipeline               | AGX Orin | AGX Xavier | x86_64 w/ RTX 3060 Ti |
| ---------------------- | -------- | ---------- | --------------------- |
| PeopleSemSegNet (544p) | 325 fps  | 208 fps    | 452 fps               |

> **Note**: The model performance without the encoder and decoder is reported

## Table of Contents
- [Isaac ROS Image Segmentation](#isaac-ros-image-segmentation)
  - [Overview](#overview)
    - [Isaac ROS NITROS Acceleration](#isaac-ros-nitros-acceleration)
  - [Performance](#performance)
  - [Table of Contents](#table-of-contents)
  - [Latest Update](#latest-update)
  - [Supported Platforms](#supported-platforms)
    - [Docker](#docker)
  - [Quickstart](#quickstart)
  - [Next Steps](#next-steps)
    - [Try NITROS-Accelerated Graph with Argus](#try-nitros-accelerated-graph-with-argus)
    - [Use Different Models](#use-different-models)
    - [Customize your Dev Environment](#customize-your-dev-environment)
  - [Package Reference](#package-reference)
    - [UNetDecoderNode](#unetdecodernode)
      - [Usage](#usage)
      - [ROS Parameters](#ros-parameters)
      - [ROS Topics Subscribed](#ros-topics-subscribed)
      - [ROS Topics Published](#ros-topics-published)
  - [Troubleshooting](#troubleshooting)
    - [Isaac ROS Troubleshooting](#isaac-ros-troubleshooting)
    - [Deep Learning Troubleshooting](#deep-learning-troubleshooting)
  - [Updates](#updates)

## Latest Update
Update 2022-06-30: Removed frame_id, queue_size and tensor_output_order parameter. Added network_output_type parameter (support for sigmoid and argmax output layers). Switched implementation to use NITROS. Removed support for odd sized images. Switched tutorial to use PeopleSemSegNet ShuffleSeg and moved som details to other READMEs.

## Supported Platforms
This package is designed and tested to be compatible with ROS2 Humble running on [Jetson](https://developer.nvidia.com/embedded-computing) or an x86_64 system with an NVIDIA GPU.


| Platform | Hardware                                                                                                                                                                                                | Software                                                                                                             | Notes                                                                                                                                                                                   |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Jetson   | [Jetson Orin](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)<br/>[Jetson Xavier](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-agx-xavier/) | [JetPack 5.0.1 DP](https://developer.nvidia.com/embedded/jetpack)                                                    | For best performance, ensure that [power settings](https://docs.nvidia.com/jetson/archives/r34.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance.html) are configured appropriately. |
| x86_64   | NVIDIA GPU                                                                                                                                                                                              | [Ubuntu 20.04+](https://releases.ubuntu.com/20.04/) <br> [CUDA 11.6.1+](https://developer.nvidia.com/cuda-downloads) |


### Docker
To simplify development, we strongly recommend leveraging the Isaac ROS Dev Docker images by following [these steps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/dev-env-setup.md). This will streamline your development environment setup with the correct versions of dependencies on both Jetson and x86_64 platforms.

> **Note:** All Isaac ROS Quickstarts, tutorials, and examples have been designed with the Isaac ROS Docker images as a prerequisite.

## Quickstart
1. Set up your development environment by following the instructions [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/dev-env-setup.md).
2. Clone this repository and its dependencies under `~/workspaces/isaac_ros-dev/src`.

    ```bash
    cd ~/workspaces/isaac_ros-dev/src
    ```

    ```bash
    git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common
    ```

    ```bash
    git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros
    ```

    ```bash
    git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_segmentation
    ```

    ```bash
    git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference
    ```

3. Pull down a ROS Bag of sample data:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_image_segmentation && \
      git lfs pull -X "" -I "resources/rosbags/"
    ```

4. Launch the Docker container using the `run_dev.sh` script:
    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```
5. Download the `PeopleSemSegNet ShuffleSeg` ETLT file and the `int8` inference mode cache file:
    ```bash
    mkdir -p /tmp/models/peoplesemsegnet_shuffleseg/1 && \
      cd /tmp/models/peoplesemsegnet_shuffleseg && \
      wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesemsegnet/versions/deployable_shuffleseg_unet_v1.0/files/peoplesemsegnet_shuffleseg_etlt.etlt && \
      wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesemsegnet/versions/deployable_shuffleseg_unet_v1.0/files/peoplesemsegnet_shuffleseg_cache.txt
    ```
6. Convert the ETLT file to a TensorRT plan file:
    ```bash
    /opt/nvidia/tao/tao-converter -k tlt_encode -d 3,544,960 -p input_2:0,1x3x544x960,1x3x544x960,1x3x544x960 -t int8 -c peoplesemsegnet_shuffleseg_cache.txt -e /tmp/models/peoplesemsegnet_shuffleseg/1/model.plan -o argmax_1 peoplesemsegnet_shuffleseg_etlt.etlt
    ```
7. Create a file called `/tmp/models/peoplesemsegnet_shuffleseg/config.pbtxt` by copying the sample Triton config file:
    ```bash
    cp /workspaces/isaac_ros-dev/src/isaac_ros_image_segmentation/resources/peoplesemsegnet_shuffleseg_config.pbtxt /tmp/models/peoplesemsegnet_shuffleseg/config.pbtxt
    ```
8.  Inside the container, build and source the workspace:
    ```bash
    cd /workspaces/isaac_ros-dev && \
      colcon build --symlink-install && \
      source install/setup.bash
    ```
9.  (Optional) Run tests to verify complete and correct installation:
    ```bash
    colcon test --executor sequential
    ```
10. Run the following launch files to spin up a demo of this package:
    ```bash
    ros2 launch isaac_ros_unet isaac_ros_unet_triton.launch.py model_name:=peoplesemsegnet_shuffleseg model_repository_paths:=['/tmp/models'] input_binding_names:=['input_2:0'] output_binding_names:=['argmax_1'] network_output_type:='argmax'
    ```
    Then open **another** terminal, and enter the Docker container again:
    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```
    Then, play the ROS bag:
    ```bash
    ros2 bag play -l src/isaac_ros_image_segmentation/resources/rosbags/unet_sample_data/
    ```
11. Visualize and validate the output of the package by launching `rqt_image_view` in another terminal:
    In a **third** terminal, enter the Docker container again:
    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```
    Then launch `rqt_image_view`:
    ```bash
    ros2 run rqt_image_view rqt_image_view
    ```
    Then inside the `rqt_image_view` GUI, change the topic to `/unet/colored_segmentation_mask` to view a colorized segmentation mask. 

    <div align="center"><img alt="Coloured Segmentation Mask" src="resources/peoplesemsegnet_shuffleseg_rqt.png" width="350" title="U-Net Shuffleseg result in rqt_image_view"/></div>

    > **Note:** The raw segmentation is also published to `/unet/raw_segmentation_mask`. However, the raw pixels correspond to the class labels and so the output is unsuitable for human visual inspection.
## Next Steps

### Try NITROS-Accelerated Graph with Argus
If you have an Argus-compatible camera, you can launch the NITROS-accelerated graph by following the [tutorial](docs/tutorial-nitros-graph.md).

### Use Different Models

Click [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference/blob/main/docs/model-preparation.md) for more information about how to use NGC models.

| Model Name                                                                                                                                              | Use Case                                              |
| ------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| [PeopleSemSegNet ShuffleSeg](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplesemsegnet/files?version=deployable_shuffleseg_unet_v1.0) | Semantically segment people inference at a high speed |
| [PeopleSemSegNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplesemsegnet/files?version=deployable_vanilla_unet_v2.0)               | Semantically segment people                           |

> **Note**: the parameters may need to be changed depending on the model used. Please verify the preprocessing steps and postprocessing steps of the model against the configuration of the launch file.

### Customize your Dev Environment
To customize your development environment, reference [this guide](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/modify-dockerfile.md).

## Package Reference
### UNetDecoderNode
#### Usage

Triton:
```bash
ros2 launch isaac_ros_unet isaac_ros_unet_triton.launch.py network_image_width:=<network_image_width> network_image_height:=<network_image_height> encoder_image_mean:=<encoder_image_mean> encoder_image_stddev:=<encoder_image_stddev> model_name:=<model_name> model_repository_paths:=<model_repository_paths> max_batch_size:=<max_batch_size> input_tensor_names:=<input_tensor_names> input_binding_names:=<input_binding_names> input_tensor_formats:=<input_tensor_formats> output_tensor_names:=<output_tensor_names> output_binding_names:=<output_binding_names> output_tensor_formats:=<output_tensor_formats> network_output_type:=<network_output_type> color_segmentation_mask_encoding:=<color_segmentation_mask_encoding> mask_width:=<mask_width> mask_height:=<mask_height>
```

TensorRT:
```bash
ros2 launch isaac_ros_unet isaac_ros_unet_tensor_rt.launch.py network_image_width:=<network_image_width> network_image_height:=<network_image_height> encoder_image_mean:=<encoder_image_mean> encoder_image_stddev:=<encoder_image_stddev> model_file_path:=<model_file_path> engine_file_path:=<engine_file_path> input_tensor_names:=<input_tensor_names> input_binding_names:=<input_binding_names> input_tensor_formats:=<input_tensor_formats> output_tensor_names:=<output_tensor_names> output_binding_names:=<output_binding_names> output_tensor_formats:=<output_tensor_formats> tensorrt_verbose:=<tensorrt_verbose> force_engine_update:=<force_engine_update> network_output_type:=<network_output_type> color_segmentation_mask_encoding:=<color_segmentation_mask_encoding> mask_width:=<mask_width> mask_height:=<mask_height>
```

#### ROS Parameters

| ROS Parameter                      | Type           | Default   | Description                                                                                                                                                                                             |
| ---------------------------------- | -------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `color_segmentation_mask_encoding` | `string`       | `rgb8`    | The image encoding of the colored segmentation mask. <br/> Supported values: `rgb8`, `bgr8`                                                                                                             |
| `color_palette `                   | `int64_t list` | `[]`      | Vector of integers where each element represents the rgb color hex code for the corresponding class label. The number of elements should equal the number of classes. <br/> E.g. `[0xFF0000, 0x76b900]` |
| `network_output_type`              | `string`       | `softmax` | The type of output that the network provides. <br/> Supported values: `softmax`, `argmax`, `sigmoid`                                                                                                    |
| `mask_width`                       | `int16_t`      | `960`     | The width of the segmentation mask.                                                                                                                                                                     |
| `mask_height`                      | `int16_t`      | `544`     | The height of the segmentation mask.                                                                                                                                                                    |

> **Warning**: the following parameters are no longer supported:
> - `queue_size`
> - `frame_id` as the `frame_id` of the header will be forwarded now
> - `tensor_output_order` as the order will be inferred from the model. Note: the model output should be `NCHW` or `NHWC`. In this context, the `C` refers to the class.

> **Note:** For the `network_output_type` parameter's `softmax` and `sigmoid `option, we currently expect only 32 bit floating point values. For the `argmax` option, we currently expect only signed 32 bit integers.

> **Note:** Models with greater than 255 classes are currently not supported. If a class label greater than 255 is detected, this mask will be downcast to 255 in the raw segmentation.

#### ROS Topics Subscribed
| ROS Topic    | Interface                                                                                                                                                         | Description                                                               |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `tensor_sub` | [isaac_ros_tensor_list_interfaces/TensorList](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/isaac_ros_tensor_list_interfaces/msg/TensorList.msg) | The tensor that contains raw probabilities for every class in each pixel. |

> **Limitation:** All input images are required to have height and width that are both an even number of pixels.

#### ROS Topics Published
| ROS Topic                         | Interface                                                                                            | Description                                                                                          |
| --------------------------------- | ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `unet/raw_segmentation_mask `     | [sensor_msgs/Image](https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/Image.msg) | The raw segmentation mask, encoded in mono8. Each pixel represents a class label.                    |
| `/unet/colored_segmentation_mask` | [sensor_msgs/Image](https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/Image.msg) | The colored segmentation mask. The color palette is user specified by the `color_palette` parameter. |

## Troubleshooting
### Isaac ROS Troubleshooting
For solutions to problems with Isaac ROS, please check [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/troubleshooting.md).

### Deep Learning Troubleshooting

For solutions to problems with using DNN models, please check [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference/blob/main/docs/troubleshooting.md).


## Updates
| Date       | Changes                                                                                                                                                                                                                                                                                                                              |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 2022-06-30 | Removed frame_id, queue_size and tensor_output_order parameter. Added network_output_type parameter (support for sigmoid and argmax output layers). Switched implementation to use NITROS. Removed support for odd sized images. Switched tutorial to use PeopleSemSegNet ShuffleSeg and moved unnecessary details to other READMEs. |
| 2021-11-15 | Isaac Sim HIL documentation update                                                                                                                                                                                                                                                                                                   |
| 2021-10-20 | Initial release                                                                                                                                                                                                                                                                                                                      |

# Isaac ROS Image Segmentation

<div align="center"><img src="resources/peoplesemsegnet_rviz2.png" width="400px"/></div>

## Overview
This repository provides NVIDIA GPU-accelerated packages for semantic image segmentation. Using a deep learned [U-Net](https://en.wikipedia.org/wiki/U-Net) model, such as [`PeopleSemSegnet`](https://ngc.nvidia.com/catalog/models/nvidia:tao:peoplesemsegnet), and a monocular camera, the `isaac_ros_unet` package can generate an image mask segmenting out objects of interest.

Packages in this repository rely on accelerated DNN model inference using [Triton](https://github.com/triton-inference-server/server) or [TensorRT](https://developer.nvidia.com/tensorrt) from [Isaac ROS DNN Inference](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/tree/main/isaac_ros_dnn_inference) and a pretrained model from  [NVIDIA GPU Cloud (NGC)](https://docs.nvidia.com/ngc/) or elsewhere.

## System Requirements
This Isaac ROS package is designed and tested to be compatible with ROS2 Foxy on Jetson hardware, in addition to on x86 systems with an Nvidia GPU. On x86 systems, packages are only supported when run in the provided Isaac ROS Dev Docker container.

### Jetson
- AGX Xavier or Xavier NX
- JetPack 4.6

### x86_64 (in Isaac ROS Dev Docker Container)
- CUDA 11.1+ supported discrete GPU
- VPI 1.1.11
- Ubuntu 20.04+

**Note:** For best performance on Jetson, ensure that power settings are configured appropriately ([Power Management for Jetson](https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/power_management_jetson_xavier.html#wwpID0EUHA)).

### Docker
You need to use the Isaac ROS development Docker image from [Isaac ROS Common](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common), based on the version 21.08 image from [Deep Learning Frameworks Containers](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

You must first install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to make use of the Docker container development/runtime environment.

Configure `nvidia-container-runtime` as the default runtime for Docker by editing `/etc/docker/daemon.json` to include the following:
```
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
```
and then restarting Docker: `sudo systemctl daemon-reload && sudo systemctl restart docker`

Run the following script in `isaac_ros_common` to build the image and launch the container on x86_64 or Jetson:

`$ scripts/run_dev.sh <optional_path>`

### Dependencies
- [isaac_ros_common](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/tree/main/isaac_ros_common)
- [isaac_ros_nvengine](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/tree/main/isaac_ros_nvengine)
- [isaac_ros_nvengine_interfaces](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/tree/main/isaac_ros_nvengine_interfaces)
- [isaac_ros_tensor_rt](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference/tree/main/isaac_ros_tensor_rt)
- [isaac_ros_triton](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference/tree/main/isaac_ros_triton)

## Setup
1. Create a ROS2 workspace if one is not already prepared:
   ```
   mkdir -p your_ws/src
   ```
   **Note**: The workspace can have any name; this guide assumes you name it `your_ws`.
   
2. Clone the Isaac ROS Image Segmentation, Isaac ROS DNN Inference, and Isaac ROS Common package repositories to `your_ws/src`. Check that you have [Git LFS](https://git-lfs.github.com/) installed before cloning to pull down all large files:
   ```
   sudo apt-get install git-lfs
   
   cd your_ws/src
   git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_segmentation
   git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference
   git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common
   ```

3. Start the Docker interactive workspace:
   ```
   isaac_ros_common/scripts/run_dev.sh your_ws
   ```
   After this command, you will be inside of the container at `/workspaces/isaac_ros-dev`. Running this command in different terminals will attach to the same container.

   **Note**: The rest of this README assumes that you are inside this container.

4. Build and source the workspace:
   ```
   cd /workspaces/isaac_ros-dev
   colcon build && . install/setup.bash
   ```
   **Note**: We recommend rebuilding the workspace each time when source files are edited. To rebuild, first clean the workspace by running `rm -r build install log`.

5. (Optional) Run tests to verify complete and correct installation:
   ```
   colcon test --executor sequential
   ```

### Download Pre-trained Encrypted TLT Model (.etlt) from NGC
The following steps show how to download models, using [`PeopleSemSegnet`](https://ngc.nvidia.com/catalog/models/nvidia:tao:peoplesemsegnet) as an example.

1. From **File Browser** on the **PeopleSemSegnet** [page](https://ngc.nvidia.com/catalog/models/nvidia:tao:peoplesemsegnet), select the model `.etlt` file in the **FILE** list. Copy the `wget` command by clicking **...** in the **ACTIONS** column. 

2. Run the copied command in a terminal to download the ETLT model, as shown in the below example:  
   ```
   wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesemsegnet/versions/deployable_v1.0/files/peoplesemsegnet.etlt
   ```

### Convert the Encrypted TLT Model (.etlt) Format to the TensorRT Engine Plan
`tao-converter` is used to convert encrypted pre-trained models (.etlt) to the TensorRT Engine Plan.  
The pre-built `tao-converter` can be downloaded [here](https://developer.nvidia.com/tao-toolkit-get-started).    

   `tao-converter` is also included in the ISAAC-ROS docker container:  
   | Platform        | Compute library                      | Directory inside docker           |
   | --------------- | ------------------------------------ | --------------------------------- |
   | x86_64          | CUDA 11.3 / cuDNN 8.1 / TensorRT 8.0 | `/opt/nvidia/tao/cuda11.3-trt8.0` |
   | Jetson(aarch64) | Library from Jetpack 4.6             | `/opt/nvidia/tao/jp4.6`           |

   A symbolic link (`/opt/nvidia/tao/tao-converter`) is created to use `tao-converter` across different platforms.   
   **Tip**: Use `tao-converter -h` for more information on using the tool.  

Here are some examples for generating the TensorRT engine file using `tao-converter`:  

1. Generate an engine file for the fp16 data type:
   ```
   mkdir -p /workspaces/isaac_ros-dev/models
   /opt/nvidia/tao/tao-converter -k tlt_encode -d 3,544,960 -p input_1,1x3x544x960,1x3x544x960,1x3x544x960 -t fp16 -e /workspaces/isaac_ros-dev/models/peoplesemsegnet.engine -o softmax_1 peoplesemsegnet.etlt
   ```
   **Note**: The information used above, such as the `model load key` and `input dimension`, can be retrieved from the **PeopleSemSegnet** page under the **Overview** tab. The model input node name and output node name can be found in `peoplesemsegnet_int8.txt` from `File Browser`. The output file is specified using the `-e` option. The tool needs write permission to the output directory.

2. Generate an engine file for the data type int8:  
   ```
   mkdir -p /workspaces/isaac_ros-dev/models
   cd /workspaces/isaac_ros-dev/models

   # Downloading calibration cache file for Int8.  Check model's webpage for updated wget command.
   wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesemsegnet/versions/deployable_v1.0/files/peoplesemsegnet_int8.txt

   # Running tao-converter
   /opt/nvidia/tao/tao-converter -k tlt_encode -d 3,544,960 -p input_1,1x3x544x960,1x3x544x960,1x3x544x960 -t int8 -c peoplesemsegnet_int8.txt -e /workspaces/isaac_ros-dev/models/peoplesemsegnet.engine -o softmax_1 peoplesemsegnet.etlt
   ```

   **Note**: The calibration cache file (specified using the `-c` option) is required to generate the int8 engine file. For the `PeopleSemSegNet` model, it is provided in the **File Browser** tab.

## Package Reference
### `isaac_ros_unet`
#### Overview
The `isaac_ros_unet` package offers functionality for generating raw and colored segmentation masks from images using a trained U-Net model. Either the `Triton Inference Server node` or `TensorRT node` can be used for inference.

Currently, this package targets U-Net image segmentation models. A model used with this package should receive a `NCHW` formatted tensor input and output a `NHWC` tensor that has already been through an activation layer, such as a softmax layer.

**Note**: `N` refers to the batch size, which must be 1, `H` refers to the height of the image, and `W` refers to the width of the image. For the input, `C` refers to the number of color channels in the image; for the output, `C` refers to the number of classes and should represent the confidence/probability of each class.

The provided model is initialized for random class weights. To get a model, visit [NGC](https://ngc.nvidia.com/catalog/). We specifically recommend using [PeopleSemSegnet](https://ngc.nvidia.com/catalog/models/nvidia:tao:peoplesemsegnet). However, the package should work if you train your own U-Net model that performs semantic segmentation, with input and output formats similar to PeopleSemSegnet. This will need to be converted to a TensorRT plan file using the [TAO Toolkit](https://docs.nvidia.com/tao/tao-toolkit/text/tao_toolkit_quick_start_guide.html).

Alternatively, you can supply any model file supported by the `Triton node` or `TensorRT node`.

#### Package Dependencies
- [isaac_ros_dnn_encoders](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference/tree/main/isaac_ros_dnn_encoders)
- [isaac_ros_nvengine_interfaces](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/tree/main/isaac_ros_nvengine_interfaces)
- Inference Packages (can pick either one)
- [isaac_ros_tensor_rt](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference/tree/main/isaac_ros_tensor_rt)
- [isaac_ros_triton](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference/tree/main/isaac_ros_triton)

#### Available Components
| Component         | Topics Subscribed                                              | Topics Published                                                                                                                                                                                                           | Parameters                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ----------------- | -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `UNetDecoderNode` | `tensor_sub`: The tensor that represents the segmentation mask | `unet/raw_segmentation_mask`: The raw segmentation mask, encoded in mono8. Each pixel represents a class label. <br> `unet/colored_segmentation_mask`: The colored segmentation mask. The color palette is user specified. | `queue_size`: The length of the subscription queues, which is `rmw_qos_profile_default.depth` by default  <br> `frame_id`: The coordinate frame ID that the published image header should be set to <br> `tensor_output_order`: The order of the tensor that the node subscribes to. Note: Currently only `NHWC` formatted tensors are supported. <br>  `color_segmentation_mask_encoding`: The image encoding of the colored segmentation mask. This should be either `rgb8` or `bgr8` <br> `color_palette`: A vector of integers where each element represents the rgb color hex code for the corresponding class label. The number of elements should equal the number of classes. Additionally, element number N corresponds to class label N (e.g. element 0 corresponds to class label 0). For example, configure as `[0xFF0000, 0x76b900]` to color class 0 red and class 1 NVIDIA green respectively (other colors can be found [here](https://htmlcolorcodes.com/)). See launch files in `isaac_ros_unet/launch` for more examples. |

## Walkthroughs
### Inference on PeopleSemSegnet using Triton
This walkthrough will run inference on the PeopleSemSegnet from NGC using `Triton`.
1. Obtain the PeopleSemSegnet ETLT file. The input dimension should be `NCHW` and the output dimension should be `NHWC` that has gone through an activation layer (e.g. softmax). The PeopleSemSegnet model follows this criteria.
   ```
   # Create a model repository for version 1
   mkdir -p /tmp/models/peoplesemsegnet/1

   # Download the model
   cd /tmp/models/peoplesemsegnet
   wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesemsegnet/versions/deployable_v1.0/files/peoplesemsegnet.etlt
   ```

2. Convert the `.etlt` file to a TensorRT plan file (which defaults to fp32).
   ```
   /opt/nvidia/tao/tao-converter -k tlt_encode -d 3,544,960 -p input_1,1x3x544x960,1x3x544x960,1x3x544x960 -e /tmp/models/peoplesemsegnet/1/model.plan -o softmax_1 peoplesemsegnet.etlt
   ```
   **Note**: The TensorRT plan file should be named `model.plan`.

3. Create file `/tmp/models/peoplesemsegnet/config.pbtxt` with the following content:
   ```
   name: "peoplesemsegnet"
   platform: "tensorrt_plan"
   max_batch_size: 0
   input [
     {
       name: "input_1"
       data_type: TYPE_FP32
       dims: [ 1, 3, 544, 960 ]
     }
   ]
   output [
     {
       name: "softmax_1"
       data_type: TYPE_FP32
       dims: [ 1, 544, 960, 2 ]
     }
   ]
   version_policy: {
     specific {
       versions: [ 1 ]
     }
   }
   ```

4. Modify the `isaac_ros_unet` launch file located in `/workspaces/isaac_ros-dev/src/isaac_ros_image_segmentation/isaac_ros_unet/launch/isaac_ros_unet_triton.launch.py`. You will need to update the following lines as:
   ```
   'model_name': 'peoplesemsegnet',
   'model_repository_paths': ['/tmp/models'],
   ```
   The rest of the parameters are already set for PeopleSemSegnet. If you are using a custom model, these parameters will also need to be modified.

5. Rebuild and source `isaac_ros_unet`:
   ```
   cd /workspaces/isaac_ros-dev
   colcon build --packages-up-to isaac_ros_unet && . install/setup.bash
   ```

6. Start `isaac_ros_unet` using the launch file:
   ```
   ros2 launch isaac_ros_unet isaac_ros_unet_triton.launch.py
   ```

#### **Using image on disk for inference**

7. Setup `image_publisher` package if not already installed.
   ```
   cd /workspaces/isaac_ros-dev/src 
   git clone --single-branch -b ros2 https://github.com/ros-perception/image_pipeline.git
   cd /workspaces/isaac_ros-dev
   colcon build --packages-up-to image_publisher && . install/setup.bash
   ```

8. In a separate terminal, publish an image to `/image` using `image_publisher`. For testing purposes, we recommend using PeopleSemSegnet sample image, which is located [here](https://developer.nvidia.com/sites/default/files/akamai/NGC_Images/models/peoplenet/input_11ft45deg_000070.jpg).
   ```   
   ros2 run image_publisher image_publisher_node /workspaces/isaac_ros-dev/src/isaac_ros_image_segmentation/isaac_ros_unet/test/test_cases/unet_sample/image.jpg --ros-args -r image_raw:=image
   ```

    <div align="center"><img src="isaac_ros_unet/test/test_cases/unet_sample/image.jpg" width="600px"/></div>

9. In another terminal, launch `rqt_image_viewer` as follows:
   ```
   ros2 run rqt_image_view rqt_image_view
   ```

10. Inside the `rqt_image_view` GUI, change the topic to `/unet/colored_segmentation_mask` to view a colorized segmentation mask. You may also view the raw segmentation, which is published to `/unet/raw_segmentation_mask`, where the raw pixels correspond to the class labels making it unsuitable for human visual inspection.

    <div align="center"><img src="resources/peoplesemsegnet_segimage.png" width="600px"/></div>

#### **Using Isaac Sim for inference**

   First, go back to step 4 of [Inference on PeopleSemSegnet using Triton](#inference-on-peoplesemsegnet-using-triton) and append the following topic remapping `('/image', '/rgb_left')` to the `isaac_ros_unet` launch file located in `/workspaces/isaac_ros-dev/src/isaac_ros_image_segmentation/isaac_ros_unet/launch/isaac_ros_unet_triton.launch.py`.
```
      remappings=[('encoded_tensor', 'tensor_pub'),
                  ('/image', '/rgb_left')]
```
Then complete steps 5 and 6.

7. Make sure you have Isaac Sim [set up](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim.html#setting-up-isaac-sim) correctly and choose the appropriate working environment[[Native](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/setup.html)/[Docker&Cloud](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/setup.html#docker-cloud-deployment)]. For this walkthrough, we are using the native workstation setup for Isaac Sim.

8. See [Running For The First Time](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/first_run.html#) section to launch Isaac Sim from the [app launcher](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/user_interface_launcher.html) and click on the **Isaac Sim** button.

9.  Set up the Isaac Sim ROS2 bridge as described [here](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/ext_omni_isaac_ros_bridge.html#ros2-bridge).

10.   Connect to the Nucleus server as shown in the [Getting Started](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/sample_jetbot.html#getting-started) section if you have not done it already.
11.   Open up the Isaac ROS Common USD scene located at:
   
   `omniverse://<your_nucleus_server>/Isaac/Samples/ROS/Scenario/carter_warehouse_apriltags_worker.usd`.
   
   And wait for it to load completly.

12.  Press **Play** to start publishing data from Isaac Sim.
<div align="center"><img src="resources/Isaac_sim_camera_view.png" width="800px"/></div>

13.  In another terminal, launch `rqt_image_viewer` as follows:
   ```
   ros2 run rqt_image_view rqt_image_view
   ```
14. Inside the `rqt_image_view` GUI, change the topic to `/unet/colored_segmentation_mask` to view a colorized segmentation mask.
    <div align="center"><img src="resources/Segmentation_output.png" width="600px"/></div>

#### **Using Isaac Sim for inference with Hardware in the loop (HIL)**


The following instructions are for a setup where we can run the sample on a Jetson device and Isaac Sim on an x86 machine. We will use the ROS_DOMAIN_ID environment variable to have a separate logical network for Isaac Sim and the sample application. 

NOTE: Before executing any of the ROS commands, make sure to set the ROS_DOMAIN_ID variable first.

1. Complete step 7 of [Using Isaac Sim for inference](#using-isaac-sim-for-inference) section if you have not done it already.
2. Open the location of the Isaac Sim package in the terminal by clicking the [**Open in Terminal**](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/user_interface_launcher.html) button.
   
   <div align="center"><img src="resources/Isaac_sim_app_launcher.png" width="400px"/></div>
3. In the terminal opened by the previous step, set the ROS_DOMAIN_ID as shown:

   `export ROS_DOMAIN_ID=<some_number>`

4. Launch Isaac Sim from the script as shown:
   
   `./isaac-sim.sh` 
   <div align="center"><img src="resources/Isaac_sim_app_terminal.png" width="600px"/></div>
5. Continue with step 9 of [Using Isaac Sim for inference](#using-isaac-sim-for-inference) section. Make sure to set the ROS_DOMAIN_ID variable before running the sample application.


These steps can easily be adapted to using TensorRT by referring to the TensorRT inference section and modifying step 3-4.

**Note:** For best results, crop/resize input images to the same dimensions your DNN model is expecting.

If you are interested in using a custom model of the U-Net architecture, please read the analogous steps for configuring [DOPE](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation/tree/main/isaac_ros_dope).
  
To configure the launch file for your specific model, consult earlier documentation that describes each of these parameters. Once again, remember to verify that the preprocessing and postprocessing supported by the nodes fit your models. For example, the model should expect a `NCHW` formatted tensor, and output a `NHWC` tensor that has gone through a activation layer (e.g. softmax).

## Troubleshooting
### Nodes crashed on initial launch reporting shared libraries have a file format not recognized
Many dependent shared library binary files are stored in `git-lfs`. These files need to be fetched in order for Isaac ROS nodes to function correctly.

#### Symptoms
```
/usr/bin/ld:/workspaces/isaac_ros-dev/ros_ws/src/isaac_ros_common/isaac_ros_nvengine/gxf/lib/gxf_jetpack46/core/libgxf_core.so: file format not recognized; treating as linker script
/usr/bin/ld:/workspaces/isaac_ros-dev/ros_ws/src/isaac_ros_common/isaac_ros_nvengine/gxf/lib/gxf_jetpack46/core/libgxf_core.so:1: syntax error
collect2: error: ld returned 1 exit status
make[2]: *** [libgxe_node.so] Error 1
make[1]: *** [CMakeFiles/gxe_node.dir/all] Error 2
make: *** [all] Error 2
```
#### Solution
Run `git lfs pull` in each Isaac ROS repository you have checked out, especially `isaac_ros_common`, to ensure all of the large binary files have been downloaded.

# Updates

| Date       | Changes                            |
| ---------- | ---------------------------------- |
| 2021-11-15 | Isaac Sim HIL documentation update |
| 2021-10-20 | Initial release                    |

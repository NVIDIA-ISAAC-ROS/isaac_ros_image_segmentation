# Tutorial for DNN Image Segmentation with Isaac Sim

<div align="center"><img alt="Coloured Segmentation Mask" src="../resources/Isaac_sim_peoplesemsegnet_shuffleseg_rqt.png" width="600px" title="U-Net Shuffleseg result in rqt_image_view"/></div>

## Overview

This tutorial walks you through a pipeline for [Image Segmentation](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_segmentation) of people using images from Isaac Sim.

## Tutorial Walkthrough

1. Complete the [Quickstart section](../README.md#quickstart) in the main README till step 9.
2. Launch the Docker container using the `run_dev.sh` script:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```

3. Inside the container, build and source the workspace:  

    ```bash
    cd /workspaces/isaac_ros-dev && \
      colcon build --symlink-install && \
      source install/setup.bash
    ```

4. Install and launch Isaac Sim following the steps in the [Isaac ROS Isaac Sim Setup Guide](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/isaac-sim-sil-setup.md)
5. Open up the Isaac ROS Common USD scene (using the "content" window) located at:
   `omniverse://localhost/NVIDIA/Assets/Isaac/2022.1/Isaac/Samples/ROS2/Scenario/carter_warehouse_apriltags_worker.usd`.

   And wait for it to load completely.
   > **Note:** To use a different server, replace `localhost` with `<your_nucleus_server>`
6. Go to the stage tab and select `/World/Carter_ROS`, then in properties tab -> Transform -> Translate -> X change `-3.0` to `0.0`.
    <div align="center"><img src="../resources/Isaac_sim_set_carter.png" width="400px"/></div>

7. Change the left camera topic name. Go to the stage tab and select `/World/Carter_ROS/ROS_Cameras/ros2_create_camera_left_rgb`, properties tab -> Compute Node -> Inputs -> topicName change `rgb_left` to `image`.
    <div align="center"><img src="../resources/Isaac_sim_topic_rename.png" width="400px"/></div>
8. Press **Play** to start publishing data from the Isaac Sim application.
    <div align="center"><img src="../resources/Isaac_sim_image_segmentation.png" width="800px"/></div>

9. Run the following launch files to start the inferencing:

    ```bash
        ros2 launch isaac_ros_unet isaac_ros_unet_triton.launch.py model_name:=peoplesemsegnet_shuffleseg model_repository_paths:=['/tmp/models'] input_binding_names:=['input_2:0'] output_binding_names:=['argmax_1'] network_output_type:='argmax'
    ```

10. Visualize and validate the output of the package by launching `rqt_image_view` in another terminal:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
        ./scripts/run_dev.sh
    ```

    Then launch `rqt_image_view`:

    ```bash
        ros2 run rqt_image_view rqt_image_view
    ```

    Then inside the `rqt_image_view` GUI, change the topic to `/unet/colored_segmentation_mask` to view a colorized segmentation mask.

    **Note:** The raw segmentation is also published to `/unet/raw_segmentation_mask`. However, the raw pixels correspond to the class labels and so the output is unsuitable for human visual inspection.

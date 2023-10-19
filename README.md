# Isaac ROS Image Segmentation

Hardware-accelerated, deep learned semantic image segmentation

<div align="center"><img alt="sample input to image segmentation" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_image_segmentation_example.png/" width="320px"/>
<img alt="sample output from image segmentation" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_image_segmentation_example_seg.png/" width="320px"/></div>

## Overview

[Isaac ROS Image Segmentation](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_segmentation) contains a ROS 2 package to produce semantic image segmentation.
`isaac_ros_unet` provides a method for classification of an input image at the pixel level.
Each pixel of the input image is predicted to belong to a set of defined classes.
Classification is performed with GPU acceleration running DNN inference on a U-NET architecture model.
The output prediction can be used by perception functions to understand where each
class is spatially in a 2D image or fuse with a corresponding depth location in a 3D scene.

<div align="center"><a class="reference internal image-reference" href="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_image_segmentation_nodegraph.png/"><img alt="image" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_image_segmentation_nodegraph.png/" width="500px"/></a></div>

A trained model based on
the [U-NET](https://en.wikipedia.org/wiki/U-Net) architecture is
required to produce a segmentation mask. Input images may need to be
cropped and resized to maintain the aspect ratio and match the input
resolution of the U-NET DNN; image resolution may be reduced to improve
DNN inference performance, which typically scales directly with the
number of pixels in the image.

<div align="center"><a class="reference internal image-reference" href="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_image_segmentation_example_bboxseg.png/"><img alt="image" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_image_segmentation_example_bboxseg.png/" width="320px"/></a></div>

Image segmentation provides more information and uses more compute than
object detection to produce classifications per pixel, whereas object
detection classifies a simpler bounding box rectangle in image
coordinates. Object detection is used to know if, and where spatially in
a 2D image, the object exists. On the other hand, image segmentation is used to know which
pixels belong to the class. One application is using the segmentation result, and fusing it with the corresponding depth
information in order to know an object location in a 3D scene.

## Isaac ROS NITROS Acceleration

This package is powered by [NVIDIA Isaac Transport for ROS (NITROS)](https://developer.nvidia.com/blog/improve-perception-performance-for-ros-2-applications-with-nvidia-isaac-transport-for-ros/), which leverages type adaptation and negotiation to optimize message formats and dramatically accelerate communication between participating nodes.

## Performance

| Sample Graph<br/><br/>                                                                                                                                      | Input Size<br/><br/>     | AGX Orin<br/><br/>                                                                                                                                     | Orin NX<br/><br/>                                                                                                                                     | Orin Nano 8GB<br/><br/>                                                                                                                                | x86_64 w/ RTX 4060 Ti<br/><br/>                                                                                                                          |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| [TensorRT Graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/scripts/isaac_ros_unet_graph.py)<br/><br/><br/>PeopleSemSegNet<br/><br/> | 544p<br/><br/><br/><br/> | [421 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_unet_graph-agx_orin.json)<br/><br/><br/>8.1 ms<br/><br/> | [238 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_unet_graph-orin_nx.json)<br/><br/><br/>9.6 ms<br/><br/> | [162 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_unet_graph-orin_nano.json)<br/><br/><br/>13 ms<br/><br/> | [704 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_unet_graph-nuc_4060ti.json)<br/><br/><br/>5.5 ms<br/><br/> |

---

## Documentation

Please visit the [Isaac ROS Documentation](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/index.html) to learn how to use this repository.

---

## Packages

* [`isaac_ros_unet`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_unet/index.html)
  * [Quickstart](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_unet/index.html#quickstart)
  * [Try More Examples](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_unet/index.html#try-more-examples)
  * [Troubleshooting](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_unet/index.html#troubleshooting)
  * [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_unet/index.html#api)

## Latest

Update 2023-10-18: Updated for Isaac ROS 2.0.0.

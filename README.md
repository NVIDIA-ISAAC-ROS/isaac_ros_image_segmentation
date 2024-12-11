# Isaac ROS Image Segmentation

NVIDIA-accelerated, deep learned semantic image segmentation

<div align="center"><img alt="sample input to image segmentation" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_image_segmentation_example.png/" width="320px"/>
<img alt="sample output from image segmentation" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_image_segmentation_example_seg.png/" width="320px"/></div>

## Overview

[Isaac ROS Image Segmentation](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_segmentation) contains ROS packages for semantic image segmentation.

These packages provide methods for classification of an input image
at the pixel level by running GPU-accelerated inference on a DNN model.
Each pixel of the input image is predicted to belong to a set of defined classes.
The output prediction can be used by perception functions to understand where each
class is spatially in a 2D image or fuse with a corresponding depth location in a 3D scene.

<div align="center"><a class="reference internal image-reference" href="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_image_segmentation_nodegraph.png/"><img alt="image" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_image_segmentation_nodegraph.png/" width="500px"/></a></div>

| Package                                                                                                                                                                  | Model Architecture                                | Description                                                              |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|--------------------------------------------------------------------------|
| [Isaac ROS U-NET](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_unet/index.html#quickstart)                        | [U-NET](https://en.wikipedia.org/wiki/U-Net)      | Convolutional network popular for biomedical imaging segmentation models |
| [Isaac ROS Segformer](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segformer/index.html#quickstart)               | [Segformer](https://arxiv.org/abs/2105.15203)     | Transformer-based network that works well for objects of varying scale   |
| [Isaac ROS Segment Anything](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segment_anything/index.html#quickstart) | [Segment Anything](https://segment-anything.com/) | Segments any object in an image when given a prompt as to which one      |

Input images may need to be cropped and resized to maintain the aspect ratio and match the input
resolution expected by the DNN model; image resolution may be reduced to improve
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

| Sample Graph<br/><br/>                                                                                                                                                                                                                  | Input Size<br/><br/>     | AGX Orin<br/><br/>                                                                                                                                                   | Orin NX<br/><br/>                                                                                                                                                    | Orin Nano 8GB<br/><br/>                                                                                                                                               | x86_64 w/ RTX 4090<br/><br/>                                                                                                                                        |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [SAM Image Segmentation Graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/benchmarks/isaac_ros_segment_anything_benchmark/scripts/isaac_ros_segment_anything_graph.py)<br/><br/><br/>Full SAM<br/><br/>          | 720p<br/><br/><br/><br/> | [2.22 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_sam_graph-agx_orin.json)<br/><br/><br/>390 ms @ 30Hz<br/><br/>        | –<br/><br/><br/><br/>                                                                                                                                                | –<br/><br/><br/><br/>                                                                                                                                                 | [16.4 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_sam_graph-x86-4090.json)<br/><br/><br/>280 ms @ 30Hz<br/><br/>       |
| [SAM Image Segmentation Graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/benchmarks/isaac_ros_segment_anything_benchmark/scripts/isaac_ros_mobile_segment_anything_graph.py)<br/><br/><br/>Mobile SAM<br/><br/> | 720p<br/><br/><br/><br/> | [8.75 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_mobile_sam_graph-agx_orin.json)<br/><br/><br/>570 ms @ 30Hz<br/><br/> | [5.34 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_mobile_sam_graph-orin_nx.json)<br/><br/><br/>1400 ms @ 30Hz<br/><br/> | [2.22 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_mobile_sam_graph-orin_nano.json)<br/><br/><br/>340 ms @ 30Hz<br/><br/> | [68.6 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_mobile_sam_graph-x86-4090.json)<br/><br/><br/>23 ms @ 30Hz<br/><br/> |
| [TensorRT Graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/benchmarks/isaac_ros_unet_benchmark/scripts/isaac_ros_unet_graph.py)<br/><br/><br/>PeopleSemSegNet<br/><br/>                                         | 544p<br/><br/><br/><br/> | [371 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_unet_graph-agx_orin.json)<br/><br/><br/>19 ms @ 30Hz<br/><br/>         | [250 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_unet_graph-orin_nx.json)<br/><br/><br/>20 ms @ 30Hz<br/><br/>          | [163 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_unet_graph-orin_nano.json)<br/><br/><br/>23 ms @ 30Hz<br/><br/>         | [670 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_unet_graph-nuc_4060ti.json)<br/><br/><br/>11 ms @ 30Hz<br/><br/>      |

---

## Documentation

Please visit the [Isaac ROS Documentation](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/index.html) to learn how to use this repository.

---

## Packages

* [`isaac_ros_segformer`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segformer/index.html)
  * [Quickstart](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segformer/index.html#quickstart)
  * [Try More Examples](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segformer/index.html#try-more-examples)
  * [Troubleshooting](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segformer/index.html#troubleshooting)
  * [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segformer/index.html#api)
* [`isaac_ros_segment_anything`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segment_anything/index.html)
  * [Quickstart](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segment_anything/index.html#quickstart)
  * [Try More Examples](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segment_anything/index.html#try-more-examples)
  * [Troubleshooting](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segment_anything/index.html#troubleshooting)
  * [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segment_anything/index.html#api)
* [`isaac_ros_unet`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_unet/index.html)
  * [Quickstart](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_unet/index.html#quickstart)
  * [Try More Examples](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_unet/index.html#try-more-examples)
  * [Troubleshooting](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_unet/index.html#troubleshooting)
  * [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_unet/index.html#api)

## Latest

Update 2024-12-10: Update to be compatible with JetPack 6.1

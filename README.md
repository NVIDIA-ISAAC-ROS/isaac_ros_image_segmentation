# Isaac ROS Image Segmentation

NVIDIA-accelerated, deep learned semantic image segmentation

<div align="center"><img alt="sample input to image segmentation" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-4.2/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_image_segmentation_example.png/" width="320px"/>
<img alt="sample output from image segmentation" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-4.2/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_image_segmentation_example_seg.png/" width="320px"/></div>

## Overview

Isaac ROS Image Segmentation contains ROS packages for semantic image segmentation.

These packages provide methods for classification of an input image
at the pixel level by running GPU-accelerated inference on a DNN model.
Each pixel of the input image is predicted to belong to a set of defined classes.
The output prediction can be used by perception functions to understand where each
class is spatially in a 2D image or fuse with a corresponding depth location in a 3D scene.

<div align="center"><a class="reference internal image-reference" href="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-4.2/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_image_segmentation_nodegraph.png/"><img alt="image" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-4.2/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_image_segmentation_nodegraph.png/" width="500px"/></a></div>

| Package                                                                                                                                                                    | Model Architecture                                                                                          | Description                                                                          |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| [Isaac ROS U-NET](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_unet/index.html#quickstart)                          | [U-NET](https://en.wikipedia.org/wiki/U-Net)                                                                | Convolutional network popular for biomedical imaging segmentation models             |
| [Isaac ROS Segformer](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segformer/index.html#quickstart)                 | [Segformer](https://arxiv.org/abs/2105.15203)                                                               | Transformer-based network that works well for objects of varying scale               |
| [Isaac ROS Segment Anything](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segment_anything/index.html#quickstart)   | [Segment Anything](https://github.com/facebookresearch/segment-anything)                                    | Segments any object in an image when given a prompt as to which one                  |
| [Isaac ROS Segment Anything2](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segment_anything2/index.html#quickstart) | [Segment Anything2](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/) | Segments and tracks any object in a video stream when given a prompt as to which one |

Input images may need to be cropped and resized to maintain the aspect ratio and match the input
resolution expected by the DNN model; image resolution may be reduced to improve
DNN inference performance, which typically scales directly with the
number of pixels in the image.

<div align="center"><a class="reference internal image-reference" href="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-4.2/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_image_segmentation_example_bboxseg.png/"><img alt="image" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-4.2/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_image_segmentation_example_bboxseg.png/" width="320px"/></a></div>

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

| Sample Graph<br/><br/>                                                                                                                                                                                                                         | Input Size<br/><br/>   | AGX Thor T5000<br/><br/>                                                                                                                                                    | AGX Thor T4000<br/><br/>                                                                                                                                                      | DGX Spark<br/><br/>                                                                                                                                                         | x86_64 w/ RTX 5090<br/><br/>                                                                                                                                               |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [SAM Image Segmentation Graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.2/benchmarks/isaac_ros_segment_anything_benchmark/scripts/isaac_ros_segment_anything_graph.py)<br/><br/><br/>Full SAM<br/><br/>          | 720p<br/><br/>         | [2.26 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.2/results/isaac_ros_sam_graph-agx_thor.json)<br/><br/><br/>350 ms @ 30Hz<br/><br/>        | [2.24 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.2/results/isaac_ros_sam_graph-thor-t4000.json)<br/><br/><br/>290 ms @ 30Hz<br/><br/>        | [2.22 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.2/results/isaac_ros_sam_graph-dgx_spark.json)<br/><br/><br/>280 ms @ 30Hz<br/><br/>       | [20.8 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.2/results/isaac_ros_sam_graph-x86-5090.json)<br/><br/><br/>57 ms @ 30Hz<br/><br/>        |
| [SAM Image Segmentation Graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.2/benchmarks/isaac_ros_segment_anything_benchmark/scripts/isaac_ros_mobile_segment_anything_graph.py)<br/><br/><br/>Mobile SAM<br/><br/> | 720p<br/><br/>         | [15.0 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.2/results/isaac_ros_mobile_sam_graph-agx_thor.json)<br/><br/><br/>230 ms @ 30Hz<br/><br/> | [15.0 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.2/results/isaac_ros_mobile_sam_graph-thor-t4000.json)<br/><br/><br/>200 ms @ 30Hz<br/><br/> | [14.6 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.2/results/isaac_ros_mobile_sam_graph-dgx_spark.json)<br/><br/><br/>82 ms @ 30Hz<br/><br/> | [70.3 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.2/results/isaac_ros_mobile_sam_graph-x86-5090.json)<br/><br/><br/>20 ms @ 30Hz<br/><br/> |
| [TensorRT Graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.2/benchmarks/isaac_ros_unet_benchmark/scripts/isaac_ros_unet_graph.py)<br/><br/><br/>PeopleSemSegNet<br/><br/>                                         | 544p<br/><br/>         | [449 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.2/results/isaac_ros_unet_graph-agx_thor.json)<br/><br/><br/>8.1 ms @ 30Hz<br/><br/>        | [319 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.2/results/isaac_ros_unet_graph-thor-t4000.json)<br/><br/><br/>19 ms @ 30Hz<br/><br/>         | [562 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.2/results/isaac_ros_unet_graph-dgx_spark.json)<br/><br/><br/>6.7 ms @ 30Hz<br/><br/>       | [1330 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.2/results/isaac_ros_unet_graph-x86-5090.json)<br/><br/><br/>6.3 ms @ 30Hz<br/><br/>      |

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
* [`isaac_ros_segment_anything2`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segment_anything2/index.html)
  * [Quickstart](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segment_anything2/index.html#quickstart)
  * [Troubleshooting](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segment_anything2/index.html#troubleshooting)
  * [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segment_anything2/index.html#api)
* [`isaac_ros_segment_anything2_interfaces`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segment_anything2_interfaces/index.html)
* [`isaac_ros_unet`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_unet/index.html)
  * [Quickstart](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_unet/index.html#quickstart)
  * [Try More Examples](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_unet/index.html#try-more-examples)
  * [Troubleshooting](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_unet/index.html#troubleshooting)
  * [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_unet/index.html#api)

## Latest

Update 2026-02-19: Support for DGX Spark and JetPack 7.1

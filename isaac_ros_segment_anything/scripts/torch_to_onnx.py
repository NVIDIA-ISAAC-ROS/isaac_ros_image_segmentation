#!/usr/bin/env python3

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import warnings

import torch
import torch.nn as nn

try:
    import onnxruntime
    onnxruntime_exists = True
except ImportError:
    onnxruntime_exists = False


class Model(nn.Module):

    def __init__(
        self,
        sam_onnx
    ) -> None:
        super().__init__()
        self.sam_onnx = sam_onnx

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: torch.Tensor,
        has_mask_input: torch.Tensor,
        orig_im_size: torch.Tensor,
    ):
        image_embeddings = self.sam_onnx.model.image_encoder.forward(images)
        return self.sam_onnx.forward(
            image_embeddings, point_coords,
            point_labels, mask_input,
            has_mask_input, orig_im_size
        )


def get_parser():
    parser = argparse.ArgumentParser(
        description='Export the Image Encoder, SAM prompt encoder and mask decoder \
        to an ONNX model.'
    )

    parser.add_argument(
        '--checkpoint', type=str, required=True, help='The path to the SAM model checkpoint.'
    )

    parser.add_argument(
        '--output', type=str, required=True, help='The filename to save the ONNX model to.'
    )

    parser.add_argument(
        '--model-type',
        type=str,
        required=True,
        help='In ["default", "vit_h", "vit_l", "vit_b", "vit_t"]. Type of SAM model to export.',
    )

    parser.add_argument(
        '--sam-type',
        type=str,
        required=True,
        help='In ["SAM", "MobileSAM"]. Which type of SAM model to export.',
    )

    parser.add_argument(
        '--opset',
        type=int,
        default=17,
        help='The ONNX opset version to use. Must be >=11',
    )

    return parser


def run_export(
    model_type: str,
    checkpoint: str,
    output: str,
    opset: int,
    sam_type: str
):
    if (sam_type == 'SAM'):
        from segment_anything import sam_model_registry
        from segment_anything.utils.onnx import SamOnnxModel
    else:
        from mobile_sam import sam_model_registry
        from mobile_sam.utils.onnx import SamOnnxModel

    print('Loading model...')
    sam = sam_model_registry[model_type](checkpoint=checkpoint)

    decoder_onnx_model = SamOnnxModel(
        model=sam,
        return_single_mask=True
    )

    dynamic_axes = {
        'point_coords': {0: 'batch_size', 1: 'num_points'},
        'point_labels': {0: 'batch_size', 1: 'num_points'},
    }

    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]

    dummy_inputs = {
        'images': torch.randn(1, 3, 1024, 1024),
        'point_coords': torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        'point_labels': torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        'mask_input': torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        'has_mask_input': torch.tensor([1], dtype=torch.float),
        'orig_im_size': torch.tensor([1500, 2250], dtype=torch.float),
    }

    onnx_model = Model(decoder_onnx_model)
    _ = onnx_model(**dummy_inputs)

    output_names = ['masks', 'iou_predictions', 'low_res_masks']

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        with open(output, 'wb') as f:
            print(f'Exporting onnx model to {output}...')
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=opset,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                dynamo=False  # Use legacy exporter for dynamic shapes
            )

    if onnxruntime_exists:
        ort_inputs = {k: to_numpy(v) for k, v in dummy_inputs.items()}
        # set cpu provider default
        providers = ['CPUExecutionProvider']
        ort_session = onnxruntime.InferenceSession(output, providers=providers)
        _ = ort_session.run(None, ort_inputs)
        print('Model has successfully been run with ONNXRuntime.')


def to_numpy(tensor):
    return tensor.cpu().numpy()


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    run_export(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        output=args.output,
        opset=args.opset,
        sam_type=args.sam_type
    )

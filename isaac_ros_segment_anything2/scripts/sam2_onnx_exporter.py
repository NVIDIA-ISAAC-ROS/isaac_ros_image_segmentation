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

from sam2.build_sam import build_sam2
from sam2_wrapper import SAM2Wrapper
import torch


def apply_rotary_enc_real(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
):
    """
    Apply rotary position encoding using real-valued operations instead of complex numbers.

    This is functionally equivalent to apply_rotary_enc but avoids using complex tensors.
    """
    # Extract real and imaginary parts from freqs_cis
    # Original freqs_cis was complex with shape [seq_len, head_dim//2]
    # We reshape to get cos and sin components
    seq_len, half_head_dim = freqs_cis.shape
    cos_parts = freqs_cis.real
    sin_parts = freqs_cis.imag

    # Reshape input tensors
    xq_reshaped = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xq_reshaped_real, xq_reshaped_imag = xq_reshaped[..., 0], xq_reshaped[..., 1]

    # Reshape for broadcasting (equivalent to reshape_for_broadcast)
    ndim = xq.ndim
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(xq.shape[:-1])] + [half_head_dim]
    cos_parts = cos_parts.view(*shape)
    sin_parts = sin_parts.view(*shape)

    # Apply rotation: [cos_θ -sin_θ; sin_θ cos_θ] · [x_real; x_imag]
    # This is equivalent to complex multiplication
    xq_out_real = xq_reshaped_real * cos_parts - xq_reshaped_imag * sin_parts
    xq_out_imag = xq_reshaped_real * sin_parts + xq_reshaped_imag * cos_parts
    xq_out = torch.stack([xq_out_real, xq_out_imag], dim=-1).flatten(3)

    # Handle case where xk is empty (from dropout)
    if xk.shape[-2] == 0:
        return xq_out.type_as(xq).to(xq.device), xk

    # Process key tensor
    xk_reshaped = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xk_reshaped_real, xk_reshaped_imag = xk_reshaped[..., 0], xk_reshaped[..., 1]

    # Repeat frequencies if needed
    if repeat_freqs_k:
        r = xk.shape[-2] // xq.shape[-2]
        cos_parts_k = cos_parts.repeat(*([1] * (cos_parts.ndim - 2)), r, 1)
        sin_parts_k = sin_parts.repeat(*([1] * (sin_parts.ndim - 2)), r, 1)
    else:
        cos_parts_k = cos_parts
        sin_parts_k = sin_parts

    # Apply rotation to key tensor
    xk_out_real = xk_reshaped_real * cos_parts_k - xk_reshaped_imag * sin_parts_k
    xk_out_imag = xk_reshaped_real * sin_parts_k + xk_reshaped_imag * cos_parts_k
    xk_out = torch.stack([xk_out_real, xk_out_imag], dim=-1).flatten(3)

    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


def patch_rotary_for_onnx_export():
    """Temporarily patch the apply_rotary_enc function with a real-valued implementation."""
    print('Patching rotary encoding function for ONNX export...')
    import sam2.modeling.position_encoding as pe
    original_func = pe.apply_rotary_enc
    pe.apply_rotary_enc = apply_rotary_enc_real
    return original_func


def restore_rotary_function(original_func):
    """Restore the original rotary encoding function."""
    import sam2.modeling.position_encoding as pe
    pe.apply_rotary_enc = original_func


def export_sam2_to_onnx(model,
                        output_path='sam2_tracker.onnx',
                        input_shape=(1, 3, 1024, 1024),
                        maskmem_shape=(5, 4, 64, 64, 64),
                        obj_ptr_shape=(5, 2, 256),
                        fp16=False):
    """Export the SAM2Wrapper model to ONNX format."""
    # Create dummy inputs for the model
    dummy_image = torch.randn(*input_shape, device='cpu')
    dummy_maskmem = torch.randn(*maskmem_shape, device='cpu')
    dummy_maskmem[1] = torch.zeros_like(dummy_maskmem[0], device='cpu')
    dummy_maskmem[2] = torch.zeros_like(dummy_maskmem[0], device='cpu')
    dummy_obj_ptr = torch.zeros(*obj_ptr_shape, device='cpu')

    # Original image size (height, width) - needed for proper aspect ratio handling
    dummy_orig_size = torch.tensor([720, 1280], device='cpu', dtype=torch.int32)

    # Add a dummy bounding box
    dummy_bbox = torch.tensor([
        [100., 100., 300., 300.],
        [200., 200., 400., 400.],
        [0, 0, 0, 0]], device='cpu')
    dummy_point_coords = torch.tensor([
        [[500, 500], [0, 0]],
        [[600, 600], [605, 605]],
        [[0, 0], [0, 0]]], dtype=torch.float32, device='cpu')
    dummy_permutation = torch.tensor([0, 1, 2, 3, 4], device='cpu')
    dummy_point_labels = torch.tensor([[1, -1], [1, 0], [-1, -1]], dtype=torch.int32, device='cpu')
    # Set the model to evaluation mode
    model.eval()

    # Create dynamic_axes for ONNX export
    dynamic_axes = {
        'image': {0: 'batch_size'},
        'mask_memory': {0: 'num_objects'},
        'obj_ptr_memory': {0: 'num_objects'},
        'bbox_coords': {0: 'num_boxes'},
        'original_size': {0: 'size_dim'},
        'permutation': {0: 'num_objects'},
        'point_coords': {0: 'num_objects', 1: 'num_points'},
        'point_labels': {0: 'num_objects', 1: 'num_points'},
        'high_res_masks': {0: 'num_objects'},
        'maskmem_features': {0: 'num_objects'},
        'maskmem_pos_enc': {0: 'num_objects'},
        'obj_ptr_features': {0: 'num_objects'}
    }

    # Export the model using legacy exporter for dynamic shapes
    torch.onnx.export(
        model,
        (dummy_image, dummy_maskmem,
         dummy_obj_ptr, dummy_orig_size,
         dummy_permutation, dummy_bbox,
         dummy_point_coords,
         dummy_point_labels),
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['image', 'mask_memory',
                     'obj_ptr_memory', 'original_size',
                     'permutation', 'bbox_coords',
                     'point_coords', 'point_labels'],
        output_names=['high_res_masks', 'object_score_logits',
                      'maskmem_features', 'maskmem_pos_enc', 'obj_ptr_features'],
        dynamic_axes=dynamic_axes,
        dynamo=False  # Use legacy exporter for dynamic shapes
    )
    if fp16:
        # Post-export optimization to fp16
        print('Optimizing model to fp16...')
        import onnx
        from onnxconverter_common import float16

        # Load the exported model
        onnx_model = onnx.load(output_path)

        # Convert weights to fp16 while preserving input/output types
        fp16_model = float16.convert_float_to_float16(
            onnx_model,
            keep_io_types=True  # This preserves original input/output datatypes
        )
        onnx.save(fp16_model, output_path)
        print(f'Model successfully exported using fp16 to {output_path}')


if __name__ == '__main__':
    # Parse command line arguments
    CONFIG_NAME_TO_PATH = {
        'tiny': 'configs/sam2.1/sam2.1_hiera_t.yaml',
        'small': 'configs/sam2.1/sam2.1_hiera_s.yaml',
        'base_plus': 'configs/sam2.1/sam2.1_hiera_b+.yaml',
        'large': 'configs/sam2.1/sam2.1_hiera_l.yaml'
    }

    import argparse
    parser = argparse.ArgumentParser(description='Export SAM2Wrapper to ONNX')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to the SAM2 checkpoint')
    parser.add_argument('--config_name', default='tiny', choices=CONFIG_NAME_TO_PATH.keys(),
                        help='Name of the model configuration file')
    parser.add_argument('--output', default='sam2_tracker.onnx',
                        help='Output path for the ONNX model')
    parser.add_argument('--fp16', action='store_true',
                        help='Export the model in FP16')
    args = parser.parse_args()
    original_rotary_func = patch_rotary_for_onnx_export()
    # Load the SAM2 model
    print(f'Loading SAM2 model from {args.checkpoint}...')
    sam2_model = build_sam2(CONFIG_NAME_TO_PATH[args.config_name],
                            ckpt_path=args.checkpoint, device='cpu')
    # Create the SAM2Wrapper
    print('Creating SAM2Wrapper...')
    model = SAM2Wrapper(sam2_model)

    # Export the model
    print(f'Exporting model to {args.output}...')
    export_sam2_to_onnx(model, args.output, fp16=args.fp16)
    restore_rotary_function(original_rotary_func)
    print('Export completed.')

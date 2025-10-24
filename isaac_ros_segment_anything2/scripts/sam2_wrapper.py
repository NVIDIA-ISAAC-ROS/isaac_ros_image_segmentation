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
from sam2.modeling.sam2_utils import get_1d_sine_pe
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAM2Wrapper(nn.Module):

    def __init__(self, sam2_model):
        """Create a streamlined version of SAM2 for real-time tracking."""
        super().__init__()
        self.sam2_model = sam2_model
        self.no_obj_embed_spatial = getattr(sam2_model, 'no_obj_embed_spatial', None)
        self.soft_no_obj_ptr = getattr(sam2_model, 'soft_no_obj_ptr', False)
        self.fixed_no_obj_ptr = getattr(sam2_model, 'fixed_no_obj_ptr', False)
        self.no_obj_ptr = getattr(sam2_model, 'no_obj_ptr', None)
        # Constants
        self.NO_OBJ_SCORE = -1024.0

    def forward_image(self, img_batch: torch.Tensor):
        """Get the image feature on the input batch."""
        backbone_out = self.sam2_model.image_encoder(img_batch)
        if self.sam2_model.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out['backbone_fpn'][0] = self.sam2_model.sam_mask_decoder.conv_s0(
                backbone_out['backbone_fpn'][0]
            )
            backbone_out['backbone_fpn'][1] = self.sam2_model.sam_mask_decoder.conv_s1(
                backbone_out['backbone_fpn'][1]
            )
        return backbone_out

    def prepare_memory_conditioned_features(self, current_vision_feats, current_vision_pos_embeds,
                                            feat_sizes, maskmem_features, obj_ptr_features,
                                            bbox_coords, point_coords):
        """Fully ONNX-exportable implementation of memory conditioning."""
        C = self.sam2_model.hidden_dim
        H, W = feat_sizes[-1]
        device = current_vision_feats[-1].device

        # Get all values as tensors to be ONNX friendly
        total_objects = torch.tensor(maskmem_features.shape[0], device=device, dtype=torch.int64)

        # Calculate object masks - replace any dynamic logic with tensor operations
        total_objects = maskmem_features.shape[0]
        new_objects = torch.tensor(0, device=device)
        new_objects = torch.tensor(bbox_coords.shape[0], device=device)
        new_objects = new_objects + torch.tensor(point_coords.shape[0], device=device)
        # Create object existence masks (1.0 where object exists, 0.0 elsewhere)
        existing_objects = torch.tensor(total_objects, device=device) - new_objects
        obj_indices = torch.arange(total_objects, device=device, dtype=torch.int64)
        existing_mask = (obj_indices < existing_objects).float().view(-1, 1, 1, 1)
        new_mask = 1.0 - existing_mask

        combined_results = torch.zeros(total_objects, C, H, W, device=device)

        # === PATH 1: Process all objects as if they have memory ===
        cond_mem_feats = maskmem_features[:, 0].to(device, non_blocking=True)
        cond_mem_pos = maskmem_features[:, 1].to(device, non_blocking=True)
        last_mem_feats = maskmem_features[:, 2].to(device, non_blocking=True)
        last_mem_pos = maskmem_features[:, 3].to(device, non_blocking=True)

        # Create a safe object count for tensor operations - use max(1, existing_objects)
        # to avoid empty tensors
        safe_existing = torch.maximum(torch.tensor(1, device=device), existing_objects)

        # Reshape for memory attention - safely process first 'safe_existing' objects
        cond_mem_feats_valid = cond_mem_feats[:safe_existing]
        cond_mem_pos_valid = cond_mem_pos[:safe_existing]
        last_mem_feats_valid = last_mem_feats[:safe_existing]
        last_mem_pos_valid = last_mem_pos[:safe_existing]

        # Reshape for memory attention
        cond_mem_feats_flat = cond_mem_feats_valid.flatten(2).permute(2, 0, 1)
        last_mem_feats_flat = last_mem_feats_valid.flatten(2).permute(2, 0, 1)

        # Add temporal position encodings
        cond_mem_pos_flat = cond_mem_pos_valid.flatten(2).permute(2, 0, 1)
        cond_mem_pos_flat = cond_mem_pos_flat + self.sam2_model.maskmem_tpos_enc[- 1]

        last_mem_pos_flat = last_mem_pos_valid.flatten(2).permute(2, 0, 1)
        last_mem_pos_flat = last_mem_pos_flat + self.sam2_model.maskmem_tpos_enc[- 1]

        # Process object pointers
        obj_ptrs_cond = obj_ptr_features[:safe_existing, 0]
        obj_ptrs_last = obj_ptr_features[:safe_existing, 1]
        obj_ptrs = torch.stack([obj_ptrs_cond, obj_ptrs_last], dim=0)

        # Add temporal position encoding to object pointers
        t_diff_max = 1
        tpos_dim = C if self.sam2_model.proj_tpos_enc_in_obj_ptrs else self.sam2_model.mem_dim
        obj_pos_indices = torch.tensor([0, 1], device=device, dtype=torch.float)

        # Continue using get_1d_sine_pe as requested
        obj_pos = get_1d_sine_pe(obj_pos_indices, tpos_dim, t_diff_max)
        obj_pos = self.sam2_model.obj_ptr_tpos_proj(obj_pos)
        # Always expand with the safe_existing count
        obj_pos = obj_pos.unsqueeze(1).expand(-1, safe_existing, self.sam2_model.mem_dim)

        # Handle dimensionality adjustment
        if self.sam2_model.mem_dim < C:
            obj_ptrs = obj_ptrs.reshape(-1,
                                        safe_existing, C // self.sam2_model.mem_dim,
                                        self.sam2_model.mem_dim)
            obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
            obj_pos = obj_pos.repeat_interleave(C // self.sam2_model.mem_dim, dim=0)

        # Concat all memory components
        memory = torch.cat([cond_mem_feats_flat, last_mem_feats_flat, obj_ptrs], dim=0)
        memory_pos = torch.cat([cond_mem_pos_flat, last_mem_pos_flat, obj_pos], dim=0)
        num_obj_ptr_tokens = obj_ptrs.shape[0]

        # Run memory attention - only use up to safe_existing objects
        curr = current_vision_feats[-1][:, :safe_existing]
        curr_pos = current_vision_pos_embeds[-1][:, :safe_existing]

        memory_enhanced = self.sam2_model.memory_attention(
            curr=curr,
            curr_pos=curr_pos,
            memory=memory,
            memory_pos=memory_pos,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )

        # Reshape the output
        memory_enhanced = memory_enhanced.permute(1, 2, 0).view(safe_existing, C, H, W)

        # Fill the result tensor for the existing objects
        # Create a padding tensor for any additional objects beyond safe_existing
        pad_size = total_objects - safe_existing
        if pad_size > 0:
            pad_tensor = torch.zeros(pad_size, C, H, W, device=device)
            memory_enhanced_padded = torch.cat([memory_enhanced, pad_tensor], dim=0)
        else:
            memory_enhanced_padded = memory_enhanced[:total_objects]

        # Store in combined results, masked by existing_mask
        existing_result = memory_enhanced_padded * existing_mask
        # Get features for all objects and add no_mem_embed
        curr_all = current_vision_feats[-1]
        # Reshape to match expected dimensions
        curr_expanded = curr_all.permute(1, 2, 0).view(total_objects, C, H, W)
        # Add no_mem_embed to each object, broadcasting across spatial dimensions
        no_mem_result = curr_expanded + self.sam2_model.no_mem_embed.view(1, C, 1, 1)

        # Apply new_mask to no_mem_result
        new_result = no_mem_result * new_mask

        # Combine both results - each mask ensures only relevant objects are updated
        combined_results = existing_result + new_result
        return combined_results

    def encode_new_memory(self, current_vision_feats,
                          feat_sizes, pred_masks_high_res,
                          object_score_logits, is_mask_from_pts):
        """Encode new memory from the current frame."""
        B = current_vision_feats[-1].size(1)  # batch size
        C = self.sam2_model.hidden_dim
        H, W = feat_sizes[-1]

        # Extract pixel features
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)

        # Apply non-overlapping constraints if needed
        if self.sam2_model.non_overlap_masks_for_mem_enc and not self.training:
            pred_masks_high_res = self._apply_non_overlapping_constraints(pred_masks_high_res)

        # Process mask for memory encoding
        binarize = self.sam2_model.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            # Apply sigmoid on raw mask logits
            mask_for_mem = torch.sigmoid(pred_masks_high_res)

        # Apply scale and bias to the sigmoid probabilities
        if self.sam2_model.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sam2_model.sigmoid_scale_for_mem_enc
        if self.sam2_model.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sam2_model.sigmoid_bias_for_mem_enc

        # Run memory encoder
        maskmem_out = self.sam2_model.memory_encoder(
            pix_feat, mask_for_mem, skip_mask_sigmoid=True  # sigmoid already applied
        )

        maskmem_features = maskmem_out['vision_features'].detach().clone()
        maskmem_pos_enc = maskmem_out['vision_pos_enc'][0].detach().clone()

        # Add no-object embedding to spatial memory if configured
        if self.no_obj_embed_spatial is not None and self.sam2_model.pred_obj_scores:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (
                1 - is_obj_appearing[..., None, None]
            ) * self.no_obj_embed_spatial[..., None, None].expand(
                *maskmem_features.shape
            )

        return maskmem_features, maskmem_pos_enc

    def _get_image_feature(self, image, batch_size):
        """Get the image feature from the image encoder."""
        backbone_out = self.forward_image(image)
        # expand the features to have the same dimension as the number of objects
        # expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            'backbone_fpn': backbone_out['backbone_fpn'].copy(),
            'vision_pos_enc': backbone_out['vision_pos_enc'].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out['backbone_fpn']):
            expanded_backbone_out['backbone_fpn'][i] = feat.expand(
                batch_size, -1, -1, -1
            )
        for i, pos in enumerate(expanded_backbone_out['vision_pos_enc']):
            pos = pos.expand(batch_size, -1, -1, -1)
            expanded_backbone_out['vision_pos_enc'][i] = pos
        features = self.sam2_model._prepare_backbone_features(expanded_backbone_out)
        features = (image,) + features
        return features

    def preprocess_coords(self, bbox_coords, point_coords, original_image_size):
        """Preprocess bounding box coordinates for top-left aligned padding."""
        # Access height and width from tensor by index rather than unpacking
        height = original_image_size[0].float()
        width = original_image_size[1].float()

        # Calculate the scale factor using torch.minimum instead of min()
        scale_h = self.sam2_model.image_size / height
        scale_w = self.sam2_model.image_size / width
        scale = torch.minimum(scale_h, scale_w)

        processed_bbox_coords = bbox_coords * scale.view(1, 1)
        processed_point_coords = point_coords * scale.view(1, 1)
        return processed_bbox_coords, processed_point_coords

    def postprocess_high_res_mask(self, high_res_masks, original_image_size):
        """Postprocess high-res masks for top-left aligned padded images."""
        # Get tensor dimensions safely
        height = original_image_size[0].float()
        width = original_image_size[1].float()

        # Calculate scale factor with tensor operations
        scale = torch.minimum(self.sam2_model.image_size / height,
                              self.sam2_model.image_size / width)

        # Calculate new dimensions with tensor ops instead of int()
        new_height = torch.floor(height * scale).to(torch.int64)
        new_width = torch.floor(width * scale).to(torch.int64)

        # Use torch.narrow instead of dynamic slicing - it's more ONNX-friendly
        # narrow(dim, start, length) extracts a slice from start
        # to start+length in the specified dimension
        # First narrow along height dimension (dim=2)
        narrowed_h = torch.narrow(high_res_masks, 2, 0, new_height)
        # Then narrow along width dimension (dim=3)
        extracted_region = torch.narrow(narrowed_h, 3, 0, new_width)

        # Resize to original dimensions
        resized_masks = F.interpolate(
            extracted_region,
            size=(height.to(torch.int64), width.to(torch.int64)),
            mode='bilinear',
            align_corners=False
        )

        return resized_masks

    def forward(self, image, maskmem_memories,
                obj_ptr_memories, original_image_size,
                permutation, bbox_coords, point_coords, point_labels):
        """Process a new frame for object tracking with ONNX-compatible operations."""
        with torch.no_grad():

            # Get indices for all elements except the last one
            bbox_coords = bbox_coords[:-1]
            point_coords = point_coords[:-1]
            point_labels = point_labels[:-1]

            bbox_coords, point_coords = self.preprocess_coords(bbox_coords,
                                                               point_coords,
                                                               original_image_size)

            # Get image features
            (_,
             _,
             current_vision_feats,
             current_vision_pos_embeds,
             feat_sizes) = self._get_image_feature(image, maskmem_memories.shape[0])

            # Use memory conditioning with vectorized implementation
            pix_feat = self.prepare_memory_conditioned_features(
                current_vision_feats[-1:],
                current_vision_pos_embeds[-1:],
                feat_sizes[-1:],
                maskmem_memories,
                obj_ptr_memories,
                bbox_coords,
                point_coords
            )

            # Get high-res features for mask prediction if needed
            if self.sam2_model.use_high_res_features_in_sam:
                high_res_features = [
                    x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                    for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
                ]
            else:
                high_res_features = None

            # Generate masks with SAM
            high_res_masks, obj_ptr, object_score_logits = self._sam_inference(
                backbone_features=pix_feat,
                bbox_coords=bbox_coords,
                high_res_features=high_res_features,
                multimask_output=False,
                point_coords=point_coords,
                point_labels=point_labels
            )
            # Update memory state if requested and object is present
            maskmem_features, maskmem_pos_enc = self.encode_new_memory(
                current_vision_feats,
                feat_sizes,
                high_res_masks,
                object_score_logits,
                is_mask_from_pts=False
            )
            resized_high_res_masks = self.postprocess_high_res_mask(high_res_masks,
                                                                    original_image_size)
            # Return results
            resized_high_res_masks = torch.gather(resized_high_res_masks, 0,
                                                  permutation.view(-1, 1, 1, 1)
                                                  .expand(-1,
                                                          resized_high_res_masks.size(1),
                                                          resized_high_res_masks.size(2),
                                                          resized_high_res_masks.size(3)))

            object_score_logits = torch.gather(object_score_logits, 0,
                                               permutation.view(-1, 1)
                                               .expand(-1, object_score_logits.size(1)))
            maskmem_features = torch.gather(maskmem_features, 0,
                                            permutation.view(-1, 1, 1, 1)
                                            .expand(-1,
                                                    maskmem_features.size(1),
                                                    maskmem_features.size(2),
                                                    maskmem_features.size(3)))
            maskmem_pos_enc = torch.gather(maskmem_pos_enc, 0,
                                           permutation.view(-1, 1, 1, 1)
                                           .expand(-1,
                                                   maskmem_pos_enc.size(1),
                                                   maskmem_pos_enc.size(2),
                                                   maskmem_pos_enc.size(3)))
            obj_ptr = torch.gather(obj_ptr, 0,
                                   permutation.view(-1, 1).expand(-1, obj_ptr.size(1)))
            return (
                resized_high_res_masks,
                object_score_logits,
                maskmem_features,
                maskmem_pos_enc,
                obj_ptr
            )

    def _get_total_objects(self, backbone_features):
        return backbone_features.shape[0]

    def _prompt_encoder(self, backbone_features, bbox_coords, point_coords, point_labels):
        """Generate prompt embeddings for all objects."""
        # --- compute counts & device ---

        MAX_POINTS_PER_OBJECT = 5
        device = backbone_features.device

        total_objects = torch.tensor(backbone_features.shape[0], device=device, dtype=torch.int64)

        # Calculate counts with tensor operations throughout
        bbox_count = bbox_coords.shape[0]
        point_count = point_coords.shape[0]
        padding_objects = total_objects - bbox_count - point_count
        # --- base embeddings from zero-points (dummy points) ---
        sam_point_coords = torch.zeros(total_objects, MAX_POINTS_PER_OBJECT, 2, device=device)
        sam_point_labels = -torch.ones(total_objects, MAX_POINTS_PER_OBJECT,
                                       dtype=torch.int32, device=device)
        if point_count > 0:
            # Create point labels with the correct values for actual points (1 for foreground)
            # then pad with -1 values for unused points
            padded_labels = -torch.ones(point_count, MAX_POINTS_PER_OBJECT,
                                        dtype=torch.int32, device=device)
            padded_labels[:, :point_coords.shape[1]] = point_labels

            # Pad point coordinates with zeros for unused points
            padded_coords = torch.zeros(point_count, MAX_POINTS_PER_OBJECT, 2, device=device)
            padded_coords[:, :point_coords.shape[1]] = point_coords
            sam_point_coords[padding_objects:padding_objects+point_count] = padded_coords
            sam_point_labels[padding_objects:padding_objects+point_count] = padded_labels

        sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=None,
        )

        # --- overwrite last `new_objects` slots with bbox embeddings ---
        # if no boxes or no new objects, bbox_sparse will be shape (0, …)
        if bbox_count > 0:
            padded_sparse = torch.zeros(bbox_coords.shape[0], MAX_POINTS_PER_OBJECT+1, 256,
                                        device=device)
            bbox_sparse, bbox_dense = self.sam2_model.sam_prompt_encoder(
                points=None,
                boxes=bbox_coords,
                masks=None,
            )
            prefix = total_objects - bbox_coords.shape[0]
            padded_sparse[:, :bbox_sparse.shape[1], :] = bbox_sparse

            # pick first `prefix` from the point embeddings, then concat the bbox ones
            sparse_embeddings = torch.cat([sparse_embeddings[:prefix], padded_sparse], dim=0)
            dense_embeddings = torch.cat([dense_embeddings[:prefix], bbox_dense], dim=0)

        return sparse_embeddings, dense_embeddings

    def _sam_inference(self, backbone_features,
                       bbox_coords, high_res_features=None,
                       multimask_output=False, point_coords=None,
                       point_labels=None):
        """Run SAM inference with the given features and prompts."""
        B = backbone_features.size(0)
        device = backbone_features.device

        sparse_embeddings, dense_embeddings = self._prompt_encoder(backbone_features,
                                                                   bbox_coords,
                                                                   point_coords,
                                                                   point_labels)
        # Run mask decoder
        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = self.sam2_model.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        # Apply object score logic
        if self.sam2_model.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                self.NO_OBJ_SCORE,
            )

        # Upscale masks to high resolution
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.sam2_model.image_size, self.sam2_model.image_size),
            mode='bilinear',
            align_corners=False,
        )

        # Select best mask or use all masks
        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            # Take the best mask prediction (with the highest IoU estimation)
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            high_res_masks = high_res_multimasks

        # Extract object pointer
        obj_ptr = self.sam2_model.obj_ptr_proj(sam_output_token)

        # Apply object pointer logic
        if self.sam2_model.pred_obj_scores:
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr

            if self.no_obj_ptr is not None:
                obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr
        return high_res_masks, obj_ptr, object_score_logits

    def _apply_non_overlapping_constraints(self, pred_masks):
        """Apply non-overlapping constraints to object masks."""
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        # Get object with highest score at each location
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        # Object index of each object slice in pred_masks
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        # Suppress overlapping regions to a low score
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        del max_obj_inds
        del batch_obj_inds
        torch.cuda.empty_cache()
        return pred_masks

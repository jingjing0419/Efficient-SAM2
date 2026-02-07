# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from collections import OrderedDict

import torch
import torch.nn.functional as F
from typing import Tuple, Set

from tqdm import tqdm

from sam2.modeling.sam2_base import NO_OBJ_SCORE, SAM2Base
from sam2.utils.misc import concat_points, fill_holes_in_mask_scores, load_video_frames

import time
import random
import os
import logging
import wandb
import numpy as np
from typing import List, Tuple, Dict
import torch.jit

logger = logging.getLogger('predictor')


def convert_local_to_global_indices(
  local_indices_list: List[Dict[str, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    
    """向量化版本 - 平衡简洁性和性能"""
    if not local_indices_list:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, empty, []

    device = local_indices_list[0]['unm_idx'].device
    
    # 向量化计算偏移量
    sizes = torch.tensor([d['tensor_size'] for d in local_indices_list], device=device)
    offsets = torch.cat([torch.zeros(1, device=device), sizes.cumsum(0)[:-1]])
    
    # 分别处理三种索引
    def process_indices(key):
        indices_with_offsets = [d[key] + offsets[i] for i, d in enumerate(local_indices_list) if len(d[key]) > 0]
        result = torch.cat(indices_with_offsets) if indices_with_offsets else torch.empty(0, dtype=torch.long, device=device)
        return result.to(dtype=torch.int64)
    # torch.set_printoptions(threshold=np.inf)
    # print()
    # print(local_indices_list[1]['unm_idx'])
    # print(local_indices_list[0]['tensor_size'])
    # print(process_indices('unm_idx')[len(local_indices_list[0]['unm_idx']):])
    # print(len(process_indices('unm_idx')))
    # exit()

    global_idx =  {'unm_idx': process_indices('unm_idx'), 
            'src_idx': process_indices('src_idx'), 
            'dst_idx': process_indices('dst_idx'),
            'merge_count': [d['merge_count'] for d in local_indices_list]}
    # print(len(global_idx['unm_idx']))
    # print(len(global_idx['src_idx']))
    # print(len(global_idx['dst_idx']))
    # exit()

    return global_idx


def adaptive_significant_selection_v1(data_batch: torch.Tensor, alpha: float = 2.0) -> Tuple[torch.Tensor, int]:
    """
    进一步优化版本 - 减少内存使用
    """
    batch_size, n = data_batch.shape
    
    # 一次性计算所有统计量
    scaled_data = data_batch * n
    batch_means = scaled_data.mean(dim=1, keepdim=True)
    batch_vars = scaled_data.var(dim=1, keepdim=True, unbiased=False)
    
    # 向量化阈值计算
    thresholds = batch_means + alpha * torch.sqrt(batch_vars / n)
    print(thresholds)
    # 直接计算并集mask
    union_mask = torch.any(scaled_data >= thresholds, dim=0)
    union_indices = torch.nonzero(union_mask, as_tuple=False).squeeze(-1)
    
    # 处理空结果
    if union_indices.numel() == 0:
        union_indices = torch.unique(torch.argmax(data_batch, dim=1))
    
    return union_indices, union_indices.size(0)

def adaptive_significant_selection(data_batch: torch.Tensor, alpha: float = 2.0) -> Tuple[torch.Tensor, int]:
    batch_size, seq_len = data_batch.shape
    
    # 计算每个样本的均值和标准差
    means = data_batch.mean(dim=1, keepdim=True)
    stds = data_batch.std(dim=1, keepdim=True)
    
    # 计算Z-scores
    z_scores = (data_batch - means) / stds
    print(z_scores)
    # 选择Z-score大于阈值的位置
    indices = torch.where(z_scores >= alpha)[1]
    print(indices)
    
    return indices

def cumulative_threshold_selection_v1(weights, threshold=0.8):
    """
    基于累积贡献度的CUDA高效批量选择

    Args:
        weights: torch.Tensor, shape [B, N], 每行和为1的重要性权重
        threshold: float, 累积贡献度阈值

    """
    device = weights.device
    B, N = weights.shape
    # 1. 沿N维度按重要性降序排序 - O(B*N*logN)
    sorted_weights, sorted_indices = torch.sort(weights, dim=1, descending=True)
    # print(sorted_indices.shape)
    # 2. 计算累积和 - O(B*N)
    cumulative_sums = torch.cumsum(sorted_weights, dim=1)
    # print(cumulative_sums.shape)
    # 3. 找到每个batch中第一个超过阈值的位置 - O(B*N)
    threshold_mask = cumulative_sums >= threshold
    # print(threshold_mask)
    # 4. 找到每行第一个True的位置 - O(B*N)
    # 使用argmax找到第一个True（值为1.0）的位置
    first_true_positions = torch.argmax(threshold_mask.float(), dim=1)
    # print(first_true_positions.shape)
    batch_range = torch.arange(B, device=device)
    weights_thresholds = sorted_weights[batch_range, first_true_positions]
    # print(weights_thresholds.shape)
    
    selected_indices = torch.where(weights >= weights_thresholds.unsqueeze(1))[1]
    # print(selected_indices.shape)
    # print(selected_indices)
    
    
    
    
    return selected_indices

def cumulative_threshold_selection(weights, threshold=0.8, max_sel_win_num=5):
    """
    基于累积贡献度的CUDA高效批量选择

    Args:
        weights: torch.Tensor, shape [B, N], 每行和为1的重要性权重
        threshold: float, 累积贡献度阈值

    """
    device = weights.device
    B, N = weights.shape

    # 排序
    sorted_weights, sorted_indices = torch.sort(weights, dim=1, descending=True)
    # 累积和
    cumsum = torch.cumsum(sorted_weights, dim=1)
    # 找到每行需要选择的元素数量
    # 第一个超过阈值的位置+1就是需要选择的数量
    exceed_mask = cumsum >= threshold
    
    # 处理边界情况：如果没有超过阈值，选择所有
    has_exceed = exceed_mask.any(dim=1)
    first_exceed_pos = torch.argmax(exceed_mask.float(), dim=1)
    select_counts = torch.where(has_exceed, first_exceed_pos + 1, N)
    # select_counts = min(select_counts, max_sel_win_num)
    # select_counts = torch.clamp(select_counts, max=max_sel_win_num)
    # print('select_counts:',select_counts)
    
    # 创建选择掩码
    positions = torch.arange(N, device=device).unsqueeze(0)
    selection_mask = positions < select_counts.unsqueeze(1)
    selected_indices= sorted_indices[selection_mask]
    
    return selected_indices
    


def dilate_mask(mask, kernel_size=5):
    """
    对bool类型的mask进行膨胀操作
    
    Args:
        mask: torch.Tensor, shape=[H,W], dtype=torch.bool
        kernel_size: int, 膨胀核大小（奇数）
    
    Returns:
        torch.Tensor, shape=[H,W], dtype=torch.bool
    """
    return F.max_pool2d(
        mask.float()[None, None],  # [H,W] -> [1,1,H,W] 并转为float
        kernel_size, 
        stride=1, 
        padding=kernel_size//2
    ).squeeze().bool()  # [1,1,H,W] -> [H,W] 并转回bool



class activation_hook:
    def __init__(self, module, out_act = True):
        if out_act:
            self.hook = module.register_forward_hook(self.hook_fn_output)
        else:
            self.hook = module.register_forward_hook(self.hook_fn_input)
            
    def hook_fn_output(self, module, input, output):
        self.feature = output
    def hook_fn_input(self, module, input, output):
        print('input')
        self.feature = input

    def remove(self):
        self.hook.remove()
        
class sam2_bypass(SAM2Base):
    def forward_image_bypass_train(self, img_batch: torch.Tensor):
        """Get the image feature on the input batch."""
        backbone_out, backbone_out_WBP = self.image_encoder.bypass_train_forward(img_batch)
        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )
            backbone_out_WBP["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out_WBP["backbone_fpn"][0]
            )
            backbone_out_WBP["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out_WBP["backbone_fpn"][1]
            )
        return backbone_out, backbone_out_WBP
    
    def track_step_bypass_train(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        current_vision_feats_WBP,
        current_vision_pos_embeds_WBP,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        # Whether to run the memory encoder on the predicted masks. Sometimes we might want
        # to skip the memory encoder with `run_mem_encoder=False`. For example,
        # in demo we might call `track_step` multiple times for each user click,
        # and only encode the memory when the user finalizes their clicks. And in ablation
        # settings like SAM training on static images, we don't need the memory encoder.
        run_mem_encoder=True,
        # The previously predicted SAM mask logits (which can be fed together with new clicks in demo).
        prev_sam_mask_logits=None,
    ):
        if self.enable_MeP_info==True:
            Pt_ATM_module = [
                    self.sam_mask_decoder.transformer.layers[0].cross_attn_token_to_image, 
                    self.sam_mask_decoder.transformer.layers[1].cross_attn_token_to_image, 
                    self.sam_mask_decoder.transformer.final_attn_token_to_image
                    ]
            Pt_ATM_hooks = [activation_hook(act, True) for act in Pt_ATM_module]

        current_out, sam_outputs, _, pix_feat = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )

        current_out_WBP, sam_outputs_WBP, _, pix_feat_WBP = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats_WBP,
            current_vision_pos_embeds_WBP,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )

        (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs
        
        if self.enable_MeP_info==True:
            Pt_ATMs = []
            for Pt_ATM_hook in Pt_ATM_hooks:
                Pt_ATMs.append(Pt_ATM_hook.feature[1][2].squeeze(0).mean(dim=(0,1)))
                
                Pt_ATM_hook.remove()
            Pt_ATMs = torch.stack(Pt_ATMs,dim=0).mean(0).reshape(1,64,64)

            self.mem_info['ious'].append(ious)
            self.mem_info['mask_region'].append(low_res_multimasks)
            self.mem_info['pt_sen_region'].append(Pt_ATMs)
            # print(object_score_logits)
            self.mem_info['obj_scores'].append(object_score_logits)

        # print(low_res_masks.shape)
        # exit()
        
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        if not self.training:
            # Only add this in inference (to avoid unused param in activation checkpointing;
            # it's mainly used in the demo to encode spatial memories w/ consolidated masks)
            current_out["object_score_logits"] = object_score_logits
        # current_out["object_score_logits"] = object_score_logits

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )

        return current_out, (pix_feat, pix_feat_WBP)

class SAM2VideoPredictor_bypass(sam2_bypass):
    """The predictor class to handle user interactions and manage inference states."""

    def __init__(
        self,
        fill_hole_area=0,
        # whether to apply non-overlapping constraints on the output object masks
        non_overlap_masks=False,
        # whether to clear non-conditioning memory of the surrounding frames (which may contain outdated information) after adding correction clicks;
        # note that this would only apply to *single-object tracking* unless `clear_non_cond_mem_for_multi_obj` is also set to True)
        clear_non_cond_mem_around_input=False,
        # if `add_all_frames_to_correct_as_cond` is True, we also append to the conditioning frame list any frame that receives a later correction click
        # if `add_all_frames_to_correct_as_cond` is False, we conditioning frame list to only use those initial conditioning frames
        add_all_frames_to_correct_as_cond=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fill_hole_area = fill_hole_area
        self.non_overlap_masks = non_overlap_masks
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond

    # @torch.inference_mode()
    def init_state(
        self,
        video_path,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    ):
        """Initialize an inference state."""
        compute_device = self.device  # device of the model
        images, video_height, video_width = load_video_frames(
            video_path=video_path,
            image_size=self.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
            compute_device=compute_device,
        )
        inference_state = {}
        inference_state["images"] = images
        inference_state["num_frames"] = len(images)
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = compute_device
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = compute_device
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        inference_state["cached_features_WBP"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["frames_tracked_per_obj"] = {}
        # Warm up the visual backbone and cache the image feature on frame 0
        self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state

    # def init_memory_info(self, enable_MeP_info=False):
    #     self.enable_MeP_info = enable_MeP_info
    #     self.mem_info = {}
    #     self.mem_info['window_sizes'] = self.image_encoder.trunk.window_spec
    #     self.mem_info['sel_win_id']=None
    #     self.image_encoder.trunk.mem_info = self.mem_info
    #     self.mem_info['ious'] = []
    #     self.mem_info['mask_region']=[]
    #     self.mem_info['pt_sen_region']=[]
    #     self.mem_info['obj_scores'] = []
    #     self.enable_mem_prune = False   
    
    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2VideoPredictor_bypass":
        """
        Load a pretrained model from the Hugging Face hub.

        Arguments:
          model_id (str): The Hugging Face repository ID.
          **kwargs: Additional arguments to pass to the model constructor.

        Returns:
          (SAM2VideoPredictor): The loaded model.
        """
        from sam2.build_sam import build_sam2_video_predictor_hf

        sam_model = build_sam2_video_predictor_hf(model_id, **kwargs)
        return sam_model

    def _obj_id_to_idx(self, inference_state, obj_id):
        """Map client-side object id to model-side object index."""
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx

        # We always allow adding new objects (including after tracking starts).
        allow_new_object = True
        if allow_new_object:
            # get the next object slot
            obj_idx = len(inference_state["obj_id_to_idx"])
            inference_state["obj_id_to_idx"][obj_id] = obj_idx
            inference_state["obj_idx_to_id"][obj_idx] = obj_id
            inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])
            # set up input and output structures for this object
            inference_state["point_inputs_per_obj"][obj_idx] = {}
            inference_state["mask_inputs_per_obj"][obj_idx] = {}
            inference_state["output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            inference_state["temp_output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            inference_state["frames_tracked_per_obj"][obj_idx] = {}
            return obj_idx
        else:
            raise RuntimeError(
                f"Cannot add new object id {obj_id} after tracking starts. "
                f"All existing object ids: {inference_state['obj_ids']}. "
                f"Please call 'reset_state' to restart from scratch."
            )

    def _obj_idx_to_id(self, inference_state, obj_idx):
        """Map model-side object index to client-side object id."""
        return inference_state["obj_idx_to_id"][obj_idx]

    def _get_obj_num(self, inference_state):
        """Get the total number of unique object ids received so far in this session."""
        return len(inference_state["obj_idx_to_id"])
    
    def window_partition_mask(self, x, window_size):
        """
        Partition into non-overlapping windows with padding if needed.
        Args:
            x (tensor): input tokens with [H, W].
            window_size (int): window size.
        Returns:
            windows: windows after partition with [B * num_windows, window_size, window_size, C].
            (Hp, Wp): padded height and width before partition
        """
        H, W = x.shape
        # x = x.reshape(1,H,W,1)
        # B, H, W, C = x.shape

        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w
        # 1,5,14,5,14,C
        x = x.view(Hp // window_size, window_size, Wp // window_size, window_size)
        # windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
        windows = x.permute(0, 2, 1, 3).reshape(-1, window_size, window_size)
        # B,H,W,C = windows.shape
        # print(windows.reshape(B,-1,C))
        # exit()
        return windows, (Hp, Wp)
    
    def select_mask_windows_1(self, pred_masks_per_obj):
        # print(pred_masks_per_obj)
        # exit()
        
        mask_put = torch.zeros((64,64),dtype=torch.uint8)
        for i, mask in enumerate(pred_masks_per_obj):
            mask_reshape=torch.nn.functional.interpolate(
                mask,
                size=(64, 64),
                mode="nearest",
                # align_corners=False,
            ).squeeze(0).squeeze(0)
            mask_put[mask_reshape>0] = 1
        window_size = self.mem_info['window_sizes'][-2] # 倒数第二层的窗口尺寸
        mask_win,(Hp,Wp) = self.window_partition_mask(mask_put, window_size)
        # B,h,w = mask_win.shape
        mask_win_sum = mask_win.sum(dim=(1,2))
        sel_win_id = torch.nonzero(mask_win_sum!=0).squeeze(-1)
        
        return sel_win_id
    
    def select_mask_windows(self, masks):
        # pred_masks = torch.cat(pred_masks_per_obj, dim=0)
        # pred_masks = pred_masks.squeeze(1)
        # masks = (pred_masks>0)
        # masks = torch.any(masks, dim=0)
        # print(masks.dtype)
            # 1. 动态填充至可整除尺寸（无填充时零拷贝）
        H,W = masks.shape
        window_size = self.mem_info['window_sizes'][-2]*4   # 特征图尺寸是64*64, mask尺寸是256*256，需要统一比例
        Hp = (H + window_size - 1) // window_size * window_size
        Wp = (W + window_size - 1) // window_size * window_size
        masks_padded = torch.nn.functional.pad(
            masks,  # 兼容非二值输入
            (0, Wp - W, 0, Hp - H),
            mode='constant',
            value=False
        )  # -> [B, Hp, Wp]
        m, n = masks_padded.shape[0]//window_size, window_size
        masks_padded = masks_padded.reshape(m,n,m,n).permute(0,2,1,3).reshape(m*m,n*n)
        obj_win_mask = torch.any(masks_padded, dim=1)
        sel_win_id = torch.nonzero(obj_win_mask).squeeze(-1)
        # print(sel_win_id.shape)
        # print(sel_win_id)
        
        return sel_win_id
    
    # @torch.jit.script
    def select_mask_windows_v3(self, masks):
        # pred_masks = torch.cat(pred_masks_per_obj, dim=0)
        # pred_masks = pred_masks.squeeze(1)
        # masks = (pred_masks>0)
        # masks = torch.any(masks, dim=0)
        # print(masks.dtype)
            # 1. 动态填充至可整除尺寸（无填充时零拷贝）
        if masks.sum() == 0:
            # print('zero mask')
            return torch.empty(0, dtype=torch.int64, device=masks.device)
        H,W = masks.shape
        window_size = self.mem_info['window_sizes'][-2]*4   # 特征图尺寸是64*64, mask尺寸是256*256，需要统一比例
        Hp = (H + window_size - 1) // window_size * window_size
        Wp = (W + window_size - 1) // window_size * window_size
        masks_padded = torch.nn.functional.pad(
            masks,  # 兼容非二值输入
            (0, Wp - W, 0, Hp - H),
            mode='constant',
            value=False
        )  # -> [B, Hp, Wp]
        m, n = masks_padded.shape[0]//window_size, window_size
        masks_padded = masks_padded.reshape(m,n,m,n).permute(0,2,1,3).reshape(m*m,n*n)
        obj_win_mask = torch.any(masks_padded, dim=1)
        sel_win_id = torch.nonzero(obj_win_mask).squeeze(-1)
        # print(sel_win_id.shape)
        # print(sel_win_id)
        # print(sel_win_id.dtype)
        
        return sel_win_id
        # return sel_win_id.cpu().tolist()
    
    # @torch.jit.script
    def select_mask_windows_v2(self, masks):
        """
        针对CUDA张量优化的窗口选择函数
        Args:
            masks: shape [H, W], dtype=torch.bool, device=cuda
        Returns:
            sel_win_id: 包含目标的窗口索引
        """
        # assert masks.is_cuda, "This implementation is optimized for CUDA tensors only"
        H, W = masks.shape  # 256, 256
        window_size = self.mem_info['window_sizes'][-2] * 4 # 56
        
        # 1. 快速计算网格参数
        num_windows_h = (H + window_size - 1) // window_size
        num_windows_w = (W + window_size - 1) // window_size
        
        # 2. 使用高效的kernel级并行化
        y_coords, x_coords = torch.nonzero(masks, as_tuple=True)
        if len(y_coords) == 0:
            return torch.empty(0, dtype=torch.int64, device=masks.device)
        
        # 3. 一次性计算所有坐标对应的窗口索引
        window_indices = (y_coords // window_size) * num_windows_w + (x_coords // window_size)
        
        # 4. 使用CUDA优化的unique操作
        # window_indices = torch.unique(window_indices, sorted=True)
        
        return window_indices
    
    def select_active_windows(
        self,
        pred_masks_per_obj: torch.Tensor,  # 输入掩码 [1, 1, N, N]
        window_size: int,                  # 窗口大小（需能被N整除）
    ) -> torch.Tensor:
        """
        基于二值掩码选择包含物体区域的非重叠窗口索引
        Returns:
            active_win_ids: 包含物体的窗口索引 [num_active_windows]
        """
        # 1. 检查输入合法性
        assert pred_masks_per_obj.dim() == 4 and pred_masks_per_obj.shape[0] == 1, "输入需为[1,1,N,N]"
        N = pred_masks_per_obj.shape[-1]
        assert N % window_size == 0, "N必须能被window_size整除"

        # 2. 直接下采样掩码到窗口粒度（避免插值）
        # 将N×N划分为 (N//w)×(N//w) 的窗口块，每块求和判断是否含物体
        mask = pred_masks_per_obj.squeeze()  # [N, N]
        windows = mask.view(N // window_size, window_size, N // window_size, window_size)
        window_sums = windows.sum(dim=(1, 3))  # [N//w, N//w]

        # 3. 提取非零窗口的线性索引
        active_win_ids = torch.nonzero(window_sums.flatten() > 0).squeeze(-1)
        return active_win_ids
    
    
    # @torch.inference_mode()
    def add_new_points_or_box(
        self,
        inference_state,
        frame_idx,
        obj_id,
        points=None,
        labels=None,
        clear_old_points=True,
        normalize_coords=True,
        box=None,
    ):
        """Add new points to a frame."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        if (points is not None) != (labels is not None):
            raise ValueError("points and labels must be provided together")
        if points is None and box is None:
            raise ValueError("at least one of points or box must be provided as input")

        if points is None:
            points = torch.zeros(0, 2, dtype=torch.float32)
        elif not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if labels is None:
            labels = torch.zeros(0, dtype=torch.int32)
        elif not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int32)
        if points.dim() == 2:
            points = points.unsqueeze(0)  # add batch dimension
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)  # add batch dimension

        # If `box` is provided, we add it as the first two points with labels 2 and 3
        # along with the user-provided points (consistent with how SAM 2 is trained).
        if box is not None:
            if not clear_old_points:
                raise ValueError(
                    "cannot add box without clearing old points, since "
                    "box prompt must be provided before any point prompt "
                    "(please use clear_old_points=True instead)"
                )
            if not isinstance(box, torch.Tensor):
                box = torch.tensor(box, dtype=torch.float32, device=points.device)
            box_coords = box.reshape(1, 2, 2)
            box_labels = torch.tensor([2, 3], dtype=torch.int32, device=labels.device)
            box_labels = box_labels.reshape(1, 2)
            points = torch.cat([box_coords, points], dim=1)
            labels = torch.cat([box_labels, labels], dim=1)

        if normalize_coords:
            video_H = inference_state["video_height"]
            video_W = inference_state["video_width"]
            points = points / torch.tensor([video_W, video_H]).to(points.device)
        # scale the (normalized) coordinates by the model's internal image size
        points = points * self.image_size
        points = points.to(inference_state["device"])
        labels = labels.to(inference_state["device"])

        if not clear_old_points:
            point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            point_inputs = None
        point_inputs = concat_points(point_inputs, points, labels)

        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)
        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        obj_frames_tracked = inference_state["frames_tracked_per_obj"][obj_idx]
        is_init_cond_frame = frame_idx not in obj_frames_tracked
        # whether to track in reverse time order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = obj_frames_tracked[frame_idx]["reverse"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        # Add a frame to conditioning output if it's an initial conditioning frame or
        # if the model sees all frames receiving clicks/mask as conditioning frames.
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # Get any previously predicted mask logits on this object and feed it along with
        # the new clicks into the SAM mask decoder.
        prev_sam_mask_logits = None
        # lookup temporary output dict first, which contains the most recent output
        # (if not found, then lookup conditioning and non-conditioning frame output)
        prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)

        if prev_out is not None and prev_out["pred_masks"] is not None:
            device = inference_state["device"]
            prev_sam_mask_logits = prev_out["pred_masks"].to(device, non_blocking=True)
            # Clamp the scale of prev_sam_mask_logits to avoid rare numerical issues.
            prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)
        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=reverse,
            # Skip the memory encoder when adding clicks or mask. We execute the memory encoder
            # at the beginning of `propagate_in_video` (after user finalize their clicks). This
            # allows us to enforce non-overlapping constraints on all objects before encoding
            # them into memory.
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        # Add the output to the output dict (to be used as future memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize the output mask to the original video resolution
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    def add_new_points(self, *args, **kwargs):
        """Deprecated method. Please use `add_new_points_or_box` instead."""
        return self.add_new_points_or_box(*args, **kwargs)

    # @torch.inference_mode()
    # @torch.enable_grad()
    def add_new_mask(
        self,
        inference_state,
        frame_idx,
        obj_id,
        mask,
    ):
        """Add new mask to a frame."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)
        assert mask.dim() == 2
        mask_H, mask_W = mask.shape
        mask_inputs_orig = mask[None, None]  # add batch and channel dimension
        mask_inputs_orig = mask_inputs_orig.float().to(inference_state["device"])

        # resize the mask if it doesn't match the model's image size
        if mask_H != self.image_size or mask_W != self.image_size:
            mask_inputs = torch.nn.functional.interpolate(
                mask_inputs_orig,
                size=(self.image_size, self.image_size),
                align_corners=False,
                mode="bilinear",
                antialias=True,  # use antialias for downsampling
            )
            mask_inputs = (mask_inputs >= 0.5).float()
        else:
            mask_inputs = mask_inputs_orig

        mask_inputs_per_frame[frame_idx] = mask_inputs
        point_inputs_per_frame.pop(frame_idx, None)
        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        obj_frames_tracked = inference_state["frames_tracked_per_obj"][obj_idx]
        is_init_cond_frame = frame_idx not in obj_frames_tracked
        # whether to track in reverse time order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = obj_frames_tracked[frame_idx]["reverse"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        # Add a frame to conditioning output if it's an initial conditioning frame or
        # if the model sees all frames receiving clicks/mask as conditioning frames.
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=mask_inputs,
            reverse=reverse,
            # Skip the memory encoder when adding clicks or mask. We execute the memory encoder
            # at the beginning of `propagate_in_video` (after user finalize their clicks). This
            # allows us to enforce non-overlapping constraints on all objects before encoding
            # them into memory.
            run_mem_encoder=False,
        )
        # Add the output to the output dict (to be used as future memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize the output mask to the original video resolution
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    def _get_orig_video_res_output(self, inference_state, any_res_masks):
        """
        Resize the object scores to the original video resolution (video_res_masks)
        and apply non-overlapping constraints for final output.
        """
        device = inference_state["device"]
        video_H = inference_state["video_height"]
        video_W = inference_state["video_width"]
        any_res_masks = any_res_masks.to(device, non_blocking=True)
        if any_res_masks.shape[-2:] == (video_H, video_W):
            video_res_masks = any_res_masks
        else:
            video_res_masks = torch.nn.functional.interpolate(
                any_res_masks,
                size=(video_H, video_W),
                mode="bilinear",
                align_corners=False,
            )
        if self.non_overlap_masks:
            video_res_masks = self._apply_non_overlapping_constraints(video_res_masks)
        return any_res_masks, video_res_masks

    def _consolidate_temp_output_across_obj(
        self,
        inference_state,
        frame_idx,
        is_cond,
        consolidate_at_video_res=False,
    ):
        """
        Consolidate the per-object temporary outputs in `temp_output_dict_per_obj` on
        a frame into a single output for all objects, including
        1) fill any missing objects either from `output_dict_per_obj` (if they exist in
           `output_dict_per_obj` for this frame) or leave them as placeholder values
           (if they don't exist in `output_dict_per_obj` for this frame);
        2) if specified, rerun memory encoder after apply non-overlapping constraints
           on the object scores.
        """
        batch_size = self._get_obj_num(inference_state)
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        # Optionally, we allow consolidating the temporary outputs at the original
        # video resolution (to provide a better editing experience for mask prompts).
        if consolidate_at_video_res:
            consolidated_H = inference_state["video_height"]
            consolidated_W = inference_state["video_width"]
            consolidated_mask_key = "pred_masks_video_res"
        else:
            consolidated_H = consolidated_W = self.image_size // 4
            consolidated_mask_key = "pred_masks"

        # Initialize `consolidated_out`. Its "maskmem_features" and "maskmem_pos_enc"
        # will be added when rerunning the memory encoder after applying non-overlapping
        # constraints to object scores. Its "pred_masks" are prefilled with a large
        # negative value (NO_OBJ_SCORE) to represent missing objects.
        consolidated_out = {
            consolidated_mask_key: torch.full(
                size=(batch_size, 1, consolidated_H, consolidated_W),
                fill_value=NO_OBJ_SCORE,
                dtype=torch.float32,
                device=inference_state["storage_device"],
            ),
        }
        for obj_idx in range(batch_size):
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            out = obj_temp_output_dict[storage_key].get(frame_idx, None)
            # If the object doesn't appear in "temp_output_dict_per_obj" on this frame,
            # we fall back and look up its previous output in "output_dict_per_obj".
            # We look up both "cond_frame_outputs" and "non_cond_frame_outputs" in
            # "output_dict_per_obj" to find a previous output for this object.
            if out is None:
                out = obj_output_dict["cond_frame_outputs"].get(frame_idx, None)
            if out is None:
                out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx, None)
            # If the object doesn't appear in "output_dict_per_obj" either, we skip it
            # and leave its mask scores to the default scores (i.e. the NO_OBJ_SCORE
            # placeholder above) and set its object pointer to be a dummy pointer.
            if out is None:
                continue
            # Add the temporary object output mask to consolidated output mask
            obj_mask = out["pred_masks"]
            consolidated_pred_masks = consolidated_out[consolidated_mask_key]
            if obj_mask.shape[-2:] == consolidated_pred_masks.shape[-2:]:
                consolidated_pred_masks[obj_idx : obj_idx + 1] = obj_mask
            else:
                # Resize first if temporary object mask has a different resolution
                resized_obj_mask = torch.nn.functional.interpolate(
                    obj_mask,
                    size=consolidated_pred_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                consolidated_pred_masks[obj_idx : obj_idx + 1] = resized_obj_mask

        return consolidated_out

    # @torch.inference_mode()
    def propagate_in_video_preflight(self, inference_state):
        """Prepare inference_state and consolidate temporary outputs before tracking."""
        # Check and make sure that every object has received input points or masks.
        batch_size = self._get_obj_num(inference_state)
        if batch_size == 0:
            raise RuntimeError(
                "No input points or masks are provided for any object; please add inputs first."
            )

        # Consolidate per-object temporary outputs in "temp_output_dict_per_obj" and
        # add them into "output_dict".
        for obj_idx in range(batch_size):
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            for is_cond in [False, True]:
                # Separately consolidate conditioning and non-conditioning temp outputs
                storage_key = (
                    "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
                )
                # Find all the frames that contain temporary outputs for any objects
                # (these should be the frames that have just received clicks for mask inputs
                # via `add_new_points_or_box` or `add_new_mask`)
                for frame_idx, out in obj_temp_output_dict[storage_key].items():
                    # Run memory encoder on the temporary outputs (if the memory feature is missing)
                    if out["maskmem_features"] is None:
                        high_res_masks = torch.nn.functional.interpolate(
                            out["pred_masks"].to(inference_state["device"]),
                            size=(self.image_size, self.image_size),
                            mode="bilinear",
                            align_corners=False,
                        )
                        maskmem_features, maskmem_pos_enc = self._run_memory_encoder(
                            inference_state=inference_state,
                            frame_idx=frame_idx,
                            batch_size=1,  # run on the slice of a single object
                            high_res_masks=high_res_masks,
                            object_score_logits=out["object_score_logits"],
                            # these frames are what the user interacted with
                            is_mask_from_pts=True,
                        )
                        out["maskmem_features"] = maskmem_features
                        out["maskmem_pos_enc"] = maskmem_pos_enc

                    obj_output_dict[storage_key][frame_idx] = out
                    if self.clear_non_cond_mem_around_input:
                        # clear non-conditioning memory of the surrounding frames
                        self._clear_obj_non_cond_mem_around_input(
                            inference_state, frame_idx, obj_idx
                        )

                # clear temporary outputs in `temp_output_dict_per_obj`
                obj_temp_output_dict[storage_key].clear()

            # check and make sure that every object has received input points or masks
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            if len(obj_output_dict["cond_frame_outputs"]) == 0:
                obj_id = self._obj_idx_to_id(inference_state, obj_idx)
                raise RuntimeError(
                    f"No input points or masks are provided for object id {obj_id}; please add inputs first."
                )
            # edge case: if an output is added to "cond_frame_outputs", we remove any prior
            # output on the same frame in "non_cond_frame_outputs"
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
                
    # @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        """Propagate the input points across frames to track in the entire video."""
        self.propagate_in_video_preflight(inference_state)
        # print(inference_state.keys())
        # print(inference_state['output_dict_per_obj'])
        # exit()

        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)

        # set start index, end index, and processing order
        if start_frame_idx is None:
            # default: start from the earliest frame with input points
            start_frame_idx = min(
                t
                for obj_output_dict in inference_state["output_dict_per_obj"].values()
                for t in obj_output_dict["cond_frame_outputs"]
            )
        if max_frame_num_to_track is None:
            # default: track all the frames in the video
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = []  # skip reverse tracking if starting from frame 0
        else:
            end_frame_idx = min(
                start_frame_idx + max_frame_num_to_track, num_frames - 1
            )
            processing_order = range(start_frame_idx, end_frame_idx + 1)
        
        for frame_idx in tqdm(processing_order, desc="propagate in video"):
            pred_masks_per_obj = [None] * batch_size
            for obj_idx in range(batch_size):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                # We skip those frames already in consolidated outputs (these are frames
                # that received input clicks or mask). Note that we cannot directly run
                # batched forward on them via `_run_single_frame_inference` because the
                # number of clicks on each object might be different.
                if frame_idx in obj_output_dict["cond_frame_outputs"]:
                    storage_key = "cond_frame_outputs"
                    current_out = obj_output_dict[storage_key][frame_idx]
                    device = inference_state["device"]
                    pred_masks = current_out["pred_masks"].to(device, non_blocking=True)
                    if self.clear_non_cond_mem_around_input:
                        # clear non-conditioning memory of the surrounding frames
                        self._clear_obj_non_cond_mem_around_input(
                            inference_state, frame_idx, obj_idx
                        )
                else:
                    storage_key = "non_cond_frame_outputs"
                    # if frame_idx not in [1]:
                    if frame_idx not in []:
                    # if frame_idx not in [1,2,3,4,5]:
                    
                        current_out, pred_masks = self._run_single_frame_inference(
                            inference_state=inference_state,
                            output_dict=obj_output_dict,
                            frame_idx=frame_idx,
                            batch_size=1,  # run on the slice of a single object
                            is_init_cond_frame=False,
                            point_inputs=None,
                            mask_inputs=None,
                            reverse=reverse,
                            run_mem_encoder=True,
                        )
                        
                    else:
                        # global logger
                        logger.info('*******attn_save')
                        save_pth_path = '/home/zhangjing/sam2_Proj/hahaha/attention_heat_WR/'
                        if not os.path.exists(save_pth_path):
                            os.mkdir(save_pth_path)
                        act_list = [
                                    # {'sel_module': self.memory_attention.layers[0].cross_attn_image, 
                                    # 'save_path':save_pth_path+'attn_vis/memory_cross_attn_qk_layer_0_{}_frame_{}_obj_{}.pth'.format('', frame_idx,obj_idx),
                                    # 'hook_output': True,
                                    # 'index':[1]},
                                    # {'sel_module': self.memory_attention.layers[1].cross_attn_image, 
                                    # 'save_path':save_pth_path+'attn_vis/memory_cross_attn_qk_layer_1_{}_frame_{}_obj_{}.pth'.format('', frame_idx,obj_idx),
                                    # 'hook_output': True,
                                    # 'index':[1]},
                                    # {'sel_module': self.memory_attention.layers[2].cross_attn_image, 
                                    # 'save_path':save_pth_path+'attn_vis/memory_cross_attn_qk_layer_2_{}_frame_{}_obj_{}.pth'.format('', frame_idx,obj_idx),
                                    # 'hook_output': True,
                                    # 'index':[1]},
                                    # {'sel_module': self.memory_attention.layers[3].cross_attn_image, 
                                    # 'save_path':save_pth_path+'attn_vis/memory_cross_attn_qk_layer_3_{}_frame_{}_obj_{}.pth'.format('', frame_idx,obj_idx),
                                    # 'hook_output': True,
                                    # 'index':[1]},
                                    # {'sel_module': self.image_encoder.trunk.blocks[6], 
                                    # 'save_path':save_pth_path+'sim_vis/sim_pth/similarity_blk6_{}.pth'.format(frame_idx),
                                    # 'hook_output': True,
                                    # 'index':[1]},
                                    {'sel_module': self.sam_mask_decoder.transformer.layers[0].cross_attn_token_to_image, 
                                    'save_path': save_pth_path+'attn_vis/prompt_cross_attn/CA_T2I_L0_{}_obj{}.pth'.format(frame_idx,obj_idx),
                                    'hook_output': True,
                                    'index':[1]},
                                    {'sel_module': self.sam_mask_decoder.transformer.layers[0].cross_attn_image_to_token, 
                                    'save_path': save_pth_path+'attn_vis/prompt_cross_attn/CA_I2T_L0_{}_obj{}.pth'.format(frame_idx,obj_idx),
                                    'hook_output': True,
                                    'index':[1]},
                        
                                    {'sel_module': self.sam_mask_decoder.transformer.layers[1].cross_attn_token_to_image, 
                                    'save_path': save_pth_path+'attn_vis/prompt_cross_attn/CA_T2I_L1_{}_obj{}.pth'.format(frame_idx,obj_idx),
                                    'hook_output': True,
                                    'index':[1]},
                                    {'sel_module': self.sam_mask_decoder.transformer.layers[1].cross_attn_image_to_token, 
                                    'save_path': save_pth_path+'attn_vis/prompt_cross_attn/CA_I2T_L1_{}_obj{}.pth'.format(frame_idx,obj_idx),
                                    'hook_output': True,
                                    'index':[1]},
                                    {'sel_module': self.sam_mask_decoder.transformer.final_attn_token_to_image, 
                                    'save_path': save_pth_path+'attn_vis/prompt_cross_attn/CA_FT2I_{}_obj{}.pth'.format(frame_idx,obj_idx),
                                    'hook_output': True,
                                    'index':[1]},
                                    ]
                        act_hooks = {'hooks':[], 'save_paths':[], 'index':[]}
                        for act in act_list:
                            act_hooks['hooks'].append(activation_hook(act['sel_module'], act['hook_output']))
                            act_hooks['save_paths'].append(act['save_path'])
                            act_hooks['index'].append(act['index'])
                        
                        current_out, pred_masks = self._run_single_frame_inference(
                            inference_state=inference_state,
                            output_dict=obj_output_dict,
                            frame_idx=frame_idx,
                            batch_size=1,  # run on the slice of a single object
                            is_init_cond_frame=False,
                            point_inputs=None,
                            mask_inputs=None,
                            reverse=reverse,
                            run_mem_encoder=True,
                        )
                        
                        for j in range(len(act_hooks['hooks'])):
                            # print(type(act_hooks['hooks'][i].feature))
                            try:
                                save_act = act_hooks['hooks'][j].feature
                                # if not isinstance(save_act, torch.Tensor):
                                #     for id in act_hooks['index'][i]:
                                #         save_act = save_act[id]
                                # print(save_act.shape)
                                
                                torch.save(save_act, act_hooks['save_paths'][j])
                                # print(type(save_act))
                                save_act = act_hooks['hooks'][j].remove()
                            except:
                                print('save error')
                                pass
                                
                        logger.info('*******attn_save')
                        
                    obj_output_dict[storage_key][frame_idx] = current_out

                inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {
                    "reverse": reverse
                }
                pred_masks_per_obj[obj_idx] = pred_masks
            # print(pred_masks.shape)
            # print(pred_masks.dtype)
            # print(pred_masks)
            # print(pred_masks_per_obj.shape)
            # torch.save(pred_masks_per_obj,'pred_masks_per_obj.pth')
            # exit()
            
            # Resize the output mask to the original video resolution (we directly use
            # the mask scores on GPU for output to avoid any CPU conversion in between)
            # self.select_mask_windows(pred_masks_per_obj)
            
            if len(pred_masks_per_obj) > 1:
                all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
            else:
                all_pred_masks = pred_masks_per_obj[0]
            _, video_res_masks = self._get_orig_video_res_output(
                inference_state, all_pred_masks
            )
            yield frame_idx, obj_ids, video_res_masks

    # @torch.enable_grad()
    def propagate_in_video_wj_1(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        """Propagate the input points across frames to track in the entire video."""
        self.propagate_in_video_preflight(inference_state)
        # print(inference_state.keys())
        # print(inference_state['output_dict_per_obj'])
        # exit()

        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)

        # set start index, end index, and processing order
        if start_frame_idx is None:
            # default: start from the earliest frame with input points
            start_frame_idx = min(
                t
                for obj_output_dict in inference_state["output_dict_per_obj"].values()
                for t in obj_output_dict["cond_frame_outputs"]
            )
        if max_frame_num_to_track is None:
            # default: track all the frames in the video
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = []  # skip reverse tracking if starting from frame 0
        else:
            end_frame_idx = min(
                start_frame_idx + max_frame_num_to_track, num_frames - 1
            )
            processing_order = range(start_frame_idx, end_frame_idx + 1)
        
        for frame_idx in tqdm(processing_order, desc="propagate in video"):
            pred_masks_per_obj = [None] * batch_size
            for obj_idx in range(batch_size):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                # We skip those frames already in consolidated outputs (these are frames
                # that received input clicks or mask). Note that we cannot directly run
                # batched forward on them via `_run_single_frame_inference` because the
                # number of clicks on each object might be different.
                if frame_idx in obj_output_dict["cond_frame_outputs"]:
                    storage_key = "cond_frame_outputs"
                    current_out = obj_output_dict[storage_key][frame_idx]
                    device = inference_state["device"]
                    pred_masks = current_out["pred_masks"].to(device, non_blocking=True)
                    if self.clear_non_cond_mem_around_input:
                        # clear non-conditioning memory of the surrounding frames
                        self._clear_obj_non_cond_mem_around_input(
                            inference_state, frame_idx, obj_idx
                        )
                else:
                    storage_key = "non_cond_frame_outputs"
                    
                    if frame_idx not in []:
                    # if frame_idx not in [5,10,20,50]:
                        current_out, pred_masks = self._run_single_frame_inference(
                            inference_state=inference_state,
                            output_dict=obj_output_dict,
                            frame_idx=frame_idx,
                            batch_size=1,  # run on the slice of a single object
                            is_init_cond_frame=False,
                            point_inputs=None,
                            mask_inputs=None,
                            reverse=reverse,
                            run_mem_encoder=True,
                        )
                        
                    else:
                        # global logger
                        logger.info('*******attn_save')
                        save_pth_path = '/home/zhangjing/sam2_Proj/hahaha/'
                        if not os.path.exists(save_pth_path):
                            os.mkdir(save_pth_path)
                        act_list = [
                                    {'sel_module': self.memory_attention.layers[0].cross_attn_image, 
                                    'save_path':save_pth_path+'attn_vis/memory_cross_attn_qk_layer_0_{}_frame_{}_obj_{}.pth'.format('', frame_idx,obj_idx),
                                    'hook_output': True,
                                    'index':[1]},
                                    {'sel_module': self.memory_attention.layers[1].cross_attn_image, 
                                    'save_path':save_pth_path+'attn_vis/memory_cross_attn_qk_layer_1_{}_frame_{}_obj_{}.pth'.format('', frame_idx,obj_idx),
                                    'hook_output': True,
                                    'index':[1]},
                                    {'sel_module': self.memory_attention.layers[2].cross_attn_image, 
                                    'save_path':save_pth_path+'attn_vis/memory_cross_attn_qk_layer_2_{}_frame_{}_obj_{}.pth'.format('', frame_idx,obj_idx),
                                    'hook_output': True,
                                    'index':[1]},
                                    {'sel_module': self.memory_attention.layers[3].cross_attn_image, 
                                    'save_path':save_pth_path+'attn_vis/memory_cross_attn_qk_layer_3_{}_frame_{}_obj_{}.pth'.format('', frame_idx,obj_idx),
                                    'hook_output': True,
                                    'index':[1]},
                                    {'sel_module': self.image_encoder.trunk.blocks[6], 
                                    'save_path':save_pth_path+'sim_vis/sim_pth/similarity_blk6_{}.pth'.format(frame_idx),
                                    'hook_output': True,
                                    'index':[1]},
                                    {'sel_module': self.sam_mask_decoder.transformer.layers[0].cross_attn_token_to_image, 
                                    'save_path': save_pth_path+'attn_vis/prompt_cross_attn/CA_T2I_L0_{}_obj{}.pth'.format(frame_idx,obj_idx),
                                    'hook_output': True,
                                    'index':[1]},
                                    {'sel_module': self.sam_mask_decoder.transformer.layers[0].cross_attn_image_to_token, 
                                    'save_path': save_pth_path+'attn_vis/prompt_cross_attn/CA_I2T_L0_{}_obj{}.pth'.format(frame_idx,obj_idx),
                                    'hook_output': True,
                                    'index':[1]},
                        
                                    {'sel_module': self.sam_mask_decoder.transformer.layers[1].cross_attn_token_to_image, 
                                    'save_path': save_pth_path+'attn_vis/prompt_cross_attn/CA_T2I_L1_{}_obj{}.pth'.format(frame_idx,obj_idx),
                                    'hook_output': True,
                                    'index':[1]},
                                    {'sel_module': self.sam_mask_decoder.transformer.layers[1].cross_attn_image_to_token, 
                                    'save_path': save_pth_path+'attn_vis/prompt_cross_attn/CA_I2T_L1_{}_obj{}.pth'.format(frame_idx,obj_idx),
                                    'hook_output': True,
                                    'index':[1]},
                                    {'sel_module': self.sam_mask_decoder.transformer.final_attn_token_to_image, 
                                    'save_path': save_pth_path+'attn_vis/prompt_cross_attn/CA_FT2I_{}_obj{}.pth'.format(frame_idx,obj_idx),
                                    'hook_output': True,
                                    'index':[1]},
                                    ]
                        act_hooks = {'hooks':[], 'save_paths':[], 'index':[]}
                        for act in act_list:
                            act_hooks['hooks'].append(activation_hook(act['sel_module'], act['hook_output']))
                            act_hooks['save_paths'].append(act['save_path'])
                            act_hooks['index'].append(act['index'])
                        
                        current_out, pred_masks = self._run_single_frame_inference(
                            inference_state=inference_state,
                            output_dict=obj_output_dict,
                            frame_idx=frame_idx,
                            batch_size=1,  # run on the slice of a single object
                            is_init_cond_frame=False,
                            point_inputs=None,
                            mask_inputs=None,
                            reverse=reverse,
                            run_mem_encoder=True,
                        )
                        
                        for j in range(len(act_hooks['hooks'])):
                            # print(type(act_hooks['hooks'][i].feature))
                            try:
                                save_act = act_hooks['hooks'][j].feature
                                # if not isinstance(save_act, torch.Tensor):
                                #     for id in act_hooks['index'][i]:
                                #         save_act = save_act[id]
                                # print(save_act.shape)
                                
                                torch.save(save_act, act_hooks['save_paths'][j])
                                # print(type(save_act))
                                save_act = act_hooks['hooks'][j].remove()
                            except:
                                print('save error')
                                pass
                                
                        logger.info('*******attn_save')
                        
                    obj_output_dict[storage_key][frame_idx] = current_out

                inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {
                    "reverse": reverse
                }
                pred_masks_per_obj[obj_idx] = pred_masks
            # print(pred_masks.shape)
            # print(pred_masks.dtype)
            # print(pred_masks)
            # print(pred_masks_per_obj.shape)
            # torch.save(pred_masks_per_obj,'pred_masks_per_obj.pth')
            # exit()
            # Resize the output mask to the original video resolution (we directly use
            # the mask scores on GPU for output to avoid any CPU conversion in between)
            
            
            
            
            # ------------------------------
            # 使用输出mask所在区域选择窗口
            # if frame_idx %1 == 0:
            #     pred_masks = torch.cat(pred_masks_per_obj, dim=0)
            #     masks = (pred_masks[:,0,:,:]>0)
            #     # masks = (pred_masks>0)
            #     if torch.sum(masks) != 0:
            #         masks = torch.any(masks, dim=0)
            #         self.mem_info['sel_win_id'] = self.select_mask_windows(masks)
            #     else:
            #         self.mem_info['sel_win_id'] = [0,2,4,10,20,22,24,14]
            #     print(self.mem_info)
            
            # coner_win_ids + obj_win_ids + rand_win_ids
            # coner_win_ids = [0,4,24,20]
            # pred_masks = torch.cat(pred_masks_per_obj, dim=0)
            # pred_masks = pred_masks.squeeze(1)
            # masks = (pred_masks>0)
            # masks = torch.any(masks, dim=0)
            # if torch.sum(masks) != 0:
            #     obj_win_ids = self.select_mask_windows(masks)
            #     rand_win_ids = []
            # else:
            #     obj_win_ids = []
            #     rand_win_ids = random.sample([0,1,2,3,4,9,14,19,24,23,22,21,20,15,10,5],5)
            # self.mem_info['sel_win_id'] = list(set(coner_win_ids+obj_win_ids+rand_win_ids))
            # # print(self.mem_info)
            
            # coner_win_ids + obj_win_ids + edge_win_id
            # coner_win_ids = [0,4,24,20]
            # edge_win_id = [0,1,2,3,4,9,14,19,24,23,22,21,20,15,10,5]
            # # if frame_idx %2 == 0:
            # pred_masks = torch.cat(pred_masks_per_obj, dim=0)
            # pred_masks = pred_masks.squeeze(1)
            # masks = (pred_masks>0)
            # masks = torch.any(masks, dim=0)
            # if torch.sum(masks) != 0:
            #     obj_win_ids = self.select_mask_windows(masks)
            #     # edge_win_id = []
            # else:
            #     obj_win_ids = []
            #     # edge_win_id = [0,1,2,3,4,9,14,19,24,23,22,21,20,15,10,5]
            #     edge_win_id = [i for i in range(25)]
            # self.mem_info['sel_win_id'] = list(set(coner_win_ids+obj_win_ids+edge_win_id))
            # print(self.mem_info)
            
            
            
            
            if len(pred_masks_per_obj) > 1:
                all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
            else:
                all_pred_masks = pred_masks_per_obj[0]
            _, video_res_masks = self._get_orig_video_res_output(
                inference_state, all_pred_masks
            )
            
            if frame_idx==end_frame_idx:
                self.mem_info['sel_win_id'] = None
                
            yield frame_idx, obj_ids, video_res_masks

    @torch.inference_mode()
    def propagate_in_video_wj(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        """Propagate the input points across frames to track in the entire video."""
        self.propagate_in_video_preflight(inference_state)
        # print(inference_state.keys())
        # print(inference_state['output_dict_per_obj'])
        # exit()
        
        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)

        # set start index, end index, and processing order
        if start_frame_idx is None:
            # default: start from the earliest frame with input points
            start_frame_idx = min(
                t
                for obj_output_dict in inference_state["output_dict_per_obj"].values()
                for t in obj_output_dict["cond_frame_outputs"]
            )
        if max_frame_num_to_track is None:
            # default: track all the frames in the video
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = []  # skip reverse tracking if starting from frame 0
        else:
            end_frame_idx = min(
                start_frame_idx + max_frame_num_to_track, num_frames - 1
            )
            processing_order = range(start_frame_idx, end_frame_idx + 1)
        # print('start_frame_idx:',start_frame_idx)
        
        for frame_idx in tqdm(processing_order, desc="propagate in video"):
            # if frame_idx == 50 :
            #     exit()
            self.time_log[frame_idx] = {'IE':[], 'Mem_attn': [], 'MD':[], 'Mem_E':[]}
            if frame_idx > start_frame_idx+1 and self.prune_memory:
                self.mem_info['enable_mem_prune'] = True
            
            pred_masks_per_obj = [None] * batch_size
            for obj_idx in range(batch_size):
                self.mem_info['obj_idx'] = obj_idx
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                # We skip those frames already in consolidated outputs (these are frames
                # that received input clicks or mask). Note that we cannot directly run
                # batched forward on them via `_run_single_frame_inference` because the
                # number of clicks on each object might be different.
                if frame_idx in obj_output_dict["cond_frame_outputs"]:
                    storage_key = "cond_frame_outputs"
                    current_out = obj_output_dict[storage_key][frame_idx]
                    device = inference_state["device"]
                    pred_masks = current_out["pred_masks"].to(device, non_blocking=True)
                    
                    if self.clear_non_cond_mem_around_input:
                        # clear non-conditioning memory of the surrounding frames
                        self._clear_obj_non_cond_mem_around_input(
                            inference_state, frame_idx, obj_idx
                        )
                else:
                    storage_key = "non_cond_frame_outputs"
                    
                    if frame_idx not in []:
                    # if frame_idx not in [1,3,5,10,15]:
                    # if frame_idx not in [5,20,100,200,247,300,350]:
                        
                    # if frame_idx not in [15+i*1 for i in range(15)]:
                        current_out, pred_masks = self._run_single_frame_inference(
                            inference_state=inference_state,
                            output_dict=obj_output_dict,
                            frame_idx=frame_idx,
                            batch_size=1,  # run on the slice of a single object
                            is_init_cond_frame=False,
                            point_inputs=None,
                            mask_inputs=None,
                            reverse=reverse,
                            run_mem_encoder=True,
                        )
                        
                    else:
                        # global logger
                        logger.info('*******attn_save')
                        # save_pth_path = '/home/zhangjing/sam2_Proj/hahaha/attention_heat_WR/attn_vis/trunk_WR/'
                        save_pth_path = '/home/zhangjing/sam2_Proj/hahaha/attention_heat_WR/attn_vis/prompt_cross_attn_WR/'
                        if not os.path.exists(save_pth_path):
                            os.mkdir(save_pth_path)
                        act_list = [
                                    # {'sel_module': self.memory_attention.layers[0].cross_attn_image, 
                                    # 'save_path':save_pth_path+'memory_cross_attn_qk_layer_0_{}_frame_{}_obj_{}.pth'.format('', frame_idx,obj_idx),
                                    # 'hook_output': True,
                                    # 'index':[1]},
                                    # {'sel_module': self.memory_attention.layers[1].cross_attn_image, 
                                    # 'save_path':save_pth_path+'memory_cross_attn_qk_layer_1_{}_frame_{}_obj_{}.pth'.format('', frame_idx,obj_idx),
                                    # 'hook_output': True,
                                    # 'index':[1]},
                                    # {'sel_module': self.memory_attention.layers[2].cross_attn_image, 
                                    # 'save_path':save_pth_path+'memory_cross_attn_qk_layer_2_{}_frame_{}_obj_{}.pth'.format('', frame_idx,obj_idx),
                                    # 'hook_output': True,
                                    # 'index':[1]},
                                    # {'sel_module': self.memory_attention.layers[3].cross_attn_image, 
                                    # 'save_path':save_pth_path+'memory_cross_attn_qk_layer_3_{}_frame_{}_obj_{}.pth'.format('', frame_idx,obj_idx),
                                    # 'hook_output': True,
                                    # 'index':[1]},
                                    # {'sel_module': self.image_encoder.trunk.blocks[20].attn, 
                                    # 'save_path':save_pth_path+'trunk_qk_blk20_f{}.pth'.format(frame_idx),
                                    # 'hook_output': True,
                                    # 'index':[1]},
                                    {'sel_module': self.sam_mask_decoder.transformer.layers[0].cross_attn_token_to_image, 
                                    'save_path': save_pth_path+'CA_T2I_L0_{}.pth'.format(frame_idx),
                                    # 'save_path': save_pth_path+'CA_T2I_L0_{}_obj{}.pth'.format(frame_idx,obj_idx),
                                    'hook_output': True,
                                    'index':[1]},
                                    # {'sel_module': self.sam_mask_decoder.transformer.layers[0].cross_attn_image_to_token, 
                                    # 'save_path': save_pth_path+'CA_I2T_L0_{}.pth'.format(frame_idx),
                                    # # 'save_path': save_pth_path+'CA_I2T_L0_{}_obj{}.pth'.format(frame_idx,obj_idx),
                                    # 'hook_output': True,
                                    # 'index':[1]},
                        
                                    {'sel_module': self.sam_mask_decoder.transformer.layers[1].cross_attn_token_to_image, 
                                    'save_path': save_pth_path+'CA_T2I_L1_{}.pth'.format(frame_idx),
                                    # 'save_path': save_pth_path+'CA_T2I_L1_{}_obj{}.pth'.format(frame_idx,obj_idx),
                                    'hook_output': True,
                                    'index':[1]},
                                    # {'sel_module': self.sam_mask_decoder.transformer.layers[1].cross_attn_image_to_token, 
                                    # 'save_path': save_pth_path+'attn_vis/prompt_cross_attn/CA_I2T_L1_{}_obj{}.pth'.format(frame_idx,obj_idx),
                                    # 'hook_output': True,
                                    # 'index':[1]},
                                    {'sel_module': self.sam_mask_decoder.transformer.final_attn_token_to_image, 
                                    'save_path': save_pth_path+'CA_FT2I_{}.pth'.format(frame_idx),
                                    # 'save_path': save_pth_path+'CA_FT2I_{}_obj{}.pth'.format(frame_idx,obj_idx),
                                    'hook_output': True,
                                    'index':[1]},
                                    ]
                        act_hooks = {'hooks':[], 'save_paths':[], 'index':[]}
                        for act in act_list:
                            act_hooks['hooks'].append(activation_hook(act['sel_module'], act['hook_output']))
                            act_hooks['save_paths'].append(act['save_path'])
                            act_hooks['index'].append(act['index'])
                        
                        current_out, pred_masks = self._run_single_frame_inference(
                            inference_state=inference_state,
                            output_dict=obj_output_dict,
                            frame_idx=frame_idx,
                            batch_size=1,  # run on the slice of a single object
                            is_init_cond_frame=False,
                            point_inputs=None,
                            mask_inputs=None,
                            reverse=reverse,
                            run_mem_encoder=True,
                        )
                        
                        for j in range(len(act_hooks['hooks'])):
                            # print(type(act_hooks['hooks'][i].feature))
                            # try:
                            save_act = act_hooks['hooks'][j].feature
                            # if not isinstance(save_act, torch.Tensor):
                            #     for id in act_hooks['index'][i]:
                            #         save_act = save_act[id]
                            # print(save_act.shape)
                            
                            torch.save(save_act, act_hooks['save_paths'][j])
                            # print(type(save_act))
                            save_act = act_hooks['hooks'][j].remove()
                            # except:
                            #     print('save error')
                            #     pass
                                
                        logger.info('*******attn_save')
                        
                    obj_output_dict[storage_key][frame_idx] = current_out

                inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {
                    "reverse": reverse
                }
                pred_masks_per_obj[obj_idx] = pred_masks
            # print(pred_masks.shape)
            # print(pred_masks.dtype)
            # print(pred_masks)
            # print(pred_masks_per_obj.shape)
            # torch.save(pred_masks_per_obj,'pred_masks_per_obj.pth')
            # exit()
            # Resize the output mask to the original video resolution (we directly use
            # the mask scores on GPU for output to avoid any CPU conversion in between)
            
            
            
            
            # ------------------------------
            # 使用输出mask所在区域选择窗口
            if frame_idx %1 == 0:
                if not self.disable_WB:
                # if True:
                    if self.sel_max_iou:
                        pred_masks = torch.cat(pred_masks_per_obj, dim=0)
                        self.mem_info['mask_region'] = []
                        # print(pred_masks.shape)
                        masks = (pred_masks[:,0,:,:]>0)
                    else:
                        pred_masks = torch.cat(self.mem_info['mask_region'], dim=0)
                        self.mem_info['mask_region'] = []
                        # print(pred_masks.shape)
                        
                        _,_,H,W = pred_masks.shape
                        masks = (pred_masks>0).view(-1,H,W)
                    # pred_masks = torch.cat(self.mem_info['mask_region'], dim=0)
                    # self.mem_info['mask_region'] = []
                    
                    # _,_,H,W = pred_masks.shape
                    # masks = (pred_masks>0).view(-1,H,W)
                    # print(Pt_ATMs)
                    # if torch.sum(masks) != 0 and current_out['object_score_logits'] > 3:
                    if self.mem_info['sel_win_id'] == None :
                        self.mem_info['sel_win_id'] = [i for i in range(25)] if self._WB_info['window_size'] == 14 else [i for i in range(16)]
                        print('restart full')
                        print('selected windows:    ',self.mem_info['sel_win_id'])
                    else:
                        masks = torch.any(masks, dim=0)
                        if self.dilate_mask:
                            masks = dilate_mask(masks, self.dilate_kernel_size)
                        mask_win_id = self.select_mask_windows_v2(masks)
                        # mask_win_id_ = self.select_mask_windows_v3(masks)
                        # assert torch.equal(mask_win_id, mask_win_id_)
                        # assert torch.equal(torch.unique(mask_win_id), torch.unique(mask_win_id_))
                        # print(self.mem_info['obj_scores'])
                        # print(self.mem_info['best_iou'])
                        # print(self.obj_theta)
                        if mask_win_id.shape[0] == 0 or any(x < self.obj_theta for x in self.mem_info['obj_scores']):
                            Pt_ATMs = torch.stack(self.mem_info['pt_sen_region'], dim=0)
                            self.mem_info['pt_sen_region'] = []
                            
                            if self._WB_info['window_size'] == 14:
                                Pt_ATMs = F.pad(Pt_ATMs, (0, 6, 0, 6), "constant", 0).squeeze(1)
                            else:
                                Pt_ATMs = Pt_ATMs
                            B = Pt_ATMs.shape[0]
                            if self._WB_info['window_size'] == 14:
                                Pt_ATMs = Pt_ATMs.reshape(B,5,14,5,14).permute(0,1,3,2,4).reshape(B,25,-1).sum(-1)
                            else:
                                Pt_ATMs = Pt_ATMs.reshape(B,4,16,4,16).permute(0,1,3,2,4).reshape(B,16,-1).sum(-1)

                            # PF_win_id = adaptive_significant_selection(Pt_ATMs, 1)
                            PF_win_id = cumulative_threshold_selection(Pt_ATMs, threshold=self.WB_theta)
                            # PF_win_id = torch.where(Pt_ATMs > 0.1)[1]
                            # print(PF_win_id)
                            # print(self.mem_info['obj_scores'])
                        else:
                            PF_win_id = torch.tensor([], device=device, dtype=torch.int32)
                            # PF_win_id = torch.where(Pt_ATMs > 0.3)[1]
                            # self.mem_info['sel_win_id'] = self.select_mask_windows(masks)
                        # self.mem_info['sel_win_id'] = -1
                        
                        self.mem_info['sel_win_id'] = torch.unique(torch.cat([PF_win_id, mask_win_id]), sorted=True)
                        # self.mem_info['obj_scores'] = []
                        # self.mem_info['sel_win_id'] = mask_win_id
                        if self.print_WS or frame_idx % 50 ==0:
                            print('selected windows:    ',self.mem_info['sel_win_id'])
                            print('Prompt Focus windows:',torch.unique(PF_win_id))
                            # print('**** Mask windows ***',mask_win_id)
                            print(torch.unique(mask_win_id))
                        self.WS_log['Sel'].append(self.mem_info['sel_win_id'])
                        self.WS_log['Mask'].append(torch.unique(mask_win_id))
                        self.WS_log['PF'].append(torch.unique(PF_win_id))
                        
                self.mem_info['obj_scores'] = []
                self.mem_info['best_iou'] = []
                
                mem_masks_stack_obj = self.mem_info['mem_masks_stack'].get(obj_idx,None)
                if mem_masks_stack_obj != None:
                    self.mem_info['mem_masks'][obj_idx] = []
                    if len(mem_masks_stack_obj) == 5:
                        for i in range(4):
                            mem_masks_obj_Li = [mask[i] for mask in mem_masks_stack_obj]
                            self.mem_info['mem_masks'][obj_idx].append(torch.cat(mem_masks_obj_Li,dim=0))
            
            if len(pred_masks_per_obj) > 1:
                all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
            else:
                all_pred_masks = pred_masks_per_obj[0]
            _, video_res_masks = self._get_orig_video_res_output(
                inference_state, all_pred_masks
            )
            
            if frame_idx==end_frame_idx:
                self.mem_info['sel_win_id'] = None
                self.mem_info['ious'] = []
                self.mem_info['obj_scores'] = []
                self.mem_info['mask_region'] = []
                self.mem_info['pt_sen_region'] = []
                self.mem_info['mem_masks'] = {}
                self.mem_info['mem_masks_stack'] = {}
                self.mem_info['mem_merge_ids'] = {}
                self.mem_info['mem_merge_ids_stack'] = {}
                self.mem_info['enable_mem_prune'] = False
                self.mem_info['frame_delete_indices']=[]
                self.mem_info['obj_scores'] = []
                self.mem_info['best_iou'] = []
                
                
            yield frame_idx, obj_ids, video_res_masks

    
    # @torch.enable_grad()
    def propagate_in_video_wj_2(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        # 这个是要用的
        
        """Propagate the input points across frames to track in the entire video."""
        self.propagate_in_video_preflight(inference_state)
        # print(inference_state.keys())
        # print(inference_state['output_dict_per_obj'])
        # exit()

        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)

        # set start index, end index, and processing order
        if start_frame_idx is None:
            # default: start from the earliest frame with input points
            start_frame_idx = min(
                t
                for obj_output_dict in inference_state["output_dict_per_obj"].values()
                for t in obj_output_dict["cond_frame_outputs"]
            )
        if max_frame_num_to_track is None:
            # default: track all the frames in the video
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = []  # skip reverse tracking if starting from frame 0
        else:
            end_frame_idx = min(
                start_frame_idx + max_frame_num_to_track, num_frames - 1
            )
            processing_order = range(start_frame_idx, end_frame_idx + 1)
        
        for frame_idx in tqdm(processing_order, desc="propagate in video"):
            pred_masks_per_obj = [None] * batch_size
            for obj_idx in range(batch_size):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                # We skip those frames already in consolidated outputs (these are frames
                # that received input clicks or mask). Note that we cannot directly run
                # batched forward on them via `_run_single_frame_inference` because the
                # number of clicks on each object might be different.
                if frame_idx in obj_output_dict["cond_frame_outputs"]:
                    storage_key = "cond_frame_outputs"
                    current_out = obj_output_dict[storage_key][frame_idx]
                    device = inference_state["device"]
                    pred_masks = current_out["pred_masks"].to(device, non_blocking=True)
                    if self.clear_non_cond_mem_around_input:
                        # clear non-conditioning memory of the surrounding frames
                        self._clear_obj_non_cond_mem_around_input(
                            inference_state, frame_idx, obj_idx
                        )
                else:
                    storage_key = "non_cond_frame_outputs"
                    
                    if frame_idx not in []:
                    # if frame_idx not in [5,10,20,50]:
                        current_out, pred_masks = self._run_single_frame_inference(
                            inference_state=inference_state,
                            output_dict=obj_output_dict,
                            frame_idx=frame_idx,
                            batch_size=1,  # run on the slice of a single object
                            is_init_cond_frame=False,
                            point_inputs=None,
                            mask_inputs=None,
                            reverse=reverse,
                            run_mem_encoder=True,
                        )
                        
                    else:
                        # global logger
                        logger.info('*******attn_save')
                        save_pth_path = '/home/zhangjing/sam2_Proj/hahaha/'
                        if not os.path.exists(save_pth_path):
                            os.mkdir(save_pth_path)
                        act_list = [
                                    {'sel_module': self.memory_attention.layers[0].cross_attn_image, 
                                    'save_path':save_pth_path+'attn_vis/memory_cross_attn_qk_layer_0_{}_frame_{}_obj_{}.pth'.format('', frame_idx,obj_idx),
                                    'hook_output': True,
                                    'index':[1]},
                                    {'sel_module': self.memory_attention.layers[1].cross_attn_image, 
                                    'save_path':save_pth_path+'attn_vis/memory_cross_attn_qk_layer_1_{}_frame_{}_obj_{}.pth'.format('', frame_idx,obj_idx),
                                    'hook_output': True,
                                    'index':[1]},
                                    {'sel_module': self.memory_attention.layers[2].cross_attn_image, 
                                    'save_path':save_pth_path+'attn_vis/memory_cross_attn_qk_layer_2_{}_frame_{}_obj_{}.pth'.format('', frame_idx,obj_idx),
                                    'hook_output': True,
                                    'index':[1]},
                                    {'sel_module': self.memory_attention.layers[3].cross_attn_image, 
                                    'save_path':save_pth_path+'attn_vis/memory_cross_attn_qk_layer_3_{}_frame_{}_obj_{}.pth'.format('', frame_idx,obj_idx),
                                    'hook_output': True,
                                    'index':[1]},
                                    {'sel_module': self.image_encoder.trunk.blocks[6], 
                                    'save_path':save_pth_path+'sim_vis/sim_pth/similarity_blk6_{}.pth'.format(frame_idx),
                                    'hook_output': True,
                                    'index':[1]},
                                    {'sel_module': self.sam_mask_decoder.transformer.layers[0].cross_attn_token_to_image, 
                                    'save_path': save_pth_path+'attn_vis/prompt_cross_attn/CA_T2I_L0_{}_obj{}.pth'.format(frame_idx,obj_idx),
                                    'hook_output': True,
                                    'index':[1]},
                                    {'sel_module': self.sam_mask_decoder.transformer.layers[0].cross_attn_image_to_token, 
                                    'save_path': save_pth_path+'attn_vis/prompt_cross_attn/CA_I2T_L0_{}_obj{}.pth'.format(frame_idx,obj_idx),
                                    'hook_output': True,
                                    'index':[1]},
                        
                                    {'sel_module': self.sam_mask_decoder.transformer.layers[1].cross_attn_token_to_image, 
                                    'save_path': save_pth_path+'attn_vis/prompt_cross_attn/CA_T2I_L1_{}_obj{}.pth'.format(frame_idx,obj_idx),
                                    'hook_output': True,
                                    'index':[1]},
                                    {'sel_module': self.sam_mask_decoder.transformer.layers[1].cross_attn_image_to_token, 
                                    'save_path': save_pth_path+'attn_vis/prompt_cross_attn/CA_I2T_L1_{}_obj{}.pth'.format(frame_idx,obj_idx),
                                    'hook_output': True,
                                    'index':[1]},
                                    {'sel_module': self.sam_mask_decoder.transformer.final_attn_token_to_image, 
                                    'save_path': save_pth_path+'attn_vis/prompt_cross_attn/CA_FT2I_{}_obj{}.pth'.format(frame_idx,obj_idx),
                                    'hook_output': True,
                                    'index':[1]},
                                    ]
                        act_hooks = {'hooks':[], 'save_paths':[], 'index':[]}
                        for act in act_list:
                            act_hooks['hooks'].append(activation_hook(act['sel_module'], act['hook_output']))
                            act_hooks['save_paths'].append(act['save_path'])
                            act_hooks['index'].append(act['index'])
                        
                        current_out, pred_masks = self._run_single_frame_inference(
                            inference_state=inference_state,
                            output_dict=obj_output_dict,
                            frame_idx=frame_idx,
                            batch_size=1,  # run on the slice of a single object
                            is_init_cond_frame=False,
                            point_inputs=None,
                            mask_inputs=None,
                            reverse=reverse,
                            run_mem_encoder=True,
                        )
                        
                        for j in range(len(act_hooks['hooks'])):
                            # print(type(act_hooks['hooks'][i].feature))
                            try:
                                save_act = act_hooks['hooks'][j].feature
                                # if not isinstance(save_act, torch.Tensor):
                                #     for id in act_hooks['index'][i]:
                                #         save_act = save_act[id]
                                # print(save_act.shape)
                                
                                torch.save(save_act, act_hooks['save_paths'][j])
                                # print(type(save_act))
                                save_act = act_hooks['hooks'][j].remove()
                            except:
                                print('save error')
                                pass
                                
                        logger.info('*******attn_save')
                        
                    obj_output_dict[storage_key][frame_idx] = current_out

                inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {
                    "reverse": reverse
                }
                pred_masks_per_obj[obj_idx] = pred_masks
            # print(pred_masks.shape)
            # print(pred_masks.dtype)
            # print(pred_masks)
            # print(pred_masks_per_obj.shape)
            # torch.save(pred_masks_per_obj,'pred_masks_per_obj.pth')
            # exit()
            # Resize the output mask to the original video resolution (we directly use
            # the mask scores on GPU for output to avoid any CPU conversion in between)
            
            
            
            
            # ------------------------------
            # 使用输出mask所在区域选择窗口
            if frame_idx %1 == 0:
                pred_masks = torch.cat(pred_masks_per_obj, dim=0)
                masks = (pred_masks[:,0,:,:]>0)
                # masks = (pred_masks>0)
                if torch.sum(masks) != 0:
                    if self.mem_info['sel_win_id'] == None:
                        self.mem_info['sel_win_id'] = [i for i in range(25)]
                        print('restart full')
                    else:
                        masks = torch.any(masks, dim=0)
                        self.mem_info['sel_win_id'] = self.select_mask_windows(masks)
                else:
                    # self.mem_info['sel_win_id'] = [i for i in range(25)]
                    self.mem_info['sel_win_id'] = None
                # if frame_idx==246:
                #     print('frame_246',self.mem_info['sel_win_id'])
                #     self.mem_info['sel_win_id'] = [i for i in range(25)]
                print(self.mem_info['sel_win_id'])
            
                # self.mem_info['sel_win_id'] = [i for i in range(25)]
            
            # coner_win_ids + obj_win_ids + rand_win_ids
            # coner_win_ids = [0,4,24,20]
            # pred_masks = torch.cat(pred_masks_per_obj, dim=0)
            # pred_masks = pred_masks.squeeze(1)
            # masks = (pred_masks>0)
            # masks = torch.any(masks, dim=0)
            # if torch.sum(masks) != 0:
            #     obj_win_ids = self.select_mask_windows(masks)
            #     rand_win_ids = []
            # else:
            #     obj_win_ids = []
            #     rand_win_ids = random.sample([0,1,2,3,4,9,14,19,24,23,22,21,20,15,10,5],5)
            # self.mem_info['sel_win_id'] = list(set(coner_win_ids+obj_win_ids+rand_win_ids))
            # # print(self.mem_info)
            
            # coner_win_ids + obj_win_ids + edge_win_id
            # coner_win_ids = [0,4,24,20]
            # edge_win_id = [0,1,2,3,4,9,14,19,24,23,22,21,20,15,10,5]
            # # if frame_idx %2 == 0:
            # pred_masks = torch.cat(pred_masks_per_obj, dim=0)
            # pred_masks = pred_masks.squeeze(1)
            # masks = (pred_masks>0)
            # masks = torch.any(masks, dim=0)
            # if torch.sum(masks) != 0:
            #     obj_win_ids = self.select_mask_windows(masks)
            #     # edge_win_id = []
            # else:
            #     obj_win_ids = []
            #     # edge_win_id = [0,1,2,3,4,9,14,19,24,23,22,21,20,15,10,5]
            #     edge_win_id = [i for i in range(25)]
            # self.mem_info['sel_win_id'] = list(set(coner_win_ids+obj_win_ids+edge_win_id))
            # print(self.mem_info)
            
            
            
            
            if len(pred_masks_per_obj) > 1:
                all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
            else:
                all_pred_masks = pred_masks_per_obj[0]
            _, video_res_masks = self._get_orig_video_res_output(
                inference_state, all_pred_masks
            )
            
            if frame_idx==end_frame_idx:
                self.mem_info['sel_win_id'] = None
                torch.save(self.mem_info['ious'], '/home/zhangjing/sam2_Proj/hahaha/score_vis/ious/1_021093_iou_sel_2.pth')
                self.mem_info['ious'] = []
                
            yield frame_idx, obj_ids, video_res_masks

    # @torch.inference_mode()
    def clear_all_prompts_in_frame(
        self, inference_state, frame_idx, obj_id, need_output=True
    ):
        """Remove all input points or mask in a specific frame for a given object."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)

        # Clear the conditioning information on the given frame
        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        inference_state["mask_inputs_per_obj"][obj_idx].pop(frame_idx, None)

        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        temp_output_dict_per_obj[obj_idx]["cond_frame_outputs"].pop(frame_idx, None)
        temp_output_dict_per_obj[obj_idx]["non_cond_frame_outputs"].pop(frame_idx, None)

        # Remove the frame's conditioning output (possibly downgrading it to non-conditioning)
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        out = obj_output_dict["cond_frame_outputs"].pop(frame_idx, None)
        if out is not None:
            # The frame is not a conditioning frame anymore since it's not receiving inputs,
            # so we "downgrade" its output (if exists) to a non-conditioning frame output.
            obj_output_dict["non_cond_frame_outputs"][frame_idx] = out
            inference_state["frames_tracked_per_obj"][obj_idx].pop(frame_idx, None)

        if not need_output:
            return
        # Finally, output updated masks per object (after removing the inputs above)
        obj_ids = inference_state["obj_ids"]
        is_cond = any(
            frame_idx in obj_temp_output_dict["cond_frame_outputs"]
            for obj_temp_output_dict in temp_output_dict_per_obj.values()
        )
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    # @torch.inference_mode()
    def reset_state(self, inference_state):
        """Remove all input points or mask in all frames throughout the video."""
        self._reset_tracking_results(inference_state)
        # Remove all object ids
        inference_state["obj_id_to_idx"].clear()
        inference_state["obj_idx_to_id"].clear()
        inference_state["obj_ids"].clear()
        inference_state["point_inputs_per_obj"].clear()
        inference_state["mask_inputs_per_obj"].clear()
        inference_state["output_dict_per_obj"].clear()
        inference_state["temp_output_dict_per_obj"].clear()
        inference_state["frames_tracked_per_obj"].clear()

    def _reset_tracking_results(self, inference_state):
        """Reset all tracking inputs and results across the videos."""
        for v in inference_state["point_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["mask_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in inference_state["temp_output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in inference_state["frames_tracked_per_obj"].values():
            v.clear()

    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        """Compute the image features on a given frame."""
        # Look up in the cache first
        image, backbone_out = inference_state["cached_features"].get(
            frame_idx, (None, None)
        )
        if backbone_out is None:
            # Cache miss -- we will run inference on a single image
            device = inference_state["device"]
            image = inference_state["images"][frame_idx].to(device).float().unsqueeze(0)
            backbone_out = self.forward_image(image)
            # Cache the most recent frame's feature (for repeated interactions with
            # a frame; we can use an LRU cache for more frames in the future).
            inference_state["cached_features"] = {frame_idx: (image, backbone_out)}

        # expand the features to have the same dimension as the number of objects
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(
                batch_size, -1, -1, -1
            )
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = pos.expand(batch_size, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos

        features = self._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        return features

    def _run_single_frame_inference(
        self,
        inference_state,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
    ):
        """Run tracking on a single frame based on current inputs and previous memory."""
        # Retrieve correct image features
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size)
        # print(len(current_vision_feats))
        # print(current_vision_feats[-1].shape)
        # exit()
        # point and mask should not appear as input simultaneously on the same frame
        assert point_inputs is None or mask_inputs is None
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        # print(current_out["pred_masks"].shape)
        # exit()

        # optionally offload the output to CPU memory to save GPU space
        storage_device = inference_state["storage_device"]
        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        pred_masks_gpu = current_out["pred_masks"]
        # potentially fill holes in the predicted masks
        if self.fill_hole_area > 0:
            pred_masks_gpu = fill_holes_in_mask_scores(
                pred_masks_gpu, self.fill_hole_area
            )
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
        # object pointer is a small tensor, so we always keep it on GPU memory for fast access
        obj_ptr = current_out["obj_ptr"]
        object_score_logits = current_out["object_score_logits"]
        best_iou = current_out["best_iou"]
        # make a compact version of this frame's output to reduce the state size
        compact_current_out = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
            "best_iou": best_iou, 
        }
        return compact_current_out, pred_masks_gpu

    def _run_memory_encoder(
        self,
        inference_state,
        frame_idx,
        batch_size,
        high_res_masks,
        object_score_logits,
        is_mask_from_pts,
    ):
        """
        Run the memory encoder on `high_res_masks`. This is usually after applying
        non-overlapping constraints to object scores. Since their scores changed, their
        memory also need to be computed again with the memory encoder.
        """
        # Retrieve correct image features
        _, _, current_vision_feats, _, feat_sizes = self._get_image_feature(
            inference_state, frame_idx, batch_size
        )
        maskmem_features, maskmem_pos_enc = self._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=is_mask_from_pts,
        )

        # optionally offload the output to CPU memory to save GPU space
        storage_device = inference_state["storage_device"]
        maskmem_features = maskmem_features.to(torch.bfloat16)
        maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(
            inference_state, {"maskmem_pos_enc": maskmem_pos_enc}
        )
        return maskmem_features, maskmem_pos_enc

    def _get_maskmem_pos_enc(self, inference_state, current_out):
        """
        `maskmem_pos_enc` is the same across frames and objects, so we cache it as
        a constant in the inference session to reduce session storage size.
        """
        model_constants = inference_state["constants"]
        # "out_maskmem_pos_enc" should be either a list of tensors or None
        out_maskmem_pos_enc = current_out["maskmem_pos_enc"]
        if out_maskmem_pos_enc is not None:
            if "maskmem_pos_enc" not in model_constants:
                assert isinstance(out_maskmem_pos_enc, list)
                # only take the slice for one object, since it's same across objects
                maskmem_pos_enc = [x[0:1].clone() for x in out_maskmem_pos_enc]
                model_constants["maskmem_pos_enc"] = maskmem_pos_enc
            else:
                maskmem_pos_enc = model_constants["maskmem_pos_enc"]
            # expand the cached maskmem_pos_enc to the actual batch size
            batch_size = out_maskmem_pos_enc[0].size(0)
            expanded_maskmem_pos_enc = [
                x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc
            ]
        else:
            expanded_maskmem_pos_enc = None
        return expanded_maskmem_pos_enc

    # @torch.inference_mode()
    def remove_object(self, inference_state, obj_id, strict=False, need_output=True):
        """
        Remove an object id from the tracking state. If strict is True, we check whether
        the object id actually exists and raise an error if it doesn't exist.
        """
        old_obj_idx_to_rm = inference_state["obj_id_to_idx"].get(obj_id, None)
        updated_frames = []
        # Check whether this object_id to remove actually exists and possibly raise an error.
        if old_obj_idx_to_rm is None:
            if not strict:
                return inference_state["obj_ids"], updated_frames
            raise RuntimeError(
                f"Cannot remove object id {obj_id} as it doesn't exist. "
                f"All existing object ids: {inference_state['obj_ids']}."
            )

        # If this is the only remaining object id, we simply reset the state.
        if len(inference_state["obj_id_to_idx"]) == 1:
            self.reset_state(inference_state)
            return inference_state["obj_ids"], updated_frames

        # There are still remaining objects after removing this object id. In this case,
        # we need to delete the object storage from inference state tensors.
        # Step 0: clear the input on those frames where this object id has point or mask input
        # (note that this step is required as it might downgrade conditioning frames to
        # non-conditioning ones)
        obj_input_frames_inds = set()
        obj_input_frames_inds.update(
            inference_state["point_inputs_per_obj"][old_obj_idx_to_rm]
        )
        obj_input_frames_inds.update(
            inference_state["mask_inputs_per_obj"][old_obj_idx_to_rm]
        )
        for frame_idx in obj_input_frames_inds:
            self.clear_all_prompts_in_frame(
                inference_state, frame_idx, obj_id, need_output=False
            )

        # Step 1: Update the object id mapping (note that it must be done after Step 0,
        # since Step 0 still requires the old object id mappings in inference_state)
        old_obj_ids = inference_state["obj_ids"]
        old_obj_inds = list(range(len(old_obj_ids)))
        remain_old_obj_inds = old_obj_inds.copy()
        remain_old_obj_inds.remove(old_obj_idx_to_rm)
        new_obj_ids = [old_obj_ids[old_idx] for old_idx in remain_old_obj_inds]
        new_obj_inds = list(range(len(new_obj_ids)))
        # build new mappings
        old_idx_to_new_idx = dict(zip(remain_old_obj_inds, new_obj_inds))
        inference_state["obj_id_to_idx"] = dict(zip(new_obj_ids, new_obj_inds))
        inference_state["obj_idx_to_id"] = dict(zip(new_obj_inds, new_obj_ids))
        inference_state["obj_ids"] = new_obj_ids

        # Step 2: For per-object tensor storage, we shift their obj_idx in the dict keys.
        def _map_keys(container):
            new_kvs = []
            for k in old_obj_inds:
                v = container.pop(k)
                if k in old_idx_to_new_idx:
                    new_kvs.append((old_idx_to_new_idx[k], v))
            container.update(new_kvs)

        _map_keys(inference_state["point_inputs_per_obj"])
        _map_keys(inference_state["mask_inputs_per_obj"])
        _map_keys(inference_state["output_dict_per_obj"])
        _map_keys(inference_state["temp_output_dict_per_obj"])
        _map_keys(inference_state["frames_tracked_per_obj"])

        # Step 3: Further collect the outputs on those frames in `obj_input_frames_inds`, which
        # could show an updated mask for objects previously occluded by the object being removed
        if need_output:
            temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
            for frame_idx in obj_input_frames_inds:
                is_cond = any(
                    frame_idx in obj_temp_output_dict["cond_frame_outputs"]
                    for obj_temp_output_dict in temp_output_dict_per_obj.values()
                )
                consolidated_out = self._consolidate_temp_output_across_obj(
                    inference_state,
                    frame_idx,
                    is_cond=is_cond,
                    consolidate_at_video_res=True,
                )
                _, video_res_masks = self._get_orig_video_res_output(
                    inference_state, consolidated_out["pred_masks_video_res"]
                )
                updated_frames.append((frame_idx, video_res_masks))

        return inference_state["obj_ids"], updated_frames

    def _clear_non_cond_mem_around_input(self, inference_state, frame_idx):
        """
        Remove the non-conditioning memory around the input frame. When users provide
        correction clicks, the surrounding frames' non-conditioning memories can still
        contain outdated object appearance information and could confuse the model.

        This method clears those non-conditioning memories surrounding the interacted
        frame to avoid giving the model both old and new information about the object.
        """
        r = self.memory_temporal_stride_for_eval
        frame_idx_begin = frame_idx - r * self.num_maskmem
        frame_idx_end = frame_idx + r * self.num_maskmem
        batch_size = self._get_obj_num(inference_state)
        for obj_idx in range(batch_size):
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            non_cond_frame_outputs = obj_output_dict["non_cond_frame_outputs"]
            for t in range(frame_idx_begin, frame_idx_end + 1):
                non_cond_frame_outputs.pop(t, None)
    
    def _get_image_feature_bypass_train(self, inference_state, frame_idx, batch_size):
        """Compute the image features on a given frame."""
        # Look up in the cache first
        # image, backbone_out = inference_state["cached_features"].get(
        #     frame_idx, (None, None)
        # )
        # _, backbone_out_WBP = inference_state["cached_features_WBP"].get(
        #     frame_idx, (None, None)
        # )
        # if backbone_out is None or backbone_out_WBP is None:
            # print('Backbone None')
            # exit()
            # Cache miss -- we will run inference on a single image
        # 为了保持计算图完整，对于多个物体的情况，每次还是要进行image encoder的forward
        device = inference_state["device"]
        image = inference_state["images"][frame_idx].to(device).float().unsqueeze(0)
        backbone_out, backbone_out_WBP = self.forward_image_bypass_train(image)
        # Cache the most recent frame's feature (for repeated interactions with
        # a frame; we can use an LRU cache for more frames in the future).
        inference_state["cached_features"] = {frame_idx: (image, backbone_out)}
        inference_state["cached_features_WBP"] = {frame_idx: (image, backbone_out_WBP)}

        # print(backbone_out)
        # expand the features to have the same dimension as the number of objects
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        expanded_backbone_out_WBP = {
            "backbone_fpn": backbone_out_WBP["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out_WBP["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(
                batch_size, -1, -1, -1
            )
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = pos.expand(batch_size, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos
        
        for i, feat in enumerate(expanded_backbone_out_WBP["backbone_fpn"]):
            expanded_backbone_out_WBP["backbone_fpn"][i] = feat.expand(
                batch_size, -1, -1, -1
            )
        for i, pos in enumerate(expanded_backbone_out_WBP["vision_pos_enc"]):
            pos = pos.expand(batch_size, -1, -1, -1)
            expanded_backbone_out_WBP["vision_pos_enc"][i] = pos

        features = self._prepare_backbone_features(expanded_backbone_out)
        features_WBP = self._prepare_backbone_features(expanded_backbone_out_WBP)
        features = (expanded_image,) + features
        features_WBP = (expanded_image,) + features_WBP
        return features, features_WBP
    
    def _run_single_frame_bypass_train(
        self,
        inference_state,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
    ):
        """Run tracking on a single frame based on current inputs and previous memory."""
        # Retrieve correct image features
        
        
        features, features_WBP = self._get_image_feature_bypass_train(inference_state, frame_idx, batch_size)
        
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = features
        (
            _,
            _,
            current_vision_feats_WBP,
            current_vision_pos_embeds_WBP,
            _,
        ) = features_WBP
        
        
        # (
        #     _,
        #     _,
        #     current_vision_feats,
        #     current_vision_pos_embeds,
        #     feat_sizes,
        # ) = self._get_image_feature_bypass_train(inference_state, frame_idx, batch_size)
       
        # point and mask should not appear as input simultaneously on the same frame
        assert point_inputs is None or mask_inputs is None
        current_out, (pix_feat, pix_feat_WBP) = self.track_step_bypass_train(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            current_vision_feats_WBP=current_vision_feats_WBP,
            current_vision_pos_embeds_WBP=current_vision_pos_embeds_WBP,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        # print(current_out["pred_masks"].shape)
        # exit()

        # optionally offload the output to CPU memory to save GPU space
        storage_device = inference_state["storage_device"]
        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        pred_masks_gpu = current_out["pred_masks"]
        # potentially fill holes in the predicted masks
        if self.fill_hole_area > 0:
            pred_masks_gpu = fill_holes_in_mask_scores(
                pred_masks_gpu, self.fill_hole_area
            )
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
        # object pointer is a small tensor, so we always keep it on GPU memory for fast access
        obj_ptr = current_out["obj_ptr"]
        object_score_logits = current_out["object_score_logits"]
        # make a compact version of this frame's output to reduce the state size
        compact_current_out = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
        }
        return compact_current_out, pred_masks_gpu, (pix_feat, pix_feat_WBP)
    
    
    # @torch.enable_grad()
    def propagate_in_video_preflight_bypass_train(self, inference_state):
        """Prepare inference_state and consolidate temporary outputs before tracking."""
        # Check and make sure that every object has received input points or masks.
        
        
        batch_size = self._get_obj_num(inference_state)
        if batch_size == 0:
            raise RuntimeError(
                "No input points or masks are provided for any object; please add inputs first."
            )

        # Consolidate per-object temporary outputs in "temp_output_dict_per_obj" and
        # add them into "output_dict".
        for obj_idx in range(batch_size):
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            for is_cond in [False, True]:
                # Separately consolidate conditioning and non-conditioning temp outputs
                storage_key = (
                    "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
                )
                # Find all the frames that contain temporary outputs for any objects
                # (these should be the frames that have just received clicks for mask inputs
                # via `add_new_points_or_box` or `add_new_mask`)
                for frame_idx, out in obj_temp_output_dict[storage_key].items():
                    # Run memory encoder on the temporary outputs (if the memory feature is missing)
                    if out["maskmem_features"] is None:
                        high_res_masks = torch.nn.functional.interpolate(
                            out["pred_masks"].to(inference_state["device"]),
                            size=(self.image_size, self.image_size),
                            mode="bilinear",
                            align_corners=False,
                        )
                        maskmem_features, maskmem_pos_enc = self._run_memory_encoder(
                            inference_state=inference_state,
                            frame_idx=frame_idx,
                            batch_size=1,  # run on the slice of a single object
                            high_res_masks=high_res_masks,
                            object_score_logits=out["object_score_logits"],
                            # these frames are what the user interacted with
                            is_mask_from_pts=True,
                        )
                        out["maskmem_features"] = maskmem_features
                        out["maskmem_pos_enc"] = maskmem_pos_enc

                    obj_output_dict[storage_key][frame_idx] = out
                    if self.clear_non_cond_mem_around_input:
                        # clear non-conditioning memory of the surrounding frames
                        self._clear_obj_non_cond_mem_around_input(
                            inference_state, frame_idx, obj_idx
                        )

                # clear temporary outputs in `temp_output_dict_per_obj`
                obj_temp_output_dict[storage_key].clear()

            # check and make sure that every object has received input points or masks
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            if len(obj_output_dict["cond_frame_outputs"]) == 0:
                obj_id = self._obj_idx_to_id(inference_state, obj_idx)
                raise RuntimeError(
                    f"No input points or masks are provided for object id {obj_id}; please add inputs first."
                )
            # edge case: if an output is added to "cond_frame_outputs", we remove any prior
            # output on the same frame in "non_cond_frame_outputs"
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
    
    
    @torch.enable_grad()
    def propagate_in_video_for_bypass_train(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
        
    ):
        """Propagate the input points across frames to track in the entire video."""
        self.propagate_in_video_preflight_bypass_train(inference_state)
        # print(inference_state.keys())
        # print(inference_state['output_dict_per_obj'])
        # exit()

        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)

        # set start index, end index, and processing order
        if start_frame_idx is None:
            # default: start from the earliest frame with input points
            start_frame_idx = min(
                t
                for obj_output_dict in inference_state["output_dict_per_obj"].values()
                for t in obj_output_dict["cond_frame_outputs"]
            )
        if max_frame_num_to_track is None:
            # default: track all the frames in the video
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = []  # skip reverse tracking if starting from frame 0
        else:
            end_frame_idx = min(
                start_frame_idx + max_frame_num_to_track, num_frames - 1
            )
            processing_order = range(start_frame_idx, end_frame_idx + 1)
        
        for frame_idx in tqdm(processing_order, desc="propagate in video"):
            self.time_log[frame_idx] = {'IE':[], 'Mem_attn': [], 'MD':[], 'Mem_E':[]}
            
            torch.cuda.empty_cache()
            
                
            
            pred_masks_per_obj = [None] * batch_size
            for obj_idx in range(batch_size):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                # We skip those frames already in consolidated outputs (these are frames
                # that received input clicks or mask). Note that we cannot directly run
                # batched forward on them via `_run_single_frame_inference` because the
                # number of clicks on each object might be different.
                if frame_idx in obj_output_dict["cond_frame_outputs"]:
                    storage_key = "cond_frame_outputs"
                    current_out = obj_output_dict[storage_key][frame_idx]
                    device = inference_state["device"]
                    pred_masks = current_out["pred_masks"].to(device, non_blocking=True)
                    if self.clear_non_cond_mem_around_input:
                        # clear non-conditioning memory of the surrounding frames
                        self._clear_obj_non_cond_mem_around_input(
                            inference_state, frame_idx, obj_idx
                        )
                    (pix_feat, pix_feat_WBP) = (None, None)
                else:
                    storage_key = "non_cond_frame_outputs"
                    # if frame_idx >0 and ((frame_idx-1) // 32) % 2 == 1:
                    if frame_idx % 3 != 0 and frame_idx != end_frame_idx:
                        # self.optimizer.zero_grad()
                        torch.cuda.empty_cache()
                        # print('***********zero_grad frame_{}****************'.format(frame_idx))
                        with torch.no_grad():
                            current_out, pred_masks = self._run_single_frame_inference(
                                inference_state=inference_state,
                                output_dict=obj_output_dict,
                                frame_idx=frame_idx,
                                batch_size=1,  # run on the slice of a single object
                                is_init_cond_frame=False,
                                point_inputs=None,
                                mask_inputs=None,
                                reverse=reverse,
                                run_mem_encoder=True,
                            )
                    else:
                        current_out, pred_masks, (pix_feat, pix_feat_WBP) = self._run_single_frame_bypass_train(
                                inference_state=inference_state,
                                output_dict=obj_output_dict,
                                frame_idx=frame_idx,
                                batch_size=1,  # run on the slice of a single object
                                is_init_cond_frame=False,
                                point_inputs=None,
                                mask_inputs=None,
                                reverse=reverse,
                                run_mem_encoder=True,
                            )
                    
                    # if frame_idx >0 and ((frame_idx-1) // 32) % 2 == 1:
                    #     self.optimizer.zero_grad()
                    #     torch.cuda.empty_cache()
                    #     print('***********zero_grad frame_{}****************'.format(frame_idx))
                    # else:
                    
                        if frame_idx >0 and self.mem_info['sel_win_id'] != None:
                            loss = self.criterion(pix_feat, pix_feat_WBP)
                            self.accumulated_loss += loss.item()
                            (loss/self.train_steps).backward()
                            self.accu_cnt +=1
                            # self.optimizer.step()
                            
                            if self.accu_cnt == self.train_steps or frame_idx==end_frame_idx:
                                for name, param in self.image_encoder.named_parameters():
                                    if param.requires_grad:
                                        print(name, 'requires_grad:',param.requires_grad) 
                                # avg_loss.backward()
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                                torch.cuda.empty_cache()
                                avg_loss = self.accumulated_loss / self.accu_cnt
                                self.iters += 1
                                print(f"=================frame{frame_idx}, Step {self.iters}, avg Loss: {avg_loss:.4f}=================")
                                if self.use_wandb:
                                    try:
                                        wandb.log({'iters':self.iters,'loss':avg_loss})
                                    except:
                                        pass
                                    
                                    # self.writer.add_scalar('training loss (epoch_{})'.format(self.epoch),
                                    #                     avg_loss,
                                    #                     self.iters)
                                    
                                    self.writer.add_scalar('training loss'.format(self.epoch),
                                                        avg_loss,
                                                        self.iters)
                                
                                
                                # 重置累积loss
                                self.accumulated_loss = 0.0
                                self.accu_cnt = 0
                                
                                
                                    # print(self.image_encoder.trunk.blocks[20].bypass_branch.down_project.weight)
                    
                        
                    obj_output_dict[storage_key][frame_idx] = current_out

                inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {
                    "reverse": reverse
                }
                pred_masks_per_obj[obj_idx] = pred_masks
            # print(pred_masks.shape)
            # print(pred_masks.dtype)
            # print(pred_masks)
            # print(pred_masks_per_obj.shape)
            # torch.save(pred_masks_per_obj,'pred_masks_per_obj.pth')
            # exit()
            # Resize the output mask to the original video resolution (we directly use
            # the mask scores on GPU for output to avoid any CPU conversion in between)
            
            
            
            
            # ------------------------------
            # 使用输出mask所在区域选择窗口
            if frame_idx %1 == 0:
                pred_masks = torch.cat(self.mem_info['mask_region'], dim=0)
                self.mem_info['mask_region'] = []
                
                _,_,H,W = pred_masks.shape
                masks = (pred_masks>0).view(-1,H,W)
                # print(Pt_ATMs)
                # if torch.sum(masks) != 0 and current_out['object_score_logits'] > 3:
                if self.mem_info['sel_win_id'] == None :
                    self.mem_info['sel_win_id'] = [i for i in range(25)] if self._WB_info['window_size'] == 14 else [i for i in range(16)]
                    
                    print('restart full')
                    print('selected windows:    ',self.mem_info['sel_win_id'])
                else:
                    masks = torch.any(masks, dim=0)
                    # if self.dilate_mask:
                    masks = dilate_mask(masks)
                    mask_win_id = self.select_mask_windows_v2(masks)
                    if mask_win_id.shape[0] == 0 or any(x < 3 for x in self.mem_info['obj_scores']):
                        Pt_ATMs = torch.stack(self.mem_info['pt_sen_region'], dim=0)
                        self.mem_info['pt_sen_region'] = []
                        # print(Pt_ATMs.shape)
                        
                        if self._WB_info['window_size'] == 14:
                            Pt_ATMs = F.pad(Pt_ATMs, (0, 6, 0, 6), "constant", 0).squeeze(1)
                        else:
                            Pt_ATMs = Pt_ATMs
                        # print(Pt_ATMs.shape)
                        B = Pt_ATMs.shape[0]
                        if self._WB_info['window_size'] == 14:
                            Pt_ATMs = Pt_ATMs.reshape(B,5,14,5,14).permute(0,1,3,2,4).reshape(B,25,-1).sum(-1)
                        else:
                            Pt_ATMs = Pt_ATMs.reshape(B,4,16,4,16).permute(0,1,3,2,4).reshape(B,16,-1).sum(-1)
                        # PF_win_id = torch.where(Pt_ATMs > 0.1)[1]
                        PF_win_id = cumulative_threshold_selection(Pt_ATMs, threshold=0.6)
                        # print(PF_win_id)
                        # print(self.mem_info['obj_scores'])
                    else:
                        PF_win_id = torch.tensor([], device=device, dtype=torch.int32)
                        
                        # PF_win_id = torch.where(Pt_ATMs > 0.3)[1]
                        # self.mem_info['sel_win_id'] = self.select_mask_windows(masks)
                # else:
                    # self.mem_info['sel_win_id'] = -1
                    self.mem_info['sel_win_id'] = torch.unique(torch.cat([PF_win_id, mask_win_id]))
                    self.mem_info['obj_scores'] = []
                    # self.mem_info['sel_win_id'] = mask_win_id
                    if self.print_WS or frame_idx % 50 ==0:
                        print('selected windows:    ',self.mem_info['sel_win_id'])
                        print('Prompt Focus windows:',torch.unique(PF_win_id))
                        # print('**** Mask windows ***',mask_win_id)
                        print(torch.unique(mask_win_id))
                    # print('selected windows:    ',self.mem_info['sel_win_id'])
                    # print('Prompt Focus windows:',torch.unique(PF_win_id))
                    # print('Mask windows:        ',mask_win_id)
                    '''
                    mem_masks_stack_obj = self.mem_info['mem_masks_stack'].get(obj_idx,None)
                    if mem_masks_stack_obj != None:
                        self.mem_info['mem_masks'][obj_idx] = []
                        if len(mem_masks_stack_obj) == 5:
                            for i in range(4):
                                mem_masks_obj_Li = [mask[i] for mask in mem_masks_stack_obj]
                                self.mem_info['mem_masks'][obj_idx].append(torch.cat(mem_masks_obj_Li,dim=0))
                    '''

            if len(pred_masks_per_obj) > 1:
                all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
            else:
                all_pred_masks = pred_masks_per_obj[0]
            _, video_res_masks = self._get_orig_video_res_output(
                inference_state, all_pred_masks
            )
            
            if frame_idx==end_frame_idx:
                self.mem_info['sel_win_id'] = None
                # torch.save(self.mem_info['ious'], '/home/zhangjing/sam2_Proj/hahaha/score_vis/ious/1_021093_iou_sel_4.pth')
                self.mem_info['ious'] = []
                self.mem_info['obj_scores'] = []
                self.mem_info['mask_region'] = []
                self.mem_info['pt_sen_region'] = []
                self.mem_info['mem_masks'] = {}
                self.mem_info['mem_masks_stack'] = {}
                self.mem_info['mem_merge_ids'] = {}
                self.mem_info['mem_merge_ids_stack'] = {}
                self.mem_info['enable_mem_prune'] = False
                self.mem_info['frame_delete_indices']=[]
                
            # yield frame_idx, obj_ids, video_res_masks, (pix_feat, pix_feat_WBP)



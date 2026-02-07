
from heapq import merge
import math
from this import d
from typing import Tuple, List, Optional, Callable
import torch
import torch.nn as nn
from sam2.modeling.backbones.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)
from sam2.modeling.backbones.hieradet import Hiera, MultiScaleBlock, MultiScaleAttention, do_pool
from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from sam2.modeling.sam.transformer import RoPEAttention, Attention
from sam2.modeling.position_encoding import apply_rotary_enc, compute_axial_cis, reshape_for_broadcast

from build_model.merger import global_merger, attn_global_merger, restore
import torch.nn.functional as F
# from build_model.merge import bipartite_soft_matching, bipartite_soft_matching_random2d, do_nothing
from build_model.merge import *
from build_model.utils import *
from sam2.sam2_video_predictor_with_bypass import SAM2VideoPredictor_bypass
from sam2.sam2_video_predictor_with_bypass_stg import SAM2VideoPredictor_bypass_stg
from sam2.sam2_video_predictor_with_bypass_mem_filter import SAM2VideoPredictor_bypass_mem_filter

import random
import time
from torch import nn, Tensor


def create_mask_randperm(shape, false_ratio, device='cuda'):
    """使用随机排列索引，精确控制False的数量"""
    mask = torch.ones(shape, dtype=torch.bool, device=device)
    num_false = int(shape * false_ratio)
    false_indices = torch.randperm(shape, device=device)[:num_false]
    mask[false_indices] = False
    return mask


def create_mask_rand(shape, sample_ratio, device='cuda'):
    """使用随机数与阈值比较生成mask"""
    mask = torch.rand(shape, device=device) <= sample_ratio
    return mask

def global_merge(
  C: torch.Tensor,  # [1,1,N, C] - 拼接后的大tensor
  global_unm_idx: torch.Tensor,  # 全局未合并索引
  global_src_idx: torch.Tensor,  # 全局源索引
  global_dst_idx: torch.Tensor,  # 全局目标索引
  mode: str = "mean"
) -> torch.Tensor:
    """
    使用全局索引一次性merge大tensor
    
    Args:
        C: 拼接后的tensor [1, 1, N, C]
        global_unm_idx: 全局未合并索引
        global_src_idx: 全局源索引  
        global_dst_idx: 全局目标索引
        mode: merge模式 ("mean", "sum", etc.)
    
    Returns:
        merged_tensor: merge后的tensor
    """
    C = C.squeeze(0).squeeze(0)
    N, feat_dim = C.shape
    # global_src_idx.unsqueeze(0).unsqueeze(0)
    # global_dst_idx.unsqueeze(0).unsqueeze(0)
    # 1. 自动计算未合并索引：所有不在src和dst中的索引
    all_indices = torch.arange(N, device=C.device)
    used_indices = torch.cat([global_src_idx, global_dst_idx])
    
    # 找出未使用的索引作为unm_idx
    mask = torch.ones(N, dtype=torch.bool, device=C.device)
    mask[used_indices] = False
    global_unm_idx = all_indices[mask]
    # print(len(global_src_idx))
    # print(global_unm_idx.dtype)
    # 获取未合并的tokens
    C_work = C.clone()
    src_tokens = C_work.gather(dim=-2, index=global_src_idx.unsqueeze(-1).expand(len(global_src_idx),feat_dim))
    C_work.scatter_reduce_(-2, 
                           global_dst_idx.unsqueeze(-1).expand(len(global_src_idx), feat_dim), 
                           src_tokens, 
                           reduce=mode, 
                           include_self=True)
    
    # 4. 提取最终结果：未合并tokens + 更新后的dst tokens
    unique_dst_idx = torch.unique(global_dst_idx, sorted=True)
    final_idx = torch.cat([global_unm_idx, unique_dst_idx])
    result = C_work.gather(dim=-2, index=final_idx.unsqueeze(-1).expand(len(final_idx), feat_dim))
    result = result.unsqueeze(0).unsqueeze(0)
    # print(N, result.shape[0])
    return result


def global_merge_0(
  C: torch.Tensor,  # [N, C] - 拼接后的大tensor
  global_unm_idx: torch.Tensor,  # 全局未合并索引
  global_src_idx: torch.Tensor,  # 全局源索引
  global_dst_idx: torch.Tensor,  # 全局目标索引
  mode: str = "mean"
) -> torch.Tensor:
    """
    使用全局索引一次性merge大tensor
    
    Args:
        C: 拼接后的tensor [N, C]
        global_unm_idx: 全局未合并索引
        global_src_idx: 全局源索引  
        global_dst_idx: 全局目标索引
        mode: merge模式 ("mean", "sum", etc.)
    
    Returns:
        merged_tensor: merge后的tensor
    """
    # C = C.squeeze(0).squeeze(0)
    N, feat_dim = C.shape
    # print(global_unm_idx)
    # print(global_unm_idx.dtype)
    # 获取未合并的tokens
    if len(global_unm_idx) > 0:
        unmerged = C[global_unm_idx]
    else:
        unmerged = torch.empty(0, feat_dim, device=C.device, dtype=C.dtype)
    # print(unmerged.shape)
    # 执行merge操作
    if len(global_src_idx) > 0:
        src_tokens = C[global_src_idx]  # [num_merges, C]
        dst_tokens = C[global_dst_idx]  # [num_merges, C]
        
        if mode == "mean":
            merged_tokens = (src_tokens + dst_tokens) / 2
        elif mode == "sum":
            merged_tokens = src_tokens + dst_tokens
        else:
            raise ValueError(f"Unsupported merge mode: {mode}")
    else:
        merged_tokens = torch.empty(0, feat_dim, device=C.device, dtype=C.dtype)
    
    # 拼接结果
    if len(unmerged) > 0 and len(merged_tokens) > 0:
        result = torch.cat([unmerged, merged_tokens], dim=0)
    elif len(unmerged) > 0:
        result = unmerged
    elif len(merged_tokens) > 0:
        result = merged_tokens
    else:
        result = torch.empty(0, feat_dim, device=C.device, dtype=C.dtype)
    # result = result.unsqueeze(0).unsqueeze(0)
    # print(N, result.shape[0])
    return result


def mask_window_partition(mask, window_size):
    # window_size = self.mem_info['window_sizes'][-2]*4   # 特征图尺寸是64*64, mask尺寸是256*256，需要统一比例
    # mask需为2维bool类型
    H, W = mask.shape
    Hp = (H + window_size - 1) // window_size * window_size
    Wp = (W + window_size - 1) // window_size * window_size
    masks_padded = F.pad(
        mask,  # 兼容非二值输入
        (0, Wp - W, 0, Hp - H),
        mode='constant',
        value=False
    )  # -> [B, Hp, Wp]
    m, n = masks_padded.shape[0]//window_size, window_size
    masks_padded = masks_padded.reshape(m,n,m,n).permute(0,2,1,3).reshape(m*m,n*n)
    
    return masks_padded

def apply_rotary_enc_prune(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    N_k: int,
    repeat_freqs_k: bool = False,
    prune_mask: torch.Tensor = None,
    MTP_id_st: int = 4096,
    MTP_id_ed: int = 4096*6
):
    # print(xq.shape)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = (
        torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        if xk.shape[-2] != 0
        else None
    )
    # print(xq_.shape)
    # print(freqs_cis.shape)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # print(xq_.shape)
    # print(freqs_cis.shape)
    # x_t1 = xq_ * freqs_cis
    # print(x_t1.shape)
    # x_t2 = torch.view_as_real(xq_ * freqs_cis)
    # print(x_t2.shape)
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    if xk_ is None:
        # no keys to rotate, due to dropout
        return xq_out.type_as(xq).to(xq.device), xk
    # repeat freqs along seq_len dim to match k seq_len
    if repeat_freqs_k:
        r = N_k // xq_.shape[-2]
        if freqs_cis.is_cuda:
            freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
        else:
            # torch.repeat on complex numbers may not be supported on non-CUDA devices
            # (freqs_cis has 4 dims and we repeat on dim 2) so we use expand + flatten
            freqs_cis = freqs_cis.unsqueeze(2).expand(-1, -1, r, -1, -1).flatten(2, 3)
    # freqs_cis.shape=torch.Size([1, 1, 28672, 128])
    # print(prune_mask)
    # A = freqs_cis[:,:,prune_mask,:]
    # print(A.shape)
    prune_mask_pad = torch.ones(freqs_cis.shape[2], dtype=torch.bool, device=freqs_cis.device)
    assert MTP_id_ed - MTP_id_st == len(prune_mask)
    prune_mask_pad[MTP_id_st:MTP_id_ed] = prune_mask
    xk_out = torch.view_as_real(xk_ * freqs_cis[:,:,prune_mask_pad,:]).flatten(3)
    
    
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


class WB_Hiera(Hiera):
    
    def backbone_forward(self, x: torch.Tensor) -> List[torch.Tensor]:

        
        x = self.patch_embed(x)
        # x: (B, H, W, C)

        # Add pos embed
        x = x + self._get_pos_embed(x.shape[1:3])
        # print('input x:', x.shape)

        outputs = []
        for i, blk in enumerate(self.blocks):
            if i in self.selected_layers:
                x,_ = blk.backbone_forward(x)
            else:
                x = blk(x)
                
            # print(x.shape)
                
            # try:
            #     x,_ = blk(x)
            # except:
            #     x = blk(x)
            # print('block_{} x:'.format(i), x.shape)
            if (i == self.stage_ends[-1]) or (
                i in self.stage_ends and self.return_interm_layers
            ):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)
        # exit()
        return outputs
    
    
    def bypass_forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        self._WB_info["size"] = None
        self._WB_info["source"] = None
        self._WB_info["rel_pos"] = None
        self._WB_info["selected_layers"] = list(self.selected_layers)
        # self._WB_info["window_size"] = self.window_size
        # self._WB_info["threshold"] = self.threshold
        self._WB_info["mask"] = None
        self._WB_function['merge'] = [do_nothing]
        self._WB_function['unmerge'] = [do_nothing]
        self._WB_info['cur_token'] = 196
        self._WB_info["short_cut"] = None
        self.mem_info['frame_delete_indices']=[]
        
        
        x = self.patch_embed(x)
        # x: (B, H, W, C)

        # Add pos embed
        x = x + self._get_pos_embed(x.shape[1:3])
        # print('input x:', x.shape)

        outputs = []
        for i, blk in enumerate(self.blocks):
            if i in self._WB_info["selected_layers"]:
                x,_ = blk.bypass_forward(x)
            else:
                x = blk(x)
            # print(x.shape)
                
            # try:
            #     x,_ = blk(x)
            # except:
            #     x = blk(x)
            # print('block_{} x:'.format(i), x.shape)
            if (i == self.stage_ends[-1]) or (
                i in self.stage_ends and self.return_interm_layers
            ):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)
        # exit()
        return outputs
    
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        self._WB_info["size"] = None
        self._WB_info["source"] = None
        self._WB_info["rel_pos"] = None
        self._WB_info["selected_layers"] = list(self.selected_layers)
        # self._WB_info["window_size"] = self.window_size
        # self._WB_info["threshold"] = self.threshold
        self._WB_info["mask"] = None
        self._WB_function['merge'] = [do_nothing]
        self._WB_function['unmerge'] = [do_nothing]
        self._WB_info['cur_token'] = 196
        self._WB_info["short_cut"] = None
        self.mem_info['frame_delete_indices']=[]
        
        x = self.patch_embed(x)
        # x: (B, H, W, C)

        # Add pos embed
        x = x + self._get_pos_embed(x.shape[1:3])
        # print('input x:', x.shape)

        outputs = []
        for i, blk in enumerate(self.blocks):
            if i in self._WB_info["selected_layers"]:
                # x,_ = blk(x)
                if self.disable_WB:
                    x,_ = blk.backbone_forward(x)
                    # print('----backbone forrward----')
                else:
                    x,_ = blk.bypass_forward(x)
                    # print('****bypass forrward****')
                    
            else:
                x = blk(x)
                
            if (i == self.stage_ends[-1]) or (
                i in self.stage_ends and self.return_interm_layers
            ):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)
        # exit()
        return outputs


class WB_Hiera_1(nn.Module):
    # def sub_init(self, org_model:Hiera, ):
    #     self.window_spec = org_model.self.window_spec
    #     self.q_stride = org_model.q_stride
    #     self.stage_ends = org_model.stage_ends
    #     self.q_pool_blocks
    def __init__(self, org_model:Hiera):
        super().__init__()
        self.window_spec = org_model.window_spec
        self.q_stride = org_model.q_stride
        self.stage_ends = org_model.stage_ends
        self.q_pool_blocks = org_model.q_pool_blocks
        self.return_interm_layers = org_model.return_interm_layers
        self.patch_embed = org_model.patch_embed
        self.global_att_blocks = org_model.global_att_blocks
        self.window_pos_embed_bkg_spatial_size = org_model.window_pos_embed_bkg_spatial_size
        self.pos_embed = org_model.pos_embed
        self.pos_embed_window = org_model.pos_embed_window
        self.blocks = org_model.blocks
        self.channel_list = org_model.channel_list
        # self._WB_info = {}
        
    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        self._WB_info["size"] = None
        self._WB_info["source"] = None
        self._WB_info["rel_pos"] = None
        self._WB_info["selected_layers"] = list(self.selected_layers)
        self._WB_info["window_size"] = self.window_size
        self._WB_info["threshold"] = self.threshold
        
        x = self.patch_embed(x)
        # x: (B, H, W, C)

        # Add pos embed
        x = x + self._get_pos_embed(x.shape[1:3])

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.stage_ends[-1]) or (
                i in self.stage_ends and self.return_interm_layers
            ):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)

        return outputs

    def get_layer_id(self, layer_name):
        # https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
        num_layers = self.get_num_layers()

        if layer_name.find("rel_pos") != -1:
            return num_layers + 1
        elif layer_name.find("pos_embed") != -1:
            return 0
        elif layer_name.find("patch_embed") != -1:
            return 0
        elif layer_name.find("blocks") != -1:
            return int(layer_name.split("blocks")[1].split(".")[1]) + 1
        else:
            return num_layers + 1

    def get_num_layers(self) -> int:
        return len(self.blocks)

   
    # def forward(self, *args, **kwdargs) -> torch.Tensor:
    
    #     self._WB_info["size"] = None
    #     self._WB_info["source"] = None
    #     self._WB_info["rel_pos"] = None
    #     self._WB_info["selected_layers"] = list(self.selected_layers)
    #     self._WB_info["window_size"] = self.window_size
    #     self._WB_info["threshold"] = self.threshold

        
    #     return super().forward(*args, **kwdargs)


class WB_MultiScaleBlock_small(MultiScaleBlock):
    # 所有全局注意力前加小bypass
    
    def bypass_forward(self, x: torch.Tensor) -> torch.Tensor:
        # Window partition
        layer_idx = self._WB_info["selected_layers"].pop(0)
        window_size = self.window_size
        # print(layer_idx, self.mem_info)

        if window_size > 0:
            # shortcut = x
            # x = self.norm1(x)
            # if layer_idx in [7,14,18]:
            if layer_idx in [7,14,18]:
                
                H, W = x.shape[1], x.shape[2]
                x, pad_hw = window_partition(x, window_size)
                self._WB_info["x_shape"] = x.shape
                mask = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
               
                if self.mem_info['sel_win_id'] != None and self.mem_info['sel_win_id'] != -1:
                    mask_idx = self.mem_info['sel_win_id']
                else:
                    mask_idx = [i for i in range(x.shape[0])]
                    # mask_idx = [0,4,20,24]

                mask[mask_idx] = True
                # x_shape = x.shape
                self._WB_info["x_R"] = x[~mask]
                # print(x_.shape)
                self._WB_info["mask"] = mask
                self._WB_info["pad_hw"] = pad_hw
                self._WB_info["HW"] = (H,W)
                self._WB_info["window_size"] = window_size
                
                x_ = x[mask]
                
            else:
                x_ = x
            
            shortcut_ = x_
            
            x_ = self.norm1(x_)
            x_ = self.attn(x_)
            x_ = shortcut_ + self.drop_path(x_)
            x_ = x_ + self.drop_path(self.mlp(self.norm2(x_)))
            # 不参与注意力计算的token如果仍然通过mlp和ln，会不会好一些
            # self._WB_info["x_R"] = self.norm1(self._WB_info["x_R"])
            # self._WB_info["x_R"] = self._WB_info["x_R"] + self.drop_path(self.mlp(self.norm2(self._WB_info["x_R"])))

            # print(x_.shape)
            return x_, None
          
        else:
            x_R= self.bypass_branch(self._WB_info["x_R"])
            x_ = x
            
            mask = self._WB_info["mask"]
            x_combined = torch.empty(self._WB_info["x_shape"], 
                                device=x.device, dtype=x.dtype)
            if x_R.dtype != x_.dtype:
                x_R = x_R.to(x_.dtype)
            
            x_combined[~mask] = x_R
            x_combined[mask] = x_
            
            x = window_unpartition(x_combined, self._WB_info["window_size"], self._WB_info["pad_hw"], self._WB_info["HW"])
            
            shortcut = x  # B, H, W, C
            x = self.norm1(x)
            if self.dim != self.dim_out:
                shortcut = do_pool(self.proj(x), self.pool)
                
            x = self.attn(x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))                
           
            return x, None
  
    
    def backbone_forward(self,x: torch.Tensor):
        x = super().forward(x)
        
        return x, None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        x,_ = self.bypass_forward(x)
        
           
        return x, None
    
   
    
class WB_MultiScaleBlock_all(MultiScaleBlock):
    # 原本流程：shaotcut_s+norm -> 降维 -> 分窗 -> attetion -> 合并窗 -> FFN
    # 在降维层将目标窗和背景窗分别降维,并且合并窗口
    # 在降维层下一层分窗
    
    
    def bypass_forward(self, x: torch.Tensor) -> torch.Tensor:
        # shortcut = x  # B, H, W, C
        # x = self.norm1(x)

        # Skip connection
        # if self.dim != self.dim_out:
        #     shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        layer_idx = self._WB_info["selected_layers"].pop(0)
        window_size = self.window_size
        # print(layer_idx,window_size)
        # print(window_size)
        # exit()
        # print(layer_idx, self.window_size, x.shape)

        if window_size > 0:
            # shortcut = x
            # x = self.norm1(x)
            # if layer_idx in [7,14,18]:
            # 处理降维层：pool降维 -> 目标窗attention -> 合并窗口 -> FFN
            
            if self.dim != self.dim_out:
                x = self.norm1(x)
                
                H, W = x.shape[1], x.shape[2]
                x, pad_hw = window_partition(x, window_size)
                self._WB_info["x_shape"] = x.shape
                
                mask = torch.zeros(x.shape[0], dtype=torch.bool)
                mask_idx = self.mem_info['sel_win_ids'].get(layer_idx,None)
                
                # if layer_idx == self._WB_info['win_sel_layer'][-1]:
                #     window_mask = self._WB_info["window_mask"]
                #     self._WB_info["sel_window_mask"] = window_mask[mask_idx].view(-1)
               
               
                mask[mask_idx] = True
                # x_shape = x.shape
                x_R = x[~mask]
                x = x[mask]
                # print(x_.shape)
                self._WB_info["mask"] = mask
                self._WB_info["pad_hw"] = pad_hw
                self._WB_info["HW"] = (H,W)
                
                shortcut = do_pool(self.proj(x), self.pool)
                # print(shortcut.shape)
                x_R = do_pool(self.proj(x_R), self.pool)
                self._WB_info["x_R"] = x_R
                
                x_shape = list(self._WB_info["x_shape"])
                x_shape[1] = x_shape[1] // self.q_stride[0]
                x_shape[2] = x_shape[2] // self.q_stride[1]
                x_shape[3] = x_shape[3] * 2
                
                # print('DS layer:',layer_idx, x.shape)
                x,_ = self.attn(x)
                if self.q_stride:
                # Shapes have changed due to Q pooling
                    window_size = self.window_size // self.q_stride[0]
                    # H, W = shortcut.shape[1:3]
                    H, W = self._WB_info["HW"][0] // self.q_stride[0], self._WB_info["HW"][1] // self.q_stride[1]
                    pad_h = (window_size - H % window_size) % window_size
                    pad_w = (window_size - W % window_size) % window_size
                    pad_hw = (H + pad_h, W + pad_w)
                    
                    self._WB_info["pad_hw"] = pad_hw
                    self._WB_info["HW"] = (H,W)
                    self._WB_info["window_size_stg"] = window_size
                
                # x_cb = torch.empty(x_shape,device=x.device,dtype=x.dtype)
                # # x_R = x_R.to(dtype=x_.dtype)
                # # x_ = x_.to(dtype=x.dtype)
                # x_cb[~self._WB_info["mask"]] = x_R
                # x_cb[self._WB_info["mask"]] = x
                # self._WB_info["x_R"] = x_R
                # x = window_unpartition(x, window_size, pad_hw, (H, W))

                x = shortcut + self.drop_path(x)
                # MLP
                x = x + self.drop_path(self.mlp(self.norm2(x)))
                
                
                # 下一层的窗口大小变化，因此要恢复原空间结构
                x_cb = torch.empty(x_shape,device=x.device,dtype=x.dtype)
                # x_R = x_R.to(dtype=x_.dtype)
                # x_ = x_.to(dtype=x.dtype)
                x_cb[~self._WB_info["mask"]] = x_R
                x_cb[self._WB_info["mask"]] = x
                # print(x.shape)
                # print(x.shape, window_size, pad_hw, (H, W))
                x = window_unpartition(x_cb, window_size, pad_hw, (H, W))
                
                # print(x.shape)
                return x, None
            
            if layer_idx in self._WB_info['win_sel_layer']:
                
                H, W = x.shape[1], x.shape[2]
                x, pad_hw = window_partition(x, window_size)
                # print(pad_hw)
                # exit()
                self._WB_info["x_shape"] = x.shape
                # print(x.shape)
                
                mask = torch.zeros(x.shape[0], dtype=torch.bool)
                # mask_idx = [0,5]
                # mask_idx = [0,5,10,15,10,1,6,11,16,21,2,7,12,17,22]
                # print(self.mem_info)
                # ##############
                # if self.mem_info['sel_win_id'] != None:
                mask_idx = self.mem_info['sel_win_ids'].get(layer_idx,None)
                # print(mask_idx)
                
                # 最后一个stage，需要为全局注意力做一个mask的补偿
                if layer_idx == self._WB_info['win_sel_layer'][-1]:
                    window_mask = self._WB_info["window_mask"]
                    self._WB_info["sel_window_mask"] = window_mask[mask_idx].view(-1)
               
               
                mask[mask_idx] = True
                # x_shape = x.shape
                self._WB_info["x_R"] = x[~mask]
                x_ = x[mask]
                # print(x_.shape)
                self._WB_info["mask"] = mask
                self._WB_info["pad_hw"] = pad_hw
                self._WB_info["HW"] = (H,W)
                # self._WB_info["window_size"] = window_size
            else:
                x_ = x
                
            
            shortcut_ = x_
            x_ = self.norm1(x_)
            # print('sel_layer/regular_layer:',layer_idx, x_.shape)
            x_,_ = self.attn(x_)
            x_ = shortcut_ + self.drop_path(x_)
            x_ = x_ + self.drop_path(self.mlp(self.norm2(x_)))
            
            # 如果处于stage的最后一次，需要恢复原始空间结构，便于FPN处理
            if layer_idx in self._WB_info['fpn_feat_layer']:
                # x_ = x
                x_R= self._WB_info["x_R"]
                # x_R= self._WB_info["x_R"]
                x_cb = torch.empty(self._WB_info["x_shape"],device=x.device,dtype=x.dtype)
                # x_R = x_R.to(dtype=x_.dtype)
                x_ = x_.to(dtype=x.dtype)
                x_cb[~self._WB_info["mask"]] = x_R
                x_cb[self._WB_info["mask"]] = x_
                # print(x_cb.shape, self._WB_info["window_size"], self._WB_info["pad_hw"], self._WB_info["HW"])
                
                x_ = window_unpartition(x_cb, self.window_size, self._WB_info["pad_hw"], self._WB_info["HW"])

            return x_, None
          
        else:
            if len(self._WB_info["selected_layers"])==0:
                x_ = x
                x_R= self.bypass_branch(self._WB_info["x_R"])
                # x_R= self._WB_info["x_R"]
                x_cb = torch.empty(self._WB_info["x_shape"],device=x.device,dtype=x.dtype)
                # x_R = x_R.to(dtype=x_.dtype)
                x_ = x_.to(dtype=x.dtype)
                x_cb[~self._WB_info["mask"]] = x_R
                x_cb[self._WB_info["mask"]] = x_
                x = window_unpartition(x_cb, self._WB_info["window_size"], self._WB_info["pad_hw"], self._WB_info["HW"])
                
                shortcut = x  # B, H, W, C
                x = self.norm1(x)
                # if self.dim != self.dim_out:
                #     shortcut = do_pool(self.proj(x), self.pool)
                # print('final global layer:', layer_idx, x.shape)
                x,_ = self.attn(x)
                x = shortcut + self.drop_path(x)
                x = x + self.drop_path(self.mlp(self.norm2(x)))
                
                # print(x.shape)
            else:
                B,H,W,C = x.shape
                # x = x.view(1,-1,C)
                
                # window_mask = self._WB_info["window_mask"]
                # if self.mem_info['sel_win_id'] != None and self.mem_info['sel_win_id'] != -1:
                #     mask_idx = self.mem_info['sel_win_id']
                # else:
                #     mask_idx = [i for i in range(x.shape[0])]
                # sel_window_mask = window_mask[mask_idx].view(-1)
                # print(x.shape)
                x = x.view(-1,C)
                # sel_window_mask = sel_window_mask.view(-1)
                # sel_window_mask = self._WB_info["sel_window_mask"]
                x = x[self._WB_info["sel_window_mask"]].unsqueeze(0)
                
                shortcut = x  # B, H, W, C
                x = self.norm1(x)
                # if self.dim != self.dim_out:
                #     shortcut = do_pool(self.proj(x), self.pool)
                # print('internal global layer:',layer_idx, x.shape)
                x,_ = self.attn(x,True)
                x = shortcut + self.drop_path(x)
                x = x + self.drop_path(self.mlp(self.norm2(x)))
                
                # print(x.shape)
                x_ = torch.zeros((B*H*W,C), device=x.device,dtype=x.dtype)
                # x_.scatter(0, sel_window_mask.nonzero(as_tuple=True)[0].unsqueeze(1).expand(-1, C).to(device=x.device), x[0])
                x_[self._WB_info["sel_window_mask"]] = x[0]
                x = x_.view(B,H,W,C)
                # x = x.view(B,H,W,C)
                
           
            return x, None
        
    def backbone_forward(self,x: torch.Tensor):
        x = super().forward(x)
        
        return x, None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x,_ = self.backbone_forward(x)
        x,_ = self.bypass_forward(x)
        # # Window partition

           
        return x, None


class WB_MultiScaleBlock(MultiScaleBlock):
    # 窗口跳层4 + 大bypass + mask optimize
    
    def bypass_forward(self, x: torch.Tensor) -> torch.Tensor:
        # shortcut = x  # B, H, W, C
        # x = self.norm1(x)

        # Skip connection
        # if self.dim != self.dim_out:
        #     shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        layer_idx = self._WB_info["selected_layers"].pop(0)
        window_size = self.window_size
        # print(window_size)
        # exit()
        # print(layer_idx, self.mem_info)

        if window_size > 0:
            # shortcut = x
            # x = self.norm1(x)
            # if layer_idx in [7,14,18]:
            if layer_idx in self._WB_info['win_sel_layer']:
                
                H, W = x.shape[1], x.shape[2]
                x, pad_hw = window_partition(x, window_size)
                # print(pad_hw)
                # exit()
                self._WB_info["x_shape"] = x.shape
                # print(x.shape)
                
                mask = torch.zeros(x.shape[0], dtype=torch.bool)
                # mask_idx = [0,5]
                # mask_idx = [0,5,10,15,10,1,6,11,16,21,2,7,12,17,22]
                # print(self.mem_info)
                # ##############
                # if self.mem_info['sel_win_id'] != None:
                mask_idx = self.mem_info['sel_win_id']
                # print(mask_idx)
                # ##############
                # mask_idx = [i for i in range(25)]
                # mask_idx = [0,4,12,20,24]
                # mask_idx = random.sample([i for i in range(25)], 5)
                
                # mask_idx = [0,3,10,12,15]
                # elif self.mem_info['sel_win_id'] == -1:
                #     mask_idx = random.choice([[0,1,2,5,6,7,10,11,12],[2,3,4,7,8,9,12,13,14],[10,11,12,15,16,17,20,21,22],[12,13,14,17,18,19,22,23,24]])
                    
                # else:
                #     mask_idx = [i for i in range(x.shape[0])]
               
               
                mask[mask_idx] = True
                # print(mask.sum())
                # x_shape = x.shape
                self._WB_info["x_R"] = x[~mask]
                x_ = x[mask]
                # print(self._WB_info["x_R"].shape)
                # print(x_.shape)
                self._WB_info["mask"] = mask
                self._WB_info["pad_hw"] = pad_hw
                self._WB_info["HW"] = (H,W)
                
                window_mask = self._WB_info["window_mask"]
                self._WB_info["sel_window_mask"] = window_mask[mask_idx].view(-1)
                # self._WB_info["window_size"] = window_size
            else:
                x_ = x
                
            
            shortcut_ = x_
            x_ = self.norm1(x_)
            x_,_ = self.attn(x_)
            x_ = shortcut_ + self.drop_path(x_)
            x_ = x_ + self.drop_path(self.mlp(self.norm2(x_)))
            

            # print(x_.shape)
            return x_, None
          
        else:
            if len(self._WB_info["selected_layers"])==0:
                x_ = x
                x_R= self.bypass_branch(self._WB_info["x_R"])
                # x_R= self._WB_info["x_R"]
                x_cb = torch.empty(self._WB_info["x_shape"],device=x.device,dtype=x.dtype)
                # x_R = x_R.to(dtype=x_.dtype)
                x_ = x_.to(dtype=x.dtype)
                x_cb[~self._WB_info["mask"]] = x_R
                x_cb[self._WB_info["mask"]] = x_
                x = window_unpartition(x_cb, self._WB_info["window_size"], self._WB_info["pad_hw"], self._WB_info["HW"])
                
                shortcut = x  # B, H, W, C
                x = self.norm1(x)
                # if self.dim != self.dim_out:
                #     shortcut = do_pool(self.proj(x), self.pool)
                    
                x,_ = self.attn(x)
                x = shortcut + self.drop_path(x)
                x = x + self.drop_path(self.mlp(self.norm2(x)))
                
            else:
                B,H,W,C = x.shape
                
                x = x.view(-1,C)
                x = x[self._WB_info["sel_window_mask"]].unsqueeze(0)
                
                shortcut = x  # B, H, W, C
                x = self.norm1(x)
                # if self.dim != self.dim_out:
                #     shortcut = do_pool(self.proj(x), self.pool)
                    
                x, _ = self.attn(x,False)
                x = shortcut + self.drop_path(x)
                x = x + self.drop_path(self.mlp(self.norm2(x)))
                
                # print(x.shape)
                x_ = torch.zeros((B*H*W,C), device=x.device,dtype=x.dtype)
                # x_.scatter(0, sel_window_mask.nonzero(as_tuple=True)[0].unsqueeze(1).expand(-1, C).to(device=x.device), x[0])
                x_[self._WB_info["sel_window_mask"]] = x[0]
                x = x_.view(B,H,W,C)
                # x = x.view(B,H,W,C)
                
           
            return x, None
        
    def backbone_forward(self,x: torch.Tensor):
        x = super().forward(x)
        
        return x, None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x,_ = self.backbone_forward(x)
        x,_ = self.bypass_forward(x)
        # # Window partition

           
        return x, None


    
class WB_MultiScaleBlock_wj(MultiScaleBlock):
    # 窗口跳层
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shortcut = x  # B, H, W, C
        # x = self.norm1(x)

        # Skip connection
        # if self.dim != self.dim_out:
        #     shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)
            
            mask = torch.zeros(x.shape[0], dtype=torch.bool)
            mask_idx = [0]
            # mask_idx = [0,5,10,15,10,1,6,11,16,21,2,7,12,17,22]
            # mask_idx = [24,19,14,9,4]
            # mask_idx = [i for i in range(25)]
            # mask_idx.pop(4)
            mask[mask_idx] = True
            # print(x.shape)
            # x_shape = x.shape
            x_R = x[~mask]
            x_ = x[mask]
            # print(x_.shape)
            # exit()
            
            shortcut = x_
            x_ = self.norm1(x_)
            x_ = self.attn(x_)
            x_ = shortcut + self.drop_path(x_)
            x_ = x_ + self.drop_path(self.mlp(self.norm2(x_)))
            
            # x = torch.zeros_like(x,device=x.device,dtype=x_.dtype)
            x.to(dtype=x_.dtype)
            # x = torch.zeros_like(x,device=x.device,dtype=x.dtype)
            x_R = x_R.to(dtype=x_.dtype)
            # x_ = x_.to(dtype=x.dtype)
            x[~mask] = x_R
            x[mask] = x_
            x = window_unpartition(x, window_size, pad_hw, (H, W))
            
            # x = self.attn(x)
            
            # print(x.dtype, x_.dtype)
        else:
            shortcut = x  # B, H, W, C
            x = self.norm1(x)
            if self.dim != self.dim_out:
                shortcut = do_pool(self.proj(x), self.pool)
                
            x = self.attn(x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            
            # return x, None
        # print(x.shape, x_R.shape)
        # exit()
        # Window Attention + Q Pooling (if stage change)
        # if self.q_stride:
        #     # Shapes have changed due to Q pooling
        #     window_size = self.window_size // self.q_stride[0]
        #     H, W = shortcut.shape[1:3]

        #     pad_h = (window_size - H % window_size) % window_size
        #     pad_w = (window_size - W % window_size) % window_size
        #     pad_hw = (H + pad_h, W + pad_w)
        # print(x.dtype, x_.dtype)
        # exit()
        
        # Reverse window partition
        # if self.window_size > 0:
            # x = torch.zeros_like(x,device=x.device,dtype=x_.dtype)
            # x_R = x_R.to(dtype=x_.dtype)
            # x[~mask] = x_R
            # x[mask] = x_
            # x = window_unpartition(x, window_size, pad_hw, (H, W))

        # x = shortcut + self.drop_path(x)
        # MLP
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        # B,h,w,c = x.shape
        # x = x.reshape(B,h*w,c)
        # print(x.shape)
        # exit()
        # print(x.dtype)
        return x, None


class WB_MultiScaleBlock_global(MultiScaleBlock):
    # 所有层都用全局注意力
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x  # B, H, W, C
        x = self.norm1(x)

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        window_size = self.window_size
        # if window_size > 0:
        #     H, W = x.shape[1], x.shape[2]
        #     x, pad_hw = window_partition(x, window_size)

        # print(x.shape)
        # Window Attention + Q Pooling (if stage change)
        x = self.attn(x)
        # if self.q_stride:
        #     # Shapes have changed due to Q pooling
        #     window_size = self.window_size // self.q_stride[0]
        #     H, W = shortcut.shape[1:3]

        #     pad_h = (window_size - H % window_size) % window_size
        #     pad_w = (window_size - W % window_size) % window_size
        #     pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        # if self.window_size > 0:
        #     x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # B,h,w,c = x.shape
        # x = x.reshape(B,h*w,c)
        # print(x.shape)
        # exit()
        return x, None
   
   
      
class WB_MultiScaleAttention_all(MultiScaleAttention):
    
    def forward(self, x: torch.Tensor, shape_BNC=False) -> torch.Tensor:
        if shape_BNC:
            B,N,_ = x.shape
            # B,H,W,_ = x.shape
        else:
            B, H, W, _ = x.shape
            N = H*W
            
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1)
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)
        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            assert shape_BNC==False
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]  # downsampled shape
            q = q.reshape(B, H * W, self.num_heads, -1)
        # print(q.dtype)
        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        # print(x.dtype)
        
        # Transpose back
        x = x.transpose(1, 2)
        # x = x.reshape(B, H, W, -1)
        if shape_BNC:
            x = x.reshape(B, N, -1)
        else:
            x = x.reshape(B,H,W,-1)

        x = self.proj(x)

        return x,None

        
\
    
class WB_MultiScaleAttention(MultiScaleAttention):
    
    def forward(self, x: torch.Tensor, reshape_BHWC=True) -> torch.Tensor:
       
        
        if reshape_BHWC:
            B,H,W,_ = x.shape
            N = H*W
        else:
            B,N,_ = x.shape
        # qkv with shape (B, H * W, 3, nHead, C)
        # qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1)
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)
        # print(self.q_pool)
        # print(q.shape)
        assert self.q_pool==None
        # Q pooling (for downsample at stage changes)
        # if self.q_pool:
        #     q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
        #     H, W = q.shape[1:3]  # downsampled shape
        #     q = q.reshape(B, H * W, self.num_heads, -1)
        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        # print(x.dtype)
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        
        # Transpose back
        x = x.transpose(1, 2)
        if reshape_BHWC:
            x = x.reshape(B,H,W,-1)
        else:
            x = x.reshape(B, N, -1)
        x = self.proj(x)
        
        return x, (q,k)
    
class WB_MultiScaleAttention_merge(MultiScaleAttention):
    # 定义merger类，由于加速效果不佳，换一种方法试试有无区别
    def build_merger(self, class_token=False, distill_token=False):
        self.merger = attn_global_merger(class_token=False, distill_token=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # 插入merge操作
        B,H,W,C = x.shape
        x = x.reshape(B,H*W,C)

        layer_idx = self._WB_info["selected_layers"].pop(0)
        merge = self.merger.patch_matching(
                    x,  # 或者改用metrics（key）
                    layer_idx,
                    self._WB_info["source"],
                    # self._WB_info["mask"]
                )
        # if self._WB_info["trace_source"]:
        #     self._WB_info["source"] = self.merger.merge_source(
        #         x, self._WB_info["source"]
        #     )
        
        # print('x before merge:', x.shape)
        x = self.merger.merge(x)
        # x, self._WB_info["size"] = self.merger.merge_wavg(x, self._WB_info["size"])
        # print('x after merge:', x.shape)
        # x = x.reshape(B,H,W,C)
        B,N,C = x.shape
        # qkv with shape (B, H * W, 3, nHead, C)
        # qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1)
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)
        # print(self.q_pool)
        assert self.q_pool==None
        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]  # downsampled shape
            q = q.reshape(B, H * W, self.num_heads, -1)

        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        # Transpose back
        x = x.transpose(1, 2)
        # print('x before unmerge:', x.shape)

        x = self.merger.unmerge(x.reshape(B,N,C))
        # print('x after unmerge:', x.shape)
        x = x.reshape(B, H, W, -1)

        x = self.proj(x)
        # exit()
        return x, k.mean(1)

  
class WB_MemoryAttention(MemoryAttention):
    # 仅匹配一次patch
    def forward(
        self,
        curr: torch.Tensor,  # self-attention inputs
        memory: torch.Tensor,  # cross-attention inputs
        curr_pos: Optional[torch.Tensor] = None,  # pos_enc for self-attention inputs
        memory_pos: Optional[torch.Tensor] = None,  # pos_enc for cross-attention inputs
        num_obj_ptr_tokens: int = 0,  # number of object pointer *tokens*
    ):
        
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = (
                curr[0],
                curr_pos[0],
            )

        assert (
            curr.shape[1] == memory.shape[1]
        ), "Batch size must be the same for curr and memory"

        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        # print(output.shape)
        # print(memory.shape)
        # exit()
        if self.batch_first:
            # Convert to batch first
            # print('batch')
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)
        
        # print(memory_pos.shape)
        # print(curr_pos.shape)

        for i, layer in enumerate(self.layers):
            # if i > 1:
            #     break
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens,
                        "layer_idx":i}

            output = layer(
                tgt=output,
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                **kwds,
            )
        normed_output = self.norm(output)

        if self.batch_first:
            # Convert back to seq first
            normed_output = normed_output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
        

        return normed_output
    
class WB_RoPEAttention_sa(RoPEAttention):
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_k_exclude_rope: int = 0
    ) -> torch.Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Apply rotary position encoding
        w = h = math.sqrt(q.shape[-2])
        self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != q.shape[-2]:
            print('recompute')
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat
        # print(self.freqs_cis.shape)
        num_k_rope = k.size(-2) - num_k_exclude_rope
        # print(num_k_exclude_rope)
        # print(q.shape)
        # print(self.freqs_cis.shape)
        
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )
        
        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        # out = flex_attention(q, k, v)
        # out = q
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out, (q,k)




class WB_RoPEAttention_ca(RoPEAttention):
    # 分开计算注意力
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_k_exclude_rope: int = 0, layer_idx: int = None
    ) -> torch.Tensor:
        # Input projections
        
        mem_masks = self.mem_info['mem_masks'].get(self.mem_info['obj_idx'])
        # print(mem_masks)
        if self.mem_info['enable_mem_prune'] and mem_masks != None and len(mem_masks)!=0:
            
            B,N_k,C = k.shape
            
            mem_mask = mem_masks[layer_idx] # 5 * 4096
            if len(self.mem_info['frame_delete_indices'])!=0:
                frame_mask = self.mem_info['frame_mask']
                mem_mask = mem_mask.reshape(-1,4096)
                mem_mask = mem_mask[frame_mask].reshape(-1)
                # print(self.mem_info['frame_delete_indices'], mem_mask.shape[0]//4096)
                
            MTP_id_st = 4096 *1
            # 减去条件帧和上一帧（-2），减去prune的帧。位置偏置（+1）
            MTP_id_ed = 4096*(self.mem_info['num_maskmem'] - 2 - self.mem_info['num_frame_to_prune'] + 1)
            # print(MTP_id_ed, MTP_id_st, len(mem_mask))
            assert MTP_id_ed-MTP_id_st==len(mem_mask)
            k_sel = k[:,MTP_id_st:MTP_id_ed,:][:, mem_mask, :]
            v_sel = v[:,MTP_id_st:MTP_id_ed,:][:, mem_mask, :]
            # k_sel = k[:,4096*7,:][:, mem_mask, :]
            # v_sel = v[:,4096*7,:][:, mem_mask, :]
            
            k = torch.cat([k[:,:MTP_id_st,:], k_sel, k[:,MTP_id_ed:,:]], dim=1)
            v = torch.cat([v[:,:MTP_id_st,:], v_sel, v[:,MTP_id_ed:,:]], dim=1)
            
            # k = torch.cat([k_sel, k[:,4096*7:,:]], dim=1)
            # v = torch.cat([v_sel, v[:,4096*7:,:]], dim=1)

            # N_k_p = mem_mask.sum()
            assert k.shape == v.shape
            # exit()
            B,N_k_p,C = k.shape
            
            self.drop_ratio_log.append(1-N_k_p/N_k)
            # print('layer id:', layer_idx,'mem_token:', N_k, 'pruned:', k.shape, 'SR:', 1-N_k_p/N_k)
            
        # else:
            q = self.q_proj(q)
            k = self.k_proj(k)
            # k_LF = self.k_proj(k_LF)
            v = self.v_proj(v)

            # Separate into heads
            q = self._separate_heads(q, self.num_heads)
            k = self._separate_heads(k, self.num_heads)
            # k_ = self._separate_heads(k_, self.num_heads)
            v = self._separate_heads(v, self.num_heads)

            # Apply rotary position encoding
            w = h = math.sqrt(q.shape[-2])
            self.freqs_cis = self.freqs_cis.to(q.device)
            if self.freqs_cis.shape[0] != q.shape[-2]:
                print('recompute')
                self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
            if q.shape[-2] != k.shape[-2]:
                assert self.rope_k_repeat
            # print(self.freqs_cis.shape)
            # num_k_rope = k.size(-2) - num_k_exclude_rope
            num_k_rope = N_k_p - num_k_exclude_rope
            # print(num_k_exclude_rope)
            # print(q.shape)
            # print(self.freqs_cis.shape)
        
            q, k[:, :, :num_k_rope] = apply_rotary_enc_prune(
                q,
                k[:, :, :num_k_rope],
                freqs_cis=self.freqs_cis,
                repeat_freqs_k=self.rope_k_repeat,
                prune_mask=mem_mask,
                N_k = N_k, 
                MTP_id_st = MTP_id_st,
                MTP_id_ed = MTP_id_ed
            )
            
            dropout_p = self.dropout_p if self.training else 0.0
            # Attention
            # print(q.shape, k.shape)
            # out1 = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
            scale_factor = 1 / math.sqrt(q.size(-1))
            # attn_bias = torch.zeros(L, S, dtype=q.dtype, device=q.device)
            attn_weight = q @ k.transpose(-2, -1) * scale_factor
            # attn_weight += attn_bias
            attn_weight_ = torch.softmax(attn_weight, dim=-1)
            attn_weight_ = torch.dropout(attn_weight_, dropout_p, train=self.training)
            out = attn_weight_ @ v
            out = self._recombine_heads(out)
            out = self.out_proj(out)
            
            # k_ = torch.zeros((B,N_k,C),dtype=k.dtype,device=k.device)
            # print(q.shape, k_.shape)
            # return out, (q,k[:, :, num_k_rope-4096:num_k_rope], v[:,:,num_k_rope-4096:num_k_rope])
            return out, (attn_weight[0,0,:,num_k_rope-4096:num_k_rope], v[:,:,num_k_rope-4096:num_k_rope])
        
        else:
            q = self.q_proj(q)
            k = self.k_proj(k)
            v = self.v_proj(v)

            # Separate into heads
            q = self._separate_heads(q, self.num_heads)
            k = self._separate_heads(k, self.num_heads)
            v = self._separate_heads(v, self.num_heads)

            # Apply rotary position encoding
            w = h = math.sqrt(q.shape[-2])
            self.freqs_cis = self.freqs_cis.to(q.device)
            if self.freqs_cis.shape[0] != q.shape[-2]:
                print('recompute')
                self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
            if q.shape[-2] != k.shape[-2]:
                assert self.rope_k_repeat
            # print(self.freqs_cis.shape)
            num_k_rope = k.size(-2) - num_k_exclude_rope
            # print(num_k_exclude_rope)
            # print(q.shape)
            # print(self.freqs_cis.shape)
            
            q, k[:, :, :num_k_rope] = apply_rotary_enc(
                q,
                k[:, :, :num_k_rope],
                freqs_cis=self.freqs_cis,
                repeat_freqs_k=self.rope_k_repeat,
            )
            
            dropout_p = self.dropout_p if self.training else 0.0
            # Attention
            # out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
            scale_factor = 1 / math.sqrt(q.size(-1))
            # attn_bias = torch.zeros(L, S, dtype=q.dtype, device=q.device)
            attn_weight = q @ k.transpose(-2, -1) * scale_factor
            # attn_weight += attn_bias
            attn_weight_ = torch.softmax(attn_weight, dim=-1)
            attn_weight_ = torch.dropout(attn_weight_, dropout_p, train=self.training)
            out = attn_weight_ @ v
            # out = flex_attention(q, k, v)
            # out = q
            out = self._recombine_heads(out)
            out = self.out_proj(out)
            # print('sssssss')
            # exit()

            # return out, (q,k[:,:,num_k_rope-4096:num_k_rope],v[:,:,num_k_rope-4096:num_k_rope])
            return out, (attn_weight[0,0,:,num_k_rope-4096:num_k_rope], v[:,:,num_k_rope-4096:num_k_rope])

 
class EfficientRoPEAttention1(RoPEAttention):
    """Attention with rotary position encoding."""


    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0
    ) -> Tensor:
        # print('Efficient RoPe')
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Apply rotary position encoding
        w = h = math.sqrt(q.shape[-2])
        self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat

        num_k_rope = k.size(-2) - num_k_exclude_rope
        # print('num_k_rope:',num_k_rope)
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )

        dropout_p = self.dropout_p if self.training else 0.0

        if self.rope_k_repeat:
            fs, bs, ns, ds = k.shape
            nq = q.shape[-2]
            if num_k_rope <= nq:
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
            else:
                # print(k.shape)
                s_kernel_size = self.pooling_ks
                intw, inth = int(w), int(h)
                k_landmarks = k[:, :, :num_k_rope, :].reshape(fs, -1, nq, ds)
                k_landmarks = k_landmarks.transpose(-2, -1).reshape(fs, -1, intw, inth)
                k_landmarks = F.avg_pool2d(
                    k_landmarks, s_kernel_size, stride=s_kernel_size
                )
                k_landmarks = (
                    k_landmarks.reshape(
                        fs, -1, ds, nq // (s_kernel_size * s_kernel_size)
                    )
                    .transpose(-2, -1)
                    .reshape(fs, bs, -1, ds)
                )

                # print(k_landmarks.shape)
                scale_factor = 1 / math.sqrt(ds)
                attn_weight = q @ k_landmarks.transpose(
                    -2, -1
                ) * scale_factor
                # attn_weight = q @ k_landmarks.transpose(
                #     -2, -1
                # ) * scale_factor + 2 * math.log(s_kernel_size)
                attn_weight = torch.cat(
                    [
                        attn_weight,
                        q @ k[:, :, num_k_rope:, :].transpose(-2, -1) * scale_factor,
                    ],
                    dim=-1,
                )
                attn_weight = torch.softmax(attn_weight, dim=-1)
                attn_weight = torch.dropout(attn_weight, dropout_p, train=self.training)

                v_landmarks = v[:, :, :num_k_rope, :].reshape(fs, -1, nq, ds)
                v_landmarks = v_landmarks.transpose(-2, -1).reshape(fs, -1, intw, inth)
                v_landmarks = F.avg_pool2d(
                    v_landmarks, s_kernel_size, stride=s_kernel_size
                )
                v_landmarks = v_landmarks.reshape(
                    fs, -1, ds, nq // (s_kernel_size * s_kernel_size)
                ).transpose(-2, -1)
                v_landmarks = torch.cat(
                    [
                        v_landmarks.reshape(fs, bs, -1, ds),
                        v[:, :, num_k_rope:, :],
                    ],
                    dim=-2,
                )
                out = attn_weight @ v_landmarks
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out, None

class EfficientRoPEAttention2(RoPEAttention):
    """Attention with rotary position encoding."""


    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0
    ) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Apply rotary position encoding
        w = h = math.sqrt(q.shape[-2])
        self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat

        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )

        dropout_p = self.dropout_p if self.training else 0.0

        if self.rope_k_repeat:
            fs, bs, ns, ds = k.shape
            nq = q.shape[-2]
            if num_k_rope <= nq:
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
            else:
                s_kernel_size = 2
                intw, inth = int(w), int(h)
                k_landmarks = k[:, :, :num_k_rope, :].reshape(fs, -1, nq, ds)
                k_landmarks = k_landmarks.transpose(-2, -1).reshape(fs, -1, intw, inth)
                k_landmarks = F.avg_pool2d(
                    k_landmarks, s_kernel_size, stride=s_kernel_size
                )
                k_landmarks = k_landmarks.reshape(
                    fs, -1, ds, nq // (s_kernel_size * s_kernel_size)
                ).transpose(-2, -1)
                k_landmarks = torch.cat(
                    [
                        k_landmarks.reshape(fs, bs, -1, ds)
                        + 2 * math.log(s_kernel_size),
                        k[:, :, num_k_rope:, :],
                    ],
                    dim=-2,
                )

                v_landmarks = v[:, :, :num_k_rope, :].reshape(fs, -1, nq, ds)
                v_landmarks = v_landmarks.transpose(-2, -1).reshape(fs, -1, intw, inth)
                v_landmarks = F.avg_pool2d(
                    v_landmarks, s_kernel_size, stride=s_kernel_size
                )
                v_landmarks = v_landmarks.reshape(
                    fs, -1, ds, nq // (s_kernel_size * s_kernel_size)
                ).transpose(-2, -1)
                v_landmarks = torch.cat(
                    [
                        v_landmarks.reshape(fs, bs, -1, ds),
                        v[:, :, num_k_rope:, :],
                    ],
                    dim=-2,
                )
                out = F.scaled_dot_product_attention(
                    q, k_landmarks, v_landmarks, dropout_p=dropout_p
                )
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out, None
    
def memory_split_reshape(memory_tokens, num_obj_ptr_tokens, token_per_frame=4096):
    B, N, C = memory_tokens.shape
    assert (N-num_obj_ptr_tokens)%token_per_frame == 0
    
    frame_num = N // token_per_frame
    memory_tokens_feature = memory_tokens[:,:N-num_obj_ptr_tokens,:]
    memory_tokens_obj = memory_tokens[:,N-num_obj_ptr_tokens:,:]
    # print(frame_num)
    # print(memory_tokens.shape)
    # print(memory_tokens_feature.shape)
    # exit()
    memory_tokens_feature = memory_tokens_feature.reshape(B*frame_num, token_per_frame, C)
    
    return memory_tokens_feature, memory_tokens_obj


class WB_MemoryAttentionLayer(MemoryAttentionLayer):
    def _forward_sa(self, tgt, query_pos):
        # Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2, _ = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0, layer_idx=None):
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope,
                    "layer_idx":layer_idx}

        # Cross-Attention
        tgt2 = self.norm2(tgt)
        tgt2, _ = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        num_k_exclude_rope: int = 0,
        layer_idx=None
    ) -> torch.Tensor:

        # Self-Attn, Cross-Attn
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope, layer_idx)
        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt




def build_WB_model(args, sam_predictor, selected_layers: list, prop_attn: bool = True):
    
    WB_info = {
            "size": None,
            "source": None,
            "prop_attn": prop_attn,
            "class_token": None,
            "distill_token": False,
            "rel_pos": None,
            "threshold":0.88,
            "ratio": 0.35,
            "generator": None,
        }
    if args.apply_WB:
        
        if 'base' in args.sam2_model:
            selected_layers = args.selected_layers
            WB_info['window_size']=14
            WB_info['win_sel_layer'] = args.win_sel_layer
            WB_info['fpn_feat_layer'] = args.fpn_feat_layer
            WB_info['final_global_layer'] = 20
            WB_info['selected_layers'] = selected_layers
            WB_info['x_shape'] = (25,14,14,448)
            WB_info["pad_hw"] = (70,70)
            WB_info["HW"] = (64,64)
        else:
            selected_layers = args.selected_layers
            WB_info['window_size']=16
            WB_info['win_sel_layer'] = args.win_sel_layer
            WB_info['fpn_feat_layer'] = args.fpn_feat_layer
            WB_info['final_global_layer'] = 43
            WB_info['selected_layers'] = selected_layers
            WB_info['x_shape'] = (16,16,16,576)
            WB_info["pad_hw"] = (64,64)
            WB_info["HW"] = (64,64)
            
            
        device = next(sam_predictor.parameters()).device
        if args.WB_all_layer:
            sam_predictor.__class__=SAM2VideoPredictor_bypass_stg
        else:
            sam_predictor.__class__=SAM2VideoPredictor_bypass
        try:
            if args.Mem_filter:
                sam_predictor.__class__=SAM2VideoPredictor_bypass_mem_filter
        except:
            pass
        
        sam_predictor._WB_info = WB_info
        
        # sam_predictor.init_memory_info()
        model = sam_predictor.image_encoder.trunk
        model.__class__ = WB_Hiera
        # setattr(model, 'trunk', WB_Hiera(model.trunk))
        # model = WB_Hiera(model)

        model.selected_layers = selected_layers     # 由于这个列表每次推理都会被pop掉，所以在模型里存一下
        # model.window_size = (2,2)
        model.threshold = 0.88
        model._WB_info = WB_info
        model._WB_function = {}
        model.disable_WB=args.disable_WB

        if hasattr(model, "dist_token") and model.dist_token is not None:
            model._WB_info["distill_token"] = True

        # model._WB_info["generator"] = init_generator(next(model.parameters()))
        window_mask = torch.ones(64, 64, dtype=torch.bool, device=device)
        window_mask = mask_window_partition(window_mask, window_size=WB_info['window_size'])
        model._WB_info["window_mask"] = window_mask
 
        len_att = 0
        len_block = 0
        for module in model.modules():
            if isinstance(module, MultiScaleBlock):
                if len_block in model.selected_layers:
                    if args.small_bypass:
                        module.__class__ = WB_MultiScaleBlock_small
                    elif args.WB_all_layer:
                        module.__class__ = WB_MultiScaleBlock_all
                    else:
                        module.__class__ = WB_MultiScaleBlock
                    module._WB_info = model._WB_info
                    module._WB_function = model._WB_function
                    module.mem_info = model.mem_info
                len_block +=1 
                
            elif isinstance(module, MultiScaleAttention):
                if len_att in model.selected_layers: 
                    if args.WB_all_layer:
                        module.__class__ = WB_MultiScaleAttention_all
                    else:
                        module.__class__ = WB_MultiScaleAttention
                    module._WB_info = model._WB_info
                    module._WB_function = model._WB_function
                    module.mem_info = model.mem_info
                    # module.build_merger(class_token=module._WB_info['class_token'], distill_token=module._WB_info['distill_token'])
                len_att +=1
        

    sam_predictor.prune_memory = True if args.prune_memory else False
    if args.prune_memory:
        mem_attn_module = sam_predictor.memory_attention
        mem_attn_module.__class__ = WB_MemoryAttention
        mem_attn_module._WB_info = WB_info
        mem_attn_module.mem_info = sam_predictor.mem_info
        for name, module in mem_attn_module.named_modules():
            if isinstance(module, MemoryAttentionLayer):
                # if len_att in mem_attn_module.selected_layers:
                module.__class__ = WB_MemoryAttentionLayer
                module._WB_info = mem_attn_module._WB_info
                module.mem_info = mem_attn_module.mem_info
                # module._WB_function = mem_attn_module._WB_function
                    # module.build_merger(class_token=module._WB_info['class_token'], distill_token=module._WB_info['distill_token'])
                    # len_att +=1 
            elif isinstance(module, RoPEAttention):
                # if len_block in mem_attn_module.selected_layers: 
                if 'self_attn' in  name:
                    module.__class__ = WB_RoPEAttention_sa
                    module._WB_info = mem_attn_module._WB_info
                    module.mem_info = mem_attn_module.mem_info
                    # module._WB_function = mem_attn_module._WB_function
                    
                        # module.build_merger(class_token=module._WB_info['class_token'], distill_token=module._WB_info['distill_token'])
                        # len_block +=1 
                if 'cross_attn_image' in name:
                    module.__class__ = WB_RoPEAttention_ca
                    module._WB_info = mem_attn_module._WB_info
                    module.mem_info = mem_attn_module.mem_info
                    # module._WB_function = mem_attn_module._WB_function
                    
    elif args.pool_memory:
        mem_attn_module = sam_predictor.memory_attention
        for name, module in mem_attn_module.named_modules():
            if isinstance(module, RoPEAttention):
                module.__class__ = EfficientRoPEAttention1
                module.pooling_ks=args.pooling_ks
                    
                    
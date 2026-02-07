
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
from sam2.modeling.sam.transformer import RoPEAttention
from sam2.modeling.position_encoding import apply_rotary_enc, compute_axial_cis

from build_model.merger import global_merger, attn_global_merger, restore
import torch.nn.functional as F
# from build_model.merge import bipartite_soft_matching, bipartite_soft_matching_random2d, do_nothing
from build_model.merge import *
from build_model.utils import *
import time


class MP_Hiera(Hiera):
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        self._MP_info["size"] = None
        self._MP_info["source"] = None
        self._MP_info["rel_pos"] = None
        self._MP_info["selected_layers"] = list(self.selected_layers)
        self._MP_info["mask"] = None
        self._MP_function['merge'] = [do_nothing]
        self._MP_function['unmerge'] = [do_nothing]
        self._MP_info['cur_token'] = 196
        self._MP_info["short_cut"] = None
        self._MP_info["merge_log_per_frame"] = []
        
        
        
        x = self.patch_embed(x)
        # x: (B, H, W, C)

        # Add pos embed
        x = x + self._get_pos_embed(x.shape[1:3])
        # print('input x:', x.shape)

        outputs = []
        for i, blk in enumerate(self.blocks):
            if i in self._MP_info["selected_layers"]:
                x,_ = blk(x)
            else:
                x = blk(x)

            if (i == self.stage_ends[-1]) or (
                i in self.stage_ends and self.return_interm_layers
            ):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)
        # exit()
        return outputs

class MP_MultiScaleBlock(MultiScaleBlock):
    # 窗口跳层4 + 大bypass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Window partition
        layer_idx = self._MP_info["selected_layers"].pop(0)
        window_size = self.window_size
        # print(layer_idx, self.mem_info)

        if window_size > 0:
            if layer_idx in [7]:
                
                H, W = x.shape[1], x.shape[2]
                x, pad_hw = window_partition(x, window_size)
                self._MP_info["x_shape"] = x.shape
                mask = torch.zeros(x.shape[0], dtype=torch.bool)
                if self.mem_info['sel_win_id'] != None:
                    mask_idx = self.mem_info['sel_win_id']
                else:
                    mask_idx = [i for i in range(x.shape[0])]

                mask[mask_idx] = True
                # x_shape = x.shape
                self._MP_info["x_R"] = x[~mask]
                x_ = x[mask]
                # print(x_.shape)
                self._MP_info["mask"] = mask
                self._MP_info["pad_hw"] = pad_hw
                self._MP_info["HW"] = (H,W)
                self._MP_info["window_size"] = window_size
                
            else:
                x_ = x
            
            shortcut_ = x_
            
            # shortcut = x_
            x_ = self.norm1(x_)
            x_ = self.attn(x_)
            x_ = shortcut_ + self.drop_path(x_)
            x_ = x_ + self.drop_path(self.mlp(self.norm2(x_)))
            
            return x_, None
          
        else:
            if len(self._MP_info["selected_layers"])==0:
                x_R= self.bypass_branch(self._MP_info["x_R"])
                x_ = x
                x = torch.zeros(self._MP_info["x_shape"],device=x.device,dtype=x.dtype)
                # x_R = x_R.to(dtype=x_.dtype)
                x_ = x_.to(dtype=x.dtype)
                x[~self._MP_info["mask"]] = x_R
                x[self._MP_info["mask"]] = x_
                x = window_unpartition(x, self._MP_info["window_size"], self._MP_info["pad_hw"], self._MP_info["HW"])
                shortcut = x  # B, H, W, C
                x = self.norm1(x)
                if self.dim != self.dim_out:
                    shortcut = do_pool(self.proj(x), self.pool)
                    
                x = self.attn(x)
                x = shortcut + self.drop_path(x)
                x = x + self.drop_path(self.mlp(self.norm2(x)))
                # print(x.shape)
            else:
                # if self._MP_info["x_R"]
                # print(self._MP_info["x_R"])
                B1,H,W,C = x.shape
                B2,H,W,C = self._MP_info["x_R"].shape
                x_ = torch.cat([x,self._MP_info["x_R"]],dim=0)
                x_ = x_.reshape(1,-1,C)
                shortcut = x_  # B, H, W, C
                x_ = self.norm1(x_)
                if self.dim != self.dim_out:
                    shortcut = do_pool(self.proj(x_), self.pool)
                    
                x_ = self.attn(x_)
                x_ = shortcut + self.drop_path(x_)
                x_ = x_ + self.drop_path(self.mlp(self.norm2(x_)))
                # print(x.shape)
                x_ = x_.reshape(B1+B2,H,W,C)
                x, self._MP_info["x_R"] = torch.split(x_,[B1,B2],dim=0)
                
           
            return x, None

     

class MP_MultiScaleBlock_ToMe(MultiScaleBlock):
    '''
    在第一个窗口划分层进行分窗，后续不再根据空间结构分窗
    只有窗口层merge，全局层仅进行交互
    每层merge比例为r的token
    '''
    
    def merge_patch(self, x, merge:list):
        self.B,self.H,self.W,self.C = x.shape
        window_size = self.window_size
        if window_size == 0:
            # print('global x before merge:', x.shape)
            
            x, self.pad_hw = window_partition(x, self._MP_info['cur_window_size'])
            # pad_mask = torch.zeros(1,pad_hw[0],pad_hw[1],C, device=x.device, dtype=x.dtype)
            
        B,h,w,C = x.shape
        x = x.reshape(B, h*w, C)
        # print('x before merge:', x.shape)
        for sub_merge in  merge:
            x = sub_merge(x)
        self._MP_info['cur_token'] = x.shape[1]
        # print('x after merge:', x.shape)
        
        B,self.N_r, C = x.shape
        if window_size == 0:
            x = x.reshape(1, -1, C)
            # print('global x after merge:', x.shape)
            
        
        return x

    def unmerge_patch(self, x, unmerge:list):
        
        for sub_unmerge in reversed(unmerge):
            x = sub_unmerge(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        layer_idx = self._MP_info["selected_layers"].pop(0)
        window_size = self.window_size
        
        if window_size > 0:
            if layer_idx == self._MP_info['first_win_layer']:
        
                assert self.dim == self.dim_out

                # Window partition
                window_size = self.window_size
                H, W = x.shape[1], x.shape[2]
                x, pad_hw = window_partition(x, window_size)
                self._MP_info['pad_hw'] = pad_hw
                self._MP_info['window_size'] = window_size
                self._MP_info['HW'] = (H,W)
                B,H,W,C = x.shape
                # print(x.shape)
                x = x.reshape(B,H*W,C)
            
            elif layer_idx in self._MP_info['window_partion_layers']:
                N_w = self._MP_info['cur_win_token_num']
                B, N, C = x.shape
                x = x.reshape(N//N_w, N_w, C)
                assert x.shape[0]*x.shape[1] == N

            N = x.shape[1]
            r = int(N * self._MP_info['ratio'])
            # print(r)
            merge, unmerge = bipartite_soft_matching(
                    x,  # 或者改用metrics（key）
                    layer_idx,
                    r=r
                )
            self._MP_function['unmerge'].append(unmerge)
            
            
            # merge x
            x, self._MP_info["size"] = merge_wavg(merge, x, self._MP_info["size"])
            self._MP_info['cur_win_token_num'] = x.shape[1]
        

            # attention block
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        else:
            if len(self._MP_info["selected_layers"])==0:
                unmerge = self._MP_function['unmerge']
                N_m = x.shape[0] * x.shape[1]
                x = self.unmerge_patch(x,unmerge)
                N = x.shape[0] * x.shape[1]

                x = window_unpartition(x, self._MP_info["window_size"], self._MP_info["pad_hw"], self._MP_info["HW"])
                
                # attention block
                shortcut = x
                x = self.norm1(x)
                x = self.attn(x, True)
                x = shortcut + self.drop_path(x)
                x = x + self.drop_path(self.mlp(self.norm2(x)))
                # print('prune ratio:', N_m/N)

            else:
                B, N, C = x.shape
                x = x.reshape(1, B*N, C)

                # attention block
                shortcut = x
                x = self.norm1(x)
                x = self.attn(x)
                x = shortcut + self.drop_path(x)
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            
        return x, None
    
    
class MP_MultiScaleBlock_ALGM(MultiScaleBlock):
    '''
    # 在第一个窗口划分层进行分窗，后续不再根据空间结构分窗
    # 只有窗口层merge，全局层仅进行交互
    # 每层merge比例为r的token
    在第一个窗口层进行全局窗口merge，第一次全局注意力层进行全局merge
    '''
    
    def merge_patch(self, x, merge:list):
        self.B,self.H,self.W,self.C = x.shape
        window_size = self.window_size
        if window_size == 0:
            # print('global x before merge:', x.shape)
            
            x, self.pad_hw = window_partition(x, self._MP_info['cur_window_size'])
            # pad_mask = torch.zeros(1,pad_hw[0],pad_hw[1],C, device=x.device, dtype=x.dtype)
            
        B,h,w,C = x.shape
        x = x.reshape(B, h*w, C)
        # print('x before merge:', x.shape)
        for sub_merge in  merge:
            x = sub_merge(x)
        self._MP_info['cur_token'] = x.shape[1]
        # print('x after merge:', x.shape)
        
        B,self.N_r, C = x.shape
        if window_size == 0:
            x = x.reshape(1, -1, C)
            # print('global x after merge:', x.shape)
            
        
        return x

    def unmerge_patch(self, x, unmerge:list):
        
        for sub_unmerge in reversed(unmerge):
            x = sub_unmerge(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        layer_idx = self._MP_info["selected_layers"].pop(0)
        window_size = self.window_size

        if window_size > 0:
            if layer_idx == self._MP_info['first_win_layer']:
        
                assert self.dim == self.dim_out

                # Window partition
                window_size = self.window_size
                H, W = x.shape[1], x.shape[2]
                x, pad_hw = window_partition(x, window_size)
                self._MP_info['pad_hw'] = pad_hw
                self._MP_info['window_size'] = window_size
                self._MP_info['HW'] = (H,W)
                B,H,W,C = x.shape
                # print(x.shape)
                x = x.reshape(B,H*W,C)
                
                # print(x.shape)
                merge, unmerge = conditional_pooling(x, self._MP_info['pooling_threshold'], self._MP_info['pooling_window_size'])
                self._MP_function['unmerge'].append(unmerge)
            
                # merge x
                x, self._MP_info["size"] = merge_wavg(merge, x, self._MP_info["size"])
                self._MP_info['cur_win_token_num'] = x.shape[1]
                self._MP_info['merge_log_per_frame'].append(x.shape[0]*x.shape[1])
                # print(x.shape)
            
            elif layer_idx in self._MP_info['window_partion_layers']:
                N_w = self._MP_info['cur_win_token_num']
                B, N, C = x.shape
                x = x.reshape(N//N_w, N_w, C)
                assert x.shape[0]*x.shape[1] == N
                
                # print(x.shape)
                merge, unmerge = ALGM_global_patch_matching(x, layer_idx)
                self._MP_function['unmerge'].append(unmerge)
                
                x, self._MP_info["size"] = merge_wavg(merge, x, self._MP_info["size"])
                self._MP_info['cur_win_token_num'] = x.shape[1]
                self._MP_info['merge_log_per_frame'].append(x.shape[0]*x.shape[1])
                

            # attention block
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))


        else:
            if len(self._MP_info["selected_layers"])==0:
                unmerge = self._MP_function['unmerge']
                N_m = x.shape[0] * x.shape[1]
                x = self.unmerge_patch(x,unmerge)
                N = x.shape[0] * x.shape[1]

                x = window_unpartition(x, self._MP_info["window_size"], self._MP_info["pad_hw"], self._MP_info["HW"])
                
                # attention block
                shortcut = x
                x = self.norm1(x)
                x = self.attn(x, True)
                x = shortcut + self.drop_path(x)
                x = x + self.drop_path(self.mlp(self.norm2(x)))
                # print('prune ratio:', N_m/N)

            else:
                B, N, C = x.shape
                x = x.reshape(1, B*N, C)

                # attention block
                shortcut = x
                x = self.norm1(x)
                x = self.attn(x)
                x = shortcut + self.drop_path(x)
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            
        return x, None


class MP_MultiScaleBlock_ToMe4DM(MultiScaleBlock):
    '''
    在第一个窗口划分层进行分窗，后续不再根据空间结构分窗
    只有窗口层merge，全局层仅进行交互
    每层merge比例为r的token
    '''
    
    def merge_patch(self, x, merge:list):
        self.B,self.H,self.W,self.C = x.shape
        window_size = self.window_size
        if window_size == 0:
            # print('global x before merge:', x.shape)
            
            x, self.pad_hw = window_partition(x, self._MP_info['cur_window_size'])
            # pad_mask = torch.zeros(1,pad_hw[0],pad_hw[1],C, device=x.device, dtype=x.dtype)
            
        B,h,w,C = x.shape
        x = x.reshape(B, h*w, C)
        # print('x before merge:', x.shape)
        for sub_merge in  merge:
            x = sub_merge(x)
        self._MP_info['cur_token'] = x.shape[1]
        # print('x after merge:', x.shape)
        
        B,self.N_r, C = x.shape
        if window_size == 0:
            x = x.reshape(1, -1, C)
            # print('global x after merge:', x.shape)
            
        
        return x

    def unmerge_patch(self, x, unmerge:list):
        # window_size = self.window_size
        # if window_size == 0:
        #     _,N,C = x.shape
        #     x = x.reshape(N//self.N_r, self.N_r,C)
        
        for sub_unmerge in reversed(unmerge):
            x = sub_unmerge(x)
        
        # if window_size == 0:
        #     x = window_unpartition(x, self._MP_info['cur_window_size'], self.pad_hw, (self.H, self.W))
        #     del self.pad_hw
        
        # x = x.reshape(self.B,self.H,self.W,self.C)
        # del self.B, self.N_r, self.H, self.W, self.C
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        layer_idx = self._MP_info["selected_layers"].pop(0)
        window_size = self.window_size
        
        if window_size > 0:
            if layer_idx == self._MP_info['first_win_layer']:
        
                assert self.dim == self.dim_out

                # Window partition
                window_size = self.window_size
                H, W = x.shape[1], x.shape[2]
                x, pad_hw = window_partition(x, window_size)
                self._MP_info['pad_hw'] = pad_hw
                self._MP_info['window_size'] = window_size
                self._MP_info['HW'] = (H,W)
                B,H,W,C = x.shape
                # print(x.shape)
                x = x.reshape(B,H*W,C)
            
            elif layer_idx in self._MP_info['window_partion_layers']:
                N_w = self._MP_info['cur_win_token_num']
                B, N, C = x.shape
                x = x.reshape(N//N_w, N_w, C)
                assert x.shape[0]*x.shape[1] == N

            N = x.shape[1]
            r = int(N * self._MP_info['ratio'])
            h= w = int(math.sqrt(N))
            # print(r)
            merge, unmerge = bipartite_soft_matching_random2d(
                    x,  # 或者改用metrics（key）
                    w=w, h=h,
                    sx = 2, sy = 2,
                    r = r,
                    no_rand=False,
                    layer_idx = layer_idx,
                    generator = self._MP_info['generator']
                )
            # self._MP_function['unmerge'].append(unmerge)
            
            self._MP_info['cur_win_token_num'] = x.shape[1]
            
            # merge x
            # x, self._MP_info["size"] = merge_wavg(merge, x, self._MP_info["size"])
            # print('before merge:', x.shape)
            x = merge(x)
            # print('after merge:', x.shape)
        

            # attention block
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))

            x = unmerge(x)
            # print('after unmerge:', x.shape)
            

        else:
            if len(self._MP_info["selected_layers"])==0:
                # unmerge = self._MP_function['unmerge']
                N_m = x.shape[0] * x.shape[1]
                # x = self.unmerge_patch(x,unmerge)
                N = x.shape[0] * x.shape[1]

                x = window_unpartition(x, self._MP_info["window_size"], self._MP_info["pad_hw"], self._MP_info["HW"])
                
                # attention block
                shortcut = x
                x = self.norm1(x)
                x = self.attn(x, True)
                x = shortcut + self.drop_path(x)
                x = x + self.drop_path(self.mlp(self.norm2(x)))
                # print('prune ratio:', N_m/N)

            else:
                B, N, C = x.shape
                x = x.reshape(1, B*N, C)

                # attention block
                shortcut = x
                x = self.norm1(x)
                x = self.attn(x)
                x = shortcut + self.drop_path(x)
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            
        return x, None

                





        
           
        


class MP_MultiScaleBlock_global(MultiScaleBlock):
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
 


class MP_MultiScaleAttention(MultiScaleAttention):

    def forward(self, x: torch.Tensor, BHWC=False) -> torch.Tensor:

        if BHWC:
            B,H,W,_ = x.shape
            N = H*W
        else:
            B,N,_ = x.shape


        # B, H, W, _ = x.shape
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1)
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)
        assert self.q_pool==None

        # Q pooling (for downsample at stage changes)
        # if self.q_pool:
        #     q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
        #     H, W = q.shape[1:3]  # downsampled shape
        #     q = q.reshape(B, H * W, self.num_heads, -1)
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
        if BHWC:
            x = x.reshape(B,H,W,-1)
        else:
            x = x.reshape(B, N, -1)

        x = self.proj(x)

        return x




class MP_MultiScaleAttention_merge(MultiScaleAttention):
    # 定义merger类，由于加速效果不佳，换一种方法试试有无区别
    def build_merger(self, class_token=False, distill_token=False):
        self.merger = attn_global_merger(class_token=False, distill_token=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # 插入merge操作
        B,H,W,C = x.shape
        x = x.reshape(B,H*W,C)

        layer_idx = self._MP_info["selected_layers"].pop(0)
        merge = self.merger.patch_matching(
                    x,  # 或者改用metrics（key）
                    layer_idx,
                    self._MP_info["source"],
                    # self._MP_info["mask"]
                )
        # if self._MP_info["trace_source"]:
        #     self._MP_info["source"] = self.merger.merge_source(
        #         x, self._MP_info["source"]
        #     )
        
        # print('x before merge:', x.shape)
        x = self.merger.merge(x)
        # x, self._MP_info["size"] = self.merger.merge_wavg(x, self._MP_info["size"])
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


  
class MP_MemoryAttention(MemoryAttention):
    # 仅匹配一次patch
    def forward(self,
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

       
        
        if self.batch_first:
            # Convert to batch first
            # print('batch')
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)
        
        # print(output.shape)
        # print(memory.shape)
        # print(memory_pos.shape)
        
        w = h = int(math.sqrt(output.shape[1]))
        r = int(output.shape[1] * self._MP_info['ratio'])
        merge_tgt, unmerge_tgt,_,_ = bipartite_soft_matching_random2d(
            output,
            w=w, h=h,
            sx = 4, sy = 4,
            r=r
        )
        # merge_tgt, unmerge_tgt = do_nothing, do_nothing
        
        frame_memory, obj_memory=memory_split_reshape(memory, num_obj_ptr_tokens)
        merge_mem, unmerge_mem,_,_ = bipartite_soft_matching_random2d(
            frame_memory,
            w=w, h=h,
            sx = 4, sy = 4,
            r=r
        )
        
        B, N_mem, C = memory.shape
        # print(frame_memory.shape)
        # print(merge_mem(frame_memory).reshape(B,-1,C).shape)
        # print(obj_memory.shape)
        # exit()
        output = merge_tgt(output)
        curr_pos = merge_tgt(curr_pos)
        memory = torch.cat((merge_mem(frame_memory).reshape(B,-1,C), obj_memory), dim=1)
        mem_pos_f, mem_pos_o = memory_split_reshape(memory_pos, num_obj_ptr_tokens)
        memory_pos = torch.cat((merge_mem(mem_pos_f).reshape(B,-1,C), mem_pos_o), dim=1)

        for i, layer in enumerate(self.layers):
            # if i > 1:
            #     break
            kwds = {'merge_tgt':merge_tgt, 'unmerge_tgt':unmerge_tgt,
                    'merge_mem':merge_mem, 'unmerge_mem':unmerge_mem}
            # kwds = {'merge_tgt':do_nothing, 'unmerge_tgt':do_nothing,
            #         'merge_mem':do_nothing, 'unmerge_mem':do_nothing}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds["num_k_exclude_rope"]= num_obj_ptr_tokens

            output = layer(
                tgt=output,
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                **kwds,
            )
        
        output = unmerge_tgt(output)
        
        normed_output = self.norm(output)

        if self.batch_first:
            # Convert back to seq first
            normed_output = normed_output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)

        return normed_output


class MP_RoPEAttention_sa(RoPEAttention):
    def forward(
        self, 
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
        merge:Callable=None, unmerge:Callable=None, 
        num_k_exclude_rope: int = 0
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
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat
        # print(self.freqs_cis.shape)
        num_k_rope = k.size(-2) - num_k_exclude_rope
        # print(num_k_exclude_rope)
        # print(q.shape)
        # print(self.freqs_cis.shape)
        if self.freqs_cis.shape[0] != q.shape[2]:
            self.freqs_cis = merge(self.freqs_cis.unsqueeze(0)).squeeze(0)
        
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )
        
        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        # B,nH,N,C = q.shape
        # q,k,v = (q.reshape(-1,N,C), k.reshape(-1,N,C), v.reshape(-1,N,C))
        # q, k, v = merge(q), merge(k), merge(v)
        # q,k,v = (q.reshape(B,nH,-1,C), k.reshape(B,nH,-1,C), v.reshape(B,nH,-1,C))
        
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        # out = out.reshape(B*nH,-1,C)
        # out = unmerge(out)
        # out = out.reshape(B,nH,N,C)
            
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

class MP_RoPEAttention_ca(RoPEAttention):
    def forward(
        self, 
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
        q_merge:Callable=None, q_unmerge:Callable=None, 
        kv_merge:Callable=None, kv_unmerge:Callable=None, 
        num_k_exclude_rope: int = 0,
        token_per_frame = 4096
    ) -> torch.Tensor:
        
        # if kv_merge == None:
        #     kv_merge = q_merge
        #     kv_unmerge = q_unmerge
             
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
            
        if self.freqs_cis.shape[0] != q.shape[2]:
            self.freqs_cis = q_merge(self.freqs_cis.unsqueeze(0)).squeeze(0)
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
    
        # B,nH,N_q,C = q.shape
        # B,nH,N_k,C = k.shape
        # q,k,v = (q.reshape(-1,N_q,C), k.reshape(-1,N_k,C), v.reshape(-1,N_k,C))
        # k_f, k_o = memory_split_reshape(k, num_k_exclude_rope)
        # v_f, v_o = memory_split_reshape(k, num_k_exclude_rope)
        # q = q_merge(q)
        # k = torch.cat((kv_merge(k_f).reshape(B,-1,C), k_o), dim=1)
        # v = torch.cat((kv_merge(v_f).reshape(B,-1,C), v_o), dim=1)
        # q,k,v = (q.reshape(B,nH,-1,C), k.reshape(B,nH,-1,C), v.reshape(B,nH,-1,C))
        
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        
        # out = out.reshape(B*nH,-1,C)
        # out = q_unmerge(out)
        # out = out.reshape(B,nH,N_q,C)        
        # # out = q
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out
    
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


class MP_MemoryAttentionLayer(MemoryAttentionLayer):
    
    def _forward_sa(self, tgt, query_pos, 
                    merge=do_nothing, unmerge=do_nothing):
        # Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2, merge=merge, unmerge=unmerge)
        tgt = tgt + self.dropout1(tgt2)
        return tgt
    
    def _forward_ca(self, 
                    tgt, memory, query_pos, pos, num_k_exclude_rope=0, 
                    merge_tgt=do_nothing, unmerge_tgt=do_nothing,
                    merge_mem=do_nothing, unmerge_mem=do_nothing
                    ):
        kwds = {
                "q_merge": merge_tgt, "q_unmerge": unmerge_tgt,
                "kv_merge": merge_mem, "kv_unmerge": unmerge_mem
                }
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds["num_k_exclude_rope"] = num_k_exclude_rope
                   

        # Cross-Attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
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
        merge_tgt=do_nothing, unmerge_tgt=do_nothing,
        merge_mem=do_nothing, unmerge_mem=do_nothing
        
    ) -> torch.Tensor:
        # w = h = int(math.sqrt(tgt.shape[1]))
        # r = int(tgt.shape[1] * self._MP_info['ratio'])
        # merge_tgt, unmerge_tgt = bipartite_soft_matching_random2d(
        #     tgt,
        #     w=w, h=h,
        #     sx = 4, sy = 4,
        #     r=r
        # )
        
        # frame_memory, obj_memory=memory_split_reshape(memory, num_k_exclude_rope)
        # merge_mem, unmerge_mem = bipartite_soft_matching_random2d(
        #     frame_memory,
        #     w=w, h=h,
        #     sx = 4, sy = 4,
        #     r=r
        # )
        
        # print(tgt.shape)
        
        
        tgt = self._forward_sa(tgt, query_pos, merge_tgt, unmerge_tgt)
        # tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope = num_k_exclude_rope)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope = num_k_exclude_rope,
                               merge_tgt = merge_tgt, unmerge_tgt = unmerge_tgt,
                               merge_mem = merge_mem, unmerge_mem = unmerge_mem
                               )
        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        
        
        return tgt



def build_MP_model(args, sam_predictor, selected_layers: list, match_layers:list, trace_source: bool = False, prop_attn: bool = True):
    if args.apply_MP:
        # sam_predictor.init_memory_info()
        model = sam_predictor.image_encoder.trunk
        model.__class__ = MP_Hiera
        # setattr(model, 'trunk', MP_Hiera(model.trunk))
        # model = MP_Hiera(model)

        model.selected_layers = selected_layers
        model.match_layers = match_layers
        model.pooling_window_size = (2,2)
        model.pooling_threshold = args.pooling_threshold
        model._MP_info = {
            "size": None,
            "source": None,
            "trace_source": trace_source,
            "prop_attn": prop_attn,
            "class_token": None,
            "distill_token": False,
            "rel_pos": None,
            "selected_layers":model.selected_layers,
            "match_layers":model.match_layers,
            "pooling_window_size":model.pooling_window_size,
            "pooling_threshold":model.pooling_threshold,
            "ratio": args.merge_ratio,
            "generator": None,
        }
        model._MP_function = {}

        if hasattr(model, "dist_token") and model.dist_token is not None:
            model._MP_info["distill_token"] = True
       

        model._MP_info['selected_layers'] = args.selected_layers
        if 'base' in args.sam2_model:
            model._MP_info['first_win_layer'] = 6
            model._MP_info['window_partion_layers'] = [13, 17]
        else:
            model._MP_info['first_win_layer'] = 9
            model._MP_info['window_partion_layers'] = [24, 34]

        model._MP_info["generator"] = init_generator(next(model.parameters()))

        if args.merge_method == 'ToMe':
            MP_blk = MP_MultiScaleBlock_ToMe
        elif args.merge_method == 'ALGM':
            MP_blk = MP_MultiScaleBlock_ALGM
        elif args.merge_method == 'ToMe4DM':
            MP_blk = MP_MultiScaleBlock_ToMe4DM
        else:
            raise NotImplementedError(
            'check args.merge_method'
        )

        len_att = 0
        len_block = 0
        for module in model.modules():
            if isinstance(module, MultiScaleBlock):
                if len_block in model.selected_layers:
                    module.__class__ = MP_blk
                    module._MP_info = model._MP_info
                    module._MP_function = model._MP_function
                    module.token_match = True if len_block in model.match_layers else False
                len_block +=1 
                
            elif isinstance(module, MultiScaleAttention):
                if len_att in model.selected_layers: 
                    module.__class__ = MP_MultiScaleAttention
                    module._MP_info = model._MP_info
                    module._MP_function = model._MP_function
                len_att +=1 


    if args.prune_memory:
        mem_attn_module = sam_predictor.memory_attention
        mem_attn_module.__class__ = MP_MemoryAttention
        mem_attn_module._MP_info = {
            "size": None,
            "source": None,
            "trace_source": trace_source,
            "prop_attn": prop_attn,
            "class_token": None,
            "distill_token": False,
            "rel_pos": None,
            "ratio": 0.8,
            "generator": None
        }
        for name, module in mem_attn_module.named_modules():
            if isinstance(module, MemoryAttentionLayer):
                # if len_att in mem_attn_module.selected_layers:
                module.__class__ = MP_MemoryAttentionLayer
                module._MP_info = mem_attn_module._MP_info
            elif isinstance(module, RoPEAttention):
                # if len_block in mem_attn_module.selected_layers: 
                if 'self_attn' in  name:
                    module.__class__ = MP_RoPEAttention_sa
                    module._MP_info = mem_attn_module._MP_info
                if 'cross_attn_image' in name:
                    module.__class__ = MP_RoPEAttention_ca
                    module._MP_info = mem_attn_module._MP_info
                    
                    
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from torch import nn, Tensor
import math



class StraightForward(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print('straight forward...')
        return x
    

class BottleneckAdapter(nn.Module):
    """
    基础的瓶颈结构Adapter
    """
    def __init__(self, input_dim: int, bottleneck_dim: int, dropout: float = 0.1):
        super().__init__()
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        self.up_project = nn.Linear(bottleneck_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(normalized_shape=input_dim, eps=1e-6)
        # print(input_dim, bottleneck_dim)
        
        
        # 初始化权重，使adapter初始时接近恒等映射
        nn.init.kaiming_normal_(
            self.down_project.weight, 
            mode='fan_in', 
            nonlinearity='relu'
        )
        # nn.init.normal_(self.down_project.weight, std=1e-3)
        nn.init.normal_(self.up_project.weight, std=1e-3)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print('bypassing......')
        # print(x.shape) # B,H,W,C
        # exit()
        residual = x
        x = self.norm(x)
        x = self.down_project(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_project(x)
        # print(x.shape)
        return x + residual  # 残差连接
        # return x
        
class FFNAdapter(nn.Module):
    """
    基础的FFN Adapter
    """
    def __init__(self, input_dim: int, internal_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, internal_dim)
        self.output_proj = nn.Linear(internal_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(normalized_shape=input_dim, eps=1e-6)
        
        self._init_weights()
        
    def _init_weights(self):
        # Kaiming初始化 for input_proj (配合ReLU)
        nn.init.kaiming_normal_(
            self.input_proj.weight, 
            mode='fan_in', 
            nonlinearity='relu'
        )
        
        # 小方差初始化 for output_proj (adapter特性)
        nn.init.normal_(self.output_proj.weight, std=1e-3)
        
        # 偏置初始化
        nn.init.zeros_(self.input_proj.bias)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print('bypassing......')
        # print(x.shape)    # B,H,W,C
        residual = x
        x = self.norm(x)
        x = self.input_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output_proj(x)
        # print(x.shape)
        return x + residual  # 残差连接
    
class AttentionAdapter(nn.Module):
    """
    注意力结构Adapter
    """
    def __init__(
        self,
        input_dim: int,
        adapter_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.adapter_dim = adapter_dim
        self.num_heads = num_heads
        self.head_dim = self.adapter_dim// num_heads
        
        assert (
            self.adapter_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."
        # 降维投影
        self.down_project = nn.Linear(self.input_dim, self.adapter_dim)
        # 注意力线性层
        self.qkv = nn.Linear(self.adapter_dim, 3*self.adapter_dim)
        
        # 升维投影
        self.up_project = nn.Linear(self.adapter_dim, self.input_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim, eps=1e-6)
        self.scale = math.sqrt(self.head_dim)
        self.dropout_p = dropout
        
        self._init_weights_km()
    
    def _init_weights(self):
        nn.init.normal_(self.down_project.weight, std=1e-3)
        nn.init.normal_(self.up_project.weight, std=1e-3)
        nn.init.normal_(self.qkv.weight, std=1e-3)
        
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.bias)
    
    def _init_weights_km(self):
    # Kaiming初始化 - 适用于深度网络
        nn.init.kaiming_normal_(self.down_project.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.up_project.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.qkv.weight, mode='fan_in', nonlinearity='linear')
        
        # 偏置项初始化为0
        if self.down_project.bias is not None:
            nn.init.zeros_(self.down_project.bias)
        if self.up_project.bias is not None:
            nn.init.zeros_(self.up_project.bias)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)


    def forward(self, x: Tensor) -> Tensor:
        # print(x.shape)
        # if x.shape[0]==0:
        #     return x
        residual= x
        
        x = self.norm(x)
        
        x = self.down_project(x)
        
        B,H,W,C = x.shape
        
        # Input projections
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, self.head_dim)
        q, k, v = torch.unbind(qkv, 2)
        # print(q.shape)
        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        attn_weight = q @ k.transpose(-2, -1) * self.scale
        attn_weight_ = torch.softmax(attn_weight, dim=-1)
        attn_weight_ = torch.dropout(attn_weight_, dropout_p, train=self.training)
        out = attn_weight_ @ v
        
        # x = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p)

        out = out.transpose(1,2)
        out = out.reshape(B,H,W,C)
        
        out = self.up_project(out)
        out = self.dropout(out)

        return out + residual


    
    
def build_bypass_model(args, predictor, adapter_type='bottleneck', adapter=True, training=False):
    if adapter == True:
        if adapter_type=='bottleneck':
            predictor.image_encoder.trunk.blocks[args.final_global_layer].bypass_branch = BottleneckAdapter(input_dim=args.inner_channel, bottleneck_dim=args.inner_channel//2)
        elif adapter_type == 'attention':
            predictor.image_encoder.trunk.blocks[args.final_global_layer].bypass_branch = AttentionAdapter(input_dim=args.inner_channel, adapter_dim=args.inner_channel//2, num_heads=4)
        elif adapter_type == 'FFN':
            predictor.image_encoder.trunk.blocks[args.final_global_layer].bypass_branch = FFNAdapter(input_dim=args.inner_channel, internal_dim=args.inner_channel*2)
            
        predictor.image_encoder.trunk.blocks[args.final_global_layer].bypass_branch.to(device=predictor.device)
        if training:
            predictor.image_encoder.trunk.blocks[args.final_global_layer].bypass_branch.train()
    else:
        predictor.image_encoder.trunk.blocks[args.final_global_layer].bypass_branch = StraightForward()

    
    return

def build_small_bypass_model(args, predictor, bypass_layers, training=False):
    for i in bypass_layers:
        predictor.image_encoder.trunk.blocks[i].bypass_branch = BottleneckAdapter(input_dim=448, bottleneck_dim=224)
        predictor.image_encoder.trunk.blocks[i].bypass_branch.to(device=predictor.device)
        if training:
            predictor.image_encoder.trunk.blocks[i].bypass_branch.train()
    
    return
import math
from typing import Callable, Tuple
import torch

class attn_global_merger():
    def __init__(self, class_token: bool = False, distill_token: bool = False):
        self.class_token=class_token
        self.distill_token=distill_token

    def patch_matching(
        self,
        metric: torch.Tensor,
        layer_idx:int,
        source: torch.Tensor,

    ):
        # print(metric.shape)
        # exit()
        protected = 0
        if self.class_token:
            protected += 1
        if self.distill_token:
            protected += 1

        self.ori_shape = metric.shape
        t = metric.shape[1]
        r = (t - protected) // 2

        if r <= 0:
            return

        with torch.no_grad():

            # B,m_t,um_t = source.shape
            metric = metric / metric.norm(dim=-1, keepdim=True)
            a, b = metric[..., ::2, :], metric[..., 1::2, :]
            scores = a @ b.transpose(-1, -2)
            # print('scores:', scores.shape)
        
            if self.class_token:
                scores[..., 0, :] = -math.inf
            if self.distill_token:
                scores[..., :, 0] = -math.inf


            node_max, node_idx = scores.max(dim=-1) # 为每个source保留最大相似度的匹配。 node_max: A token的最大相似度，node_idx: 对应的B token的索引
            # print('node_max, node_idx:', node_max.shape, node_idx.shape)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None] # 根据相似度排序match。 edge_idx: A token按照相似度大小排序的索引

            # ------------------ start  addaptive section --------- 
            # i = layer_idx
            # n_B, n_H = node_max.shape
            # node_mean= torch.add(node_max[:,1:].mean(dim=1).mean(),node_max[:,1:].std(dim=1).mean()/i)
            # node_mean=node_mean.repeat(1,n_H)
            # self.r = torch.ge(node_max, node_mean).sum(dim=1).min()
            if layer_idx <3:
                self.r = 20
            elif layer_idx in [4,5]:
                self.r = 5
            elif layer_idx in [13,17,21]:
                self.r = 2000
            else:
                self.r=90
            # ------------------ end addaptive section --------- 
            
            self.unm_idx = edge_idx[..., self.r:, :]  # Unmerged Tokens
            self.src_idx = edge_idx[..., :self.r, :]  # Merged Tokens 选出src token（坐标）
            self.dst_idx = node_idx[..., None].gather(dim=-2, index=self.src_idx) # 从B token中抽出与src token匹配的dst token(坐标)

            if self.class_token:
                # Sort to ensure the class token is at the start
                self.unm_idx = self.unm_idx.sort(dim=1)[0]
    
    def merge(self, x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=self.unm_idx.expand(n, t1 - self.r, c))
        src = src.gather(dim=-2, index=self.src_idx.expand(n, self.r, c))
        dst = dst.scatter_reduce(-2, self.dst_idx.expand(n, self.r, c), src, reduce=mode)

        if self.distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)
    
    def unmerge(self, x: torch.Tensor) -> torch.Tensor:
        unm_len = self.unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=self.dst_idx.expand(n, self.r, c))

        out = torch.zeros(n, self.ori_shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * self.unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * self.src_idx).expand(n, self.r, c), src=src)

        return out
    
    def merge_source(
        self, x: torch.Tensor, source: torch.Tensor = None
    ) -> torch.Tensor:

        if source is None:
            n, t, _ = x.shape
            source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)
        # print('source berfore merge:',source.shape)
        source = self.merge(source, mode="amax")
        # print('source after merge:',source.shape)
        return source
    
    def merge_wavg(
       self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if size is None:
            size = torch.ones_like(x[..., 0, None])

        x = self.merge(x * size, mode="sum")
        size = self.merge(size, mode="sum")    
        x = x / size
        
        return x, size

class global_merger_1():
    def __init__(self, class_token: bool = False, distill_token: bool = False):
        self.class_token=class_token
        self.distill_token=distill_token

    def patch_matching(
        self,
        metric: torch.Tensor,
        layer_idx:int,
        source: torch.Tensor,

    ):
        # print(metric.shape)
        # exit()
        protected = 0
        if self.class_token:
            protected += 1
        if self.distill_token:
            protected += 1

        t = metric.shape[1]
        r = (t - protected) // 2

        if r <= 0:
            return

        with torch.no_grad():

            # B,m_t,um_t = source.shape
            metric = metric / metric.norm(dim=-1, keepdim=True)
            a, b = metric[..., ::2, :], metric[..., 1::2, :]
            scores = a @ b.transpose(-1, -2)
            # print('scores:', scores.shape)
        
            if self.class_token:
                scores[..., 0, :] = -math.inf
            if self.distill_token:
                scores[..., :, 0] = -math.inf


            node_max, node_idx = scores.max(dim=-1) # 为每个source保留最大相似度的匹配。 node_max: A token的最大相似度，node_idx: 对应的B token的索引
            print('node_max, node_idx:', node_max.shape, node_idx.shape)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None] # 根据相似度排序match。 edge_idx: A token按照相似度大小排序的索引

            # ------------------ start  addaptive section --------- 
            i = layer_idx
            n_B, n_H = node_max.shape
            node_mean= torch.add(node_max[:,1:].mean(dim=1).mean(),node_max[:,1:].std(dim=1).mean()/i)
            node_mean=node_mean.repeat(1,n_H)
            self.r = torch.ge(node_max, node_mean).sum(dim=1).min()

            # ------------------ end addaptive section --------- 
            
            self.unm_idx = edge_idx[..., self.r:, :]  # Unmerged Tokens
            self.src_idx = edge_idx[..., :self.r, :]  # Merged Tokens 选出src token（坐标）
            self.dst_idx = node_idx[..., None].gather(dim=-2, index=self.src_idx) # 从B token中抽出与src token匹配的dst token(坐标)

            if self.class_token:
                # Sort to ensure the class token is at the start
                self.unm_idx = self.unm_idx.sort(dim=1)[0]
    
    def merge(self, x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=self.unm_idx.expand(n, t1 - self.r, c))
        src = src.gather(dim=-2, index=self.src_idx.expand(n, self.r, c))
        dst = dst.scatter_reduce(-2, self.dst_idx.expand(n, self.r, c), src, reduce=mode)

        if self.distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)
    
    def merge_source(
        self, x: torch.Tensor, source: torch.Tensor = None
    ) -> torch.Tensor:

        if source is None:
            n, t, _ = x.shape
            source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)
        print('source berfore merge:',source.shape)
        source = self.merge(source, mode="amax")
        print('source after merge:',source.shape)
        return source
    
    def merge_wavg(
       self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if size is None:
            size = torch.ones_like(x[..., 0, None])

        x = self.merge(x * size, mode="sum")
        size = self.merge(size, mode="sum")    
        x = x / size
        
        return x, size

class global_merger_2():
    def __init__(self, class_token: bool = False, distill_token: bool = False):
        self.class_token = class_token
        self.distill_token = distill_token

    def patch_matching(
        self,
        metric: torch.Tensor,
        layer_idx: int,
        source: torch.Tensor,
    ):
        protected = 0
        if self.class_token:
            protected += 1
        if self.distill_token:
            protected += 1

        t = metric.shape[1]
        r = (t - protected) // 2

        if r <= 0:
            return

        with torch.no_grad():
            metric = metric / metric.norm(dim=-1, keepdim=True)
            a, b = metric[..., ::2, :], metric[..., 1::2, :]  # a,b: [B,N/2,C]
            scores = a @ b.transpose(-1, -2)  # scores: [B,N/2,N/2]
        
            if self.class_token:
                scores[..., 0, :] = -math.inf
            if self.distill_token:
                scores[..., :, 0] = -math.inf

            node_max, node_idx = scores.max(dim=-1)  # node_max, node_idx: [B,N/2]
            
            # ------------------ start adaptive section ---------
            # i = layer_idx
            # n_B, n_H = node_max.shape  # n_B: batch_size, n_H: N/2
            # node_mean = torch.add(node_max[:,1:].mean(dim=1).mean(), node_max[:,1:].std(dim=1).mean()/i)
            # node_mean = node_mean.repeat(1,n_H)  # [B,N/2]
            # self.r = torch.ge(node_max, node_mean).sum(dim=1).min()  # scalar
            self.r = 8
            # ------------------ end adaptive section ---------
            
            # 获取要合并的token的索引
            edge_idx = node_max.argsort(dim=-1, descending=True)  # [B,N/2]
            self.src_idx = edge_idx[..., :self.r]  # [B,r]
            self.dst_idx = node_idx.gather(dim=-1, index=self.src_idx)  # [B,r]
            
            # 将src_idx和dst_idx转换为原始tensor中的实际索引
            self.src_idx_full = self.src_idx * 2  # [B,r]
            self.dst_idx_full = self.dst_idx * 2 + 1  # [B,r]
            
    def merge(self, x: torch.Tensor, mode="mean") -> torch.Tensor:
        n, t, c = x.shape  # [B,N,C]
        output = x.clone()
        
        # 扩展索引以匹配channel维度
        src_idx_exp = self.src_idx_full.unsqueeze(-1).expand(-1, -1, c)  # [B,r,C]
        dst_idx_exp = self.dst_idx_full.unsqueeze(-1).expand(-1, -1, c)  # [B,r,C]
        
        # 获取src tokens的值
        src_tokens = torch.gather(x, 1, src_idx_exp)  # [B,r,C]
        
        # 将src tokens合并到dst位置
        if mode == "mean":
            output.scatter_add_(1, dst_idx_exp, src_tokens)
            # 计算每个dst位置的累加次数
            counts = torch.zeros_like(output)
            counts.scatter_add_(1, dst_idx_exp, torch.ones_like(src_tokens))
            # 避免除零
            counts = counts.clamp(min=1.0)
            output = output / counts
        elif mode == "sum":
            output.scatter_add_(1, dst_idx_exp, src_tokens)
        elif mode == "amax":
            output.scatter_reduce_(1, dst_idx_exp, src_tokens, reduce="amax")
            
        # 将src位置置零
        output.scatter_(1, src_idx_exp, torch.zeros_like(src_tokens))
        
        return output

    # def merge_source(self, x: torch.Tensor, source: torch.Tensor = None) -> torch.Tensor:
    #     if source is None:
    #         n, t, _ = x.shape
    #         source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)
    #     source = self.merge(source, mode="amax")
    #     return source
    
    def merge_source(self, x: torch.Tensor, source: torch.Tensor = None) -> torch.Tensor:
        """
        更新source矩阵以追踪token合并历史
        Args:
            x: 输入tensor [B,N,C]
            source: 源追踪矩阵 [B,N,N]，如果为None则初始化为单位矩阵
        Returns:
            更新后的source矩阵 [B,N,N]
        """
        if source is None:
            n, t, _ = x.shape
            source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)
        
        output = source.clone()
        
        # 对每个batch处理
        for b in range(source.shape[0]):
            # 对于每个要合并的token对
            for src, dst in zip(self.src_idx_full[b], self.dst_idx_full[b]):
                # 将src位置的所有贡献累加到dst位置
                output[b, :, dst] = output[b, :, dst] + output[b, :, src]
                # 将src位置的贡献清零
                output[b, :, src] = 0
        
        return output
    
    def merge_wavg(self, x: torch.Tensor, size: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if size is None:
            size = torch.ones_like(x[..., 0, None])
            
        x = self.merge(x * size, mode="sum")
        size = self.merge(size, mode="sum")    
        x = x / size
        
        return x, size

class global_merger():
    def __init__(self, class_token: bool = False, distill_token: bool = False):
        self.class_token = class_token
        self.distill_token = distill_token
        # self.mask = None  # 添加mask属性

    def patch_matching(
        self,
        metric: torch.Tensor,
        layer_idx: int,
        source: torch.Tensor,
        mask: torch.Tensor
    ):
        self.mask = mask
        protected = 0
        if self.class_token:
            protected += 1
        if self.distill_token:
            protected += 1

        t = metric.shape[1]
        r = (t - protected) // 2

        if r <= 0:
            return

        with torch.no_grad():
            metric = metric / metric.norm(dim=-1, keepdim=True)
            a, b = metric[..., ::2, :], metric[..., 1::2, :]
            scores = a @ b.transpose(-1, -2)
        
            if self.class_token:
                scores[..., 0, :] = -math.inf
            if self.distill_token:
                scores[..., :, 0] = -math.inf

            node_max, node_idx = scores.max(dim=-1)
            
            self.r = 8
            
            edge_idx = node_max.argsort(dim=-1, descending=True)
            self.src_idx = edge_idx[..., :self.r]
            self.dst_idx = node_idx.gather(dim=-1, index=self.src_idx)
            
            self.src_idx_full = self.src_idx * 2
            self.dst_idx_full = self.dst_idx * 2 + 1
            
            # 初始化mask
            if self.mask == None:
                self.mask = torch.ones(metric.shape[0], t, device=metric.device)
            # 将src位置的mask置零
            self.mask.scatter_(1, self.src_idx_full, torch.zeros_like(self.src_idx_full, dtype=torch.float))
        return self.mask

    def merge(self, x: torch.Tensor, mode="mean") -> torch.Tensor:
        """
        使用mask进行merge操作
        Args:
            x: 输入tensor [B,N,C]
            mode: merge模式 ("mean", "sum", "amax")
        Returns:
            merged tensor [B,N,C]
        """
        if not hasattr(self, 'mask') or self.mask is None:
            return x

        n, t, c = x.shape
        output = x.clone()
        
        # 扩展索引以匹配channel维度
        src_idx_exp = self.src_idx_full.unsqueeze(-1).expand(-1, -1, c)
        dst_idx_exp = self.dst_idx_full.unsqueeze(-1).expand(-1, -1, c)
        
        # 获取src tokens的值
        src_tokens = torch.gather(x, 1, src_idx_exp)
        
        # 将src tokens合并到dst位置
        if mode == "mean":
            output.scatter_add_(1, dst_idx_exp, src_tokens)
            # 计算每个dst位置的累加次数
            counts = torch.zeros_like(output)
            counts.scatter_add_(1, dst_idx_exp, torch.ones_like(src_tokens))
            # 避免除零
            counts = counts.clamp(min=1.0)
            output = output / counts
        elif mode == "sum":
            output.scatter_add_(1, dst_idx_exp, src_tokens)
        elif mode == "amax":
            output.scatter_reduce_(1, dst_idx_exp, src_tokens, reduce="amax")
        
        # 使用mask来控制输出
        # mask_exp = self.mask.unsqueeze(-1).expand_as(output)
        # output = output * mask_exp
        
        return output

    def merge_source(self, x: torch.Tensor, source: torch.Tensor = None) -> torch.Tensor:
        if source is None:
            n, t, _ = x.shape
            source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)
        
        output = source.clone()
        
        # 对每个batch处理
        for b in range(source.shape[0]):
            for src, dst in zip(self.src_idx_full[b], self.dst_idx_full[b]):
                # 将src位置的所有贡献累加到dst位置
                output[b, :, dst] = output[b, :, dst] + output[b, :, src]
                # 使用mask来控制source矩阵
                output[b, :, src] = output[b, :, src] * 0
        
        return output
    
    def merge_wavg(self, x: torch.Tensor, size: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用mask进行加权平均merge
        Args:
            x: 输入tensor [B,N,C]
            size: 权重tensor [B,N,1]
        Returns:
            merged_x: 合并后的tensor [B,N,C]
            merged_size: 合并后的权重 [B,N,1]
        """
        if size is None:
            size = torch.ones_like(x[..., 0, None])
            
        weighted_x = self.merge(x * size, mode="sum")
        merged_size = self.merge(size, mode="sum")
        
        # 避免除零
        eps = 1e-6
        merged_size = torch.clamp(merged_size, min=eps)
        merged_x = weighted_x / merged_size
        
        return merged_x, merged_size

    # def get_mask(self) -> torch.Tensor:
    #     """
    #     获取当前的mask
    #     Returns:
    #         mask tensor [B,N]
    #     """
    #     if not hasattr(self, 'mask') or self.mask is None:
    #         return None
    #     return self.mask.clone()


def restore(x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """
    利用source矩阵恢复被合并的tokens
    Args:
        x: 合并后的tensor，形状为[B,N,C]，某些位置为0（原src位置）
        source: 记录合并关系的矩阵，形状为[B,N,N]
    Returns:
        restored: 恢复后的tensor，形状为[B,N,C]
    """
    B, N, C = x.shape
    restored = x.clone()
    
    # source矩阵中，每一列的非零元素表示该位置的token来自哪些位置的合并
    # 找到每一列中最大值所在的行索引，即找到该位置的token主要来源
    source_indices = source.argmax(dim=1)  # [B,N]
    
    # 扩展索引以匹配channel维度
    source_indices = source_indices.unsqueeze(-1).expand(-1, -1, C)  # [B,N,C]
    
    # 对于source中非零的位置（表示这些位置被合并过）
    # 从对应的dst位置复制值到这些位置
    mask = (source > 0).any(dim=1)  # [B,N]
    mask = mask.unsqueeze(-1).expand(-1, -1, C)  # [B,N,C]
    
    # 根据source_indices从x中获取对应位置的值
    gathered_values = torch.gather(x, 1, source_indices)  # [B,N,C]
    
    # 只更新被合并过的位置（mask为True的位置）
    restored = torch.where(mask, gathered_values, restored)
    
    return restored


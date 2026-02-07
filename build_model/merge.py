import math
from typing import Callable, Tuple
import torch
from einops import rearrange
import torch.nn.functional as F


def do_nothing(x, mode=None):
    return x


def conditional_pooling(
    feat: torch.Tensor,
    threshold:float,
    window_size: Tuple[int, int],
) -> Tuple[Callable, Callable]:
    
    with torch.no_grad():
        
        ws_h, ws_w = int(window_size[0]), int(window_size[1])
        stride_h, stride_w = ws_h, ws_w
        window_topken_num = stride_h * stride_w
        
        # x_cls, feat = feat[:, :1, :], feat[:, 1:, :]
        B, N, D = feat.size()
        base_grid_H = int(math.sqrt(N))
        base_grid_W = base_grid_H
        assert base_grid_H * base_grid_W == N and base_grid_H % ws_h == 0 and base_grid_W % ws_w == 0

        feat = rearrange(feat, "b (h w) c -> b c h w", h=base_grid_H)
    
        feat = rearrange(feat, 'b c (gh ps_h) (gw ps_w) -> b gh gw c ps_h ps_w', gh=base_grid_H//ws_h, gw=base_grid_W//ws_w)
        b, gh, gw, c, ps_h, ps_w = feat.shape       # [1,32,32,488,2,2]

        # Flatten mxm window for pairwise operations
        tensor_flattened = feat.reshape(b, gh, gw, c, -1)
    

        # Expand dims for pairwise operations
        tensor_1 = tensor_flattened.unsqueeze(-1)
        tensor_2 = tensor_flattened.unsqueeze(-2)

        # Compute cosine similarities
        sims = F.cosine_similarity(tensor_1, tensor_2, dim=3)

        # Exclude the self-similarity (i.e., similarity with oneself will be 1)
        sims_mask = 1 - torch.eye(ps_h * ps_w).to(sims.device)
        sims = sims * sims_mask

        # Average similarities (excluding the self-similarity)
        similarity_map = sims.sum(-1).sum(-1) / ((ps_h * ps_w) * (ps_h * ps_w - 1))
        
        similarity_map = rearrange(similarity_map.unsqueeze(1), 'b c h w-> b (c h w)')
        
        #--- adaptive section ---#
     
        n_B, n_H = similarity_map.shape
        node_mean = torch.tensor(threshold).cuda(sims.device)
        node_mean=node_mean.repeat(1,n_H)
        r = torch.ge(similarity_map, node_mean).sum(dim=1).min()
        # -------------# 
    
        #   get top k similar super patches 
        _, sim_super_patch_idxs = similarity_map.topk(r,dim=-1) #[1,r]
        # print(similarity_map.topk(r,dim=-1))
        # print(sim_super_patch_idxs.shape)
    
        # --- creating the mergabel and unmergable super  pathes
        tensor = torch.arange(base_grid_H * base_grid_W).reshape(base_grid_H, base_grid_W).to(feat.device)

        # Repeat the tensor to create a batch of size 2
        tensor = tensor.unsqueeze(0).repeat(B, 1, 1)
        # print(tensor.shape)

        # Apply unfold operation on last two dimensions to create the sliding window
        windowed_tensor = tensor.unfold(1, ws_h, stride_h).unfold(2, ws_w, stride_w) # [1,32,32,2,2]
        # print(tensor.unfold(1, ws_h, stride_h).shape)
        # print(windowed_tensor.shape)
        # Reshape the tensor to the desired shape 
        windowed_tensor = windowed_tensor.reshape(B, -1, window_topken_num)  # [1,1024,4]
        # print(windowed_tensor.shape)
    
        # Use torch.gather to collect the desired elements
        gathered_tensor = torch.gather(windowed_tensor, 1, sim_super_patch_idxs.unsqueeze(-1).expand(-1, -1, window_topken_num))
        # [1,r,4]
        # print(sim_super_patch_idxs.unsqueeze(-1).expand(-1, -1, window_topken_num).shape)
        # print(gathered_tensor.shape)
        # print(gathered_tensor)
        # 到这里主要是想把网格坐标变换成原始坐标


        # Create a mask for all indices, for each batch
        mask = torch.ones((B, windowed_tensor.shape[1]), dtype=bool).to(feat.device)    # [1,1024]

        # Create a tensor that matches the shape of indices and fill it with False
        mask_values = torch.zeros_like(sim_super_patch_idxs, dtype=torch.bool).to(feat.device)  # [1,166]
        # print(mask.shape, mask_values.shape)
        # Use scatter_ to update the mask. This will set mask[b, indices[b]] = False for all b
        # 把选出来的窗口mask掉
        mask.scatter_(1, sim_super_patch_idxs, mask_values) # [1,1024]

        # Get the remaining tensor
        remaining_tensor = windowed_tensor[mask.unsqueeze(-1).expand(-1, -1, window_topken_num)].reshape(B, -1, window_topken_num)
        # print(remaining_tensor.shape)
        # [1, 1024-r, 4]
        unm_idx = remaining_tensor.reshape(B, -1).sort(dim=-1).values.unsqueeze(-1)
        # [1,4*(1024-r)]
        # print(unm_idx.shape)
        dim_index = (window_topken_num)- 1 
        src_idx= gathered_tensor[:, :, :dim_index].reshape(B, -1).unsqueeze(-1) # [B,3*r,1]
        dst_idx= gathered_tensor[:, :, dim_index].reshape(B, -1).unsqueeze(-1)  # [B,r,1]
        merge_idx = torch.arange(src_idx.shape[1]//dim_index).repeat_interleave(dim_index).repeat(B, 1).unsqueeze(-1).to(feat.device)
        # print(torch.arange(src_idx.shape[1]//dim_index).repeat_interleave(dim_index).shape)
        # print(unm_idx.shape,src_idx.shape,dst_idx.shape,merge_idx.shape)
        # print('reduction token num:',src_idx.shape[1])
        # print('reduction token num:',src_idx.shape[1])
        
        # exit()

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
       # TODO: window_topken_num can be undefined
        x_feat=x
        # x_cls , x_feat =  x[:, :1, :], x[:, 1:, :]
        n, t1, c = x_feat.shape
        src = x_feat.gather(dim=-2, index=src_idx.expand(n, r*dim_index, c))
        dst = x_feat.gather(dim=-2, index=dst_idx.expand(n, r, c))
        unm = x_feat.gather(dim=-2, index=unm_idx.expand(n, t1 - (r*window_topken_num), c))
        dst = dst.scatter_reduce(-2, merge_idx.expand(n,r*dim_index, c), src, reduce=mode)
        x = torch.cat([dst, unm], dim=1)
        # x = torch.cat((x_cls, x), dim=1)
        return x
    
    def unmerge(x: torch.Tensor, mode="copy") -> torch.Tensor:
        
        unm_len = unm_idx.shape[1]
        r = dst_idx.shape[1]
        dst, unm=x[..., :r,:], x[..., r:, :]
        n,_,c = unm.shape
        
        if mode == "copy":
            src = dst.gather(dim=-2, index=merge_idx.expand(n, r * dim_index, c))
        elif mode == "zero":
            src = torch.zeros((n, r * dim_index, c), device=x.device)
        
        out = torch.zeros(n, N, c, device=x.device, dtype=x.dtype)
        # print(out.shape)
        out.scatter_(dim=-2, index=src_idx.expand(n, r*dim_index, c), src=src)
        out.scatter_(dim=-2, index=dst_idx.expand(n,r,c),src=dst)
        out.scatter_(dim=-2, index=unm_idx.expand(n,unm_len,c), src=unm)
        # print(out.shape)
        # exit()
        # print(x.shape)
        return out

    return merge, unmerge


def bipartite_soft_matching_random2d(
    metric: torch.Tensor,
    w: int, h: int,
    sx: int, sy: int,
    r: int, 
    layer_idx: int=None,
    no_rand: bool=True,
    generator: torch.Generator = None,
    class_token: bool=False,
    distill_token: bool = False,
)-> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.
    """
    B, N, C = metric.shape
    if r <= 0:
        return do_nothing, do_nothing
    
    with torch.no_grad():
        hsy, wsx = h//sy, w//sy
        
        if no_rand:
            rand_idx = torch.zeros(B, hsy, wsx, 1,device=metric.device, dtype=torch.int64)
        else:
            # rand_idx = torch.randint(sy*sx, size=(B, hsy, wsx, 1), device=generator.device, generator=generator).to(metric.device)
            rand_idx = torch.randint(sy*sx, size=(B, hsy, wsx, 1)).to(metric.device)

        idx_buffer_view = torch.zeros(B, hsy, wsx, sy*sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=3, index = rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.view(B, hsy,wsx,sy,sx).transpose(2,3).reshape(B, hsy*sy,wsx*sx)
        
        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(B, h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:, :(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view
            
        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
        rand_idx = idx_buffer.reshape(B, -1, 1).argsort(dim=1)
        # We're finished with these
        del idx_buffer, idx_buffer_view
        
        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:, :]    # src
        b_idx = rand_idx[:, :num_dst, :]    # dst
        
        def split(x):
            C = x.shape[-1]
            src = torch.gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = torch.gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst
        
        # 归一化 metric 并计算余弦相似度
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)
        
        # 限制合并的最大令牌数
        r = min(a.shape[1], r)
        
        # 贪心匹配，按相似度排序
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # 未合并的令牌
        src_idx = edge_idx[..., :r, :]  # 合并的令牌
        dst_idx = torch.gather(node_idx[..., None], dim=-2, index=src_idx)
        
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        
        src, dst = split(x)
        n, t1, c = src.shape

        unm = torch.gather(src, dim=-2, index=unm_idx.expand(B, t1 - r, c))
        src = torch.gather(src, dim=-2, index=src_idx.expand(B, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(B, r, c), src, reduce=mode)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        src = torch.gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

        # 还原到原始形状
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=torch.gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=torch.gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c), src=src)

        return out
    
    def prune(x: torch.Tensor) -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape

        unm = torch.gather(src, dim=-2, index=unm_idx.expand(B, t1 - r, c))
        # 删除 src_idx 的令牌
        return torch.cat([unm, dst], dim=1)

    def restore(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        src = torch.gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

        # 还原到原始形状
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=torch.gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=torch.gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c), src=src)

        return out

    return merge, unmerge
        
def bipartite_soft_matching(
    metric: torch.Tensor,
    layer_idx: int,
    r: int,
    class_token: bool = False,
    distill_token: bool = False
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    # print('matching.........')
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1
    
    B,N,C = metric.shape
    # print(metric)
    # exit()
    # if layer_idx <3:
    #     r = 20
    # elif layer_idx in [4,5]:
    #     r = 5
    # elif layer_idx in [13,17,21]:
    #     r = 2000
    # else:
    #     r=90
        
    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric_ = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric_[..., ::2, :], metric_[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)
        scores[scores.isnan()] = float('-inf')

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]
    
    # print(metric)
    # exit()

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out
    
    def prune(x: torch.Tensor, mode="mean") -> torch.Tensor:
        
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        # src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        # dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)
        
    def restore(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        # src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        # out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge
    # return do_nothing, do_nothing
    
def bipartite_soft_matching_multi(
    metric: torch.Tensor,
    layer_idx: int,
    r: int,
    class_token: bool = False,
    distill_token: bool = False
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    # print('matching.........')
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1
    
    B,N,C = metric.shape
    # if layer_idx <3:
    #     r = 20
    # elif layer_idx in [4,5]:
    #     r = 5
    # elif layer_idx in [13,17,21]:
    #     r = 2000
    # else:
    #     r=90
        
    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out
    
    def prune(x: torch.Tensor, mode="mean") -> torch.Tensor:
        
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        # src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        # dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)
        
    def restore(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        # src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        # out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge, prune, restore

def ALGM_global_patch_matching(
    metric: torch.Tensor,
    layer_idx:int,
    # source: torch.Tensor,
    class_token: bool = False,
    distill_token: bool = False,

):
    # print(metric.shape)
    # exit()
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1
        
    ori_shape = metric.shape

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
    
        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf


        node_max, node_idx = scores.max(dim=-1) # 为每个source保留最大相似度的匹配。 node_max: A token的最大相似度，node_idx: 对应的B token的索引
        # print('node_max, node_idx:', node_max.shape, node_idx.shape)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None] # 根据相似度排序match。 edge_idx: A token按照相似度大小排序的索引

        # ------------------ start  addaptive section --------- 
        i = layer_idx
        n_B, n_H = node_max.shape
        node_mean= torch.add(node_max[:,1:].mean(dim=1).mean(),node_max[:,1:].std(dim=1).mean()/i)
        # node_mean= torch.add(node_max[:,1:].mean(dim=1).mean(),node_max[:,1:].std(dim=1).mean())
        node_mean=node_mean.repeat(1,n_H)
        r = torch.ge(node_max, node_mean).sum(dim=1).min()
        # print(r)
        # ------------------ end addaptive section --------- 
        
        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens 选出src token（坐标）
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx) # 从B token中抽出与src token匹配的dst token(坐标)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]
    
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)
    
    def unmerge( x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, ori_shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out
    
    # def merge_source(
    #     self, x: torch.Tensor, source: torch.Tensor = None
    # ) -> torch.Tensor:

    #     if source is None:
    #         n, t, _ = x.shape
    #         source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)
    #     print('source berfore merge:',source.shape)
    #     source = self.merge(source, mode="amax")
    #     print('source after merge:',source.shape)
    #     return source
    
    # def merge_wavg(
    #    self, x: torch.Tensor, size: torch.Tensor = None
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
        
    #     if size is None:
    #         size = torch.ones_like(x[..., 0, None])

    #     x = self.merge(x * size, mode="sum")
    #     size = self.merge(size, mode="sum")    
    #     x = x / size
        
        # return x, size
    return merge, unmerge
    

def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size


def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source
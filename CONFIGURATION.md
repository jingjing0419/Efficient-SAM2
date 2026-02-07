# Efficient SAM 2 - Configuration Guide

This guide provides detailed documentation for all configuration parameters available in Efficient SAM 2.

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Window Bypass Parameters](#window-bypass-parameters)
3. [Bypass Network Parameters](#bypass-network-parameters)
4. [Memory Management Parameters](#memory-management-parameters)
5. [Inference Parameters](#inference-parameters)
6. [Dataset-Specific Configurations](#dataset-specific-configurations)
7. [Example Commands](#example-commands)

---

## Quick Reference

### Basic Inference Command

```bash
CUDA_VISIBLE_DEVICES=0 python tools/vos_inference_main.py \
    --sam2_model='base+' \
    --dataset='SAV_test' \
    --apply_WB \
    --apply_bypass \
    --output_mask_dir='./outputs/predictions'
```

### Full Optimization Command

```bash
CUDA_VISIBLE_DEVICES=0 python tools/vos_inference_main.py \
    --sam2_model='base+' \
    --dataset='SAV_test' \
    --apply_WB \
    --apply_bypass \
    --prune_memory \
    --topk_mask \
    --set_drop_ratio=0.95 \
    --WB_theta=0.7 \
    --Mem_stride=1 \
    --dilate_mask \
    --output_mask_dir='./outputs/predictions'
```

---

## Window Bypass Parameters

### `apply_WB` (bool, default: False)

Enable the Window Bypass mechanism. This is the primary optimization that allows selective window processing.

```bash
--apply_WB
```

**Effect:**
- Only processes windows containing target objects
- Skips background windows using a bypass network
- Significantly reduces computation time

### `WB_theta` (float, default: 0.6)

Window selection ratio. Determines what fraction of windows to process.

```bash
--WB_theta=0.7    # Process top 70% of windows
--WB_theta=0.5    # Process top 50% of windows
--WB_theta=0.9    # Process top 90% of windows
```

**Trade-off:**
- Higher θ → More accuracy, less speedup
- Lower θ → More speedup, potential accuracy loss

### `WB_all_layer` (bool, default: False)

Apply window bypass to ALL encoder layers instead of just selected layers.

```bash
--WB_all_layer
```

**Recommended for:**
- Maximum speedup
- Large models (SAM 2 Large)

### `win_sel_layer` (list)

Layers used for window selection. Only effective when `WB_all_layer=False`.

**SAM 2 Base+ (default):**
```bash
--win_sel_layer='[6]'
```

**SAM 2 Large (default):**
```bash
--win_sel_layer='[9]'
```

**All-layer mode:**
```bash
--win_sel_layer='[0,3,6]'      # Base+
--win_sel_layer='[0,3,9]'      # Large
```

### `fpn_feat_layer` (list, optional)

Layers used for FPN feature extraction in all-layer mode.

**SAM 2 Base+:**
```bash
--fpn_feat_layer='[1,4]'
```

**SAM 2 Large:**
```bash
--fpn_feat_layer='[1,7]'
```

### `scale_layer` (list, optional)

Layers used for multi-scale feature fusion.

**SAM 2 Base+:**
```bash
--scale_layer='[2,5]'
```

### `disable_WB` (bool, default: False)

Disable window bypass during inference. Useful for comparing baseline vs. optimized performance.

```bash
--disable_WB
```

---

## Bypass Network Parameters

### `apply_bypass` (bool, default: False)

Enable the bypass network adapter. Must be used with `--apply_WB`.

```bash
--apply_bypass
```

### `bypass_type` (str, default: 'bottleneck')

Architecture of the bypass network adapter.

#### Options:

1. **bottleneck** - Balanced performance
```bash
--bypass_type='bottleneck'
```
Structure: Down → ReLU → Dropout → Up (with residual)

2. **attention** - Better accuracy for complex scenes
```bash
--bypass_type='attention'
```
Structure: Multi-head attention adapter

3. **FFN** - Maximum speed
```bash
--bypass_type='FFN'
```
Structure: FFN-based adapter

### `bypass_ckpt` (str)

Path to pretrained bypass network weights.

```bash
--bypass_ckpt='./bypass/ckpt/bypass_bottleneck_base.pth'
--bypass_ckpt='./bypass/ckpt/bypass_attention_large.pth'
```

### `final_global_layer` (int)

Layer index where the bypass network is inserted.

| Model | Layer |
|-------|-------|
| SAM 2 Base+ | 20 |
| SAM 2 Large | 43 |

### `small_bypass` (bool, default: False)

Use smaller bypass network on multiple layers instead of large bypass on final layer.

```bash
--small_bypass
```

---

## Memory Management Parameters

### `Mem_stride` (int, default: 1)

Temporal stride for memory updates. Controls how often memory banks are updated.

```bash
--Mem_stride=1    # Update every frame (baseline)
--Mem_stride=3    # Update every 3rd frame
--Mem_stride=5    # Update every 5th frame
```

**Effect:**
- Higher values reduce memory size
- May impact tracking accuracy for fast-moving objects

### `prune_memory` (bool, default: False)

Enable memory token pruning based on attention scores.

```bash
--prune_memory
```

### `set_drop_ratio` (float, default: 0.95)

Ratio of tokens to keep after pruning.

```bash
--set_drop_ratio=0.95    # Keep 95% of tokens
--set_drop_ratio=0.90    # Keep 90% of tokens
--set_drop_ratio=0.80    # Keep 80% of tokens
```

### Pruning Strategies

Choose ONE of the following:

```bash
# Top-k selection (default, recommended)
--topk_mask

# Random selection
--random_mask

# Uniform sampling
--uniform_mask
```

### `Mem_Frame_Prune` (bool, default: False)

Prune similar memory frames to reduce redundancy.

```bash
--Mem_Frame_Prune
```

### `num_frame_to_prune` (int, default: 2)

Number of frames to prune per group.

```bash
--num_frame_to_prune=2
--num_frame_to_prune=4
```

### `pool_memory` (bool, default: False)

Enable memory pooling for efficient key-value compression.

```bash
--pool_memory
```

### `pooling_ks` (int, default: 2)

Pooling kernel size for memory compression.

```bash
--pooling_ks=2    # 2x2 pooling
--pooling_ks=4    # 4x4 pooling
```

### `Mem_filter` (bool, default: False)

Enable memory filtering based on attention.

```bash
--Mem_filter
```

---

## Inference Parameters

### Model Selection

```bash
# Base+ model (recommended for efficiency)
--sam2_model='base+'

# Large model (higher accuracy)
--sam2_model='large'
```

### Output Configuration

```bash
# Output directory
--output_mask_dir='./outputs/predictions'

# Threshold for mask predictions
--score_thresh=0.0
```

### Mask Options

```bash
# Use per-object PNG files (required for SAV dataset)
--per_obj_png_file

# Use all available masks as input (instead of only first frame)
--use_all_masks

# Apply mask dilation (for better boundary quality)
--dilate_mask

# Dilation kernel size
--dilate_kernel_size=5
```

### Object Tracking Options

```bash
# Track objects appearing later in video (for MOSE, YouTube-VOS)
--track_object_appearing_later_in_video
```

---

## Dataset-Specific Configurations

### SAV (Segment Anything Video)

```bash
python tools/vos_inference_main.py \
    --dataset='SAV_test' \
    --sam2_model='base+' \
    --per_obj_png_file \
    --apply_WB \
    --apply_bypass \
    --prune_memory \
    --topk_mask \
    --set_drop_ratio=0.95 \
    --Mem_stride=1 \
    --output_mask_dir='./outputs/sav_test_pred'
```

### DAVIS 2017

```bash
python tools/vos_inference_main.py \
    --dataset='DAVIS' \
    --sam2_model='base+' \
    --base_video_dir='/path/to/DAVIS/JPEGImages/480p' \
    --input_mask_dir='/path/to/DAVIS/Annotations/480p' \
    --video_list_file='/path/to/DAVIS/ImageSets/2017/val.txt' \
    --apply_WB \
    --apply_bypass \
    --output_mask_dir='./outputs/davis_pred'
```

### MOSE

```bash
python tools/vos_inference_main.py \
    --dataset='MOSE' \
    --sam2_model='base+' \
    --per_obj_png_file \
    --track_object_appearing_later_in_video \
    --apply_WB \
    --apply_bypass \
    --output_mask_dir='./outputs/mose_pred'
```

### MOSE v2

```bash
python tools/vos_inference_main.py \
    --dataset='MOSEv2' \
    --sam2_model='large' \
    --base_video_dir='/path/to/MOSE_v2/valid/JPEGImages/' \
    --input_mask_dir='/path/to/MOSE_v2/valid/Annotations/' \
    --apply_WB \
    --apply_bypass \
    --prune_memory \
    --topk_mask \
    --set_drop_ratio=0.95 \
    --output_mask_dir='./outputs/mose_v2_pred'
```

### SeCVOS

```bash
python tools/vos_inference_main.py \
    --dataset='SeCVOS' \
    --sam2_model='base+' \
    --per_obj_png_file \
    --track_object_appearing_later_in_video \
    --apply_WB \
    --apply_bypass \
    --output_mask_dir='./outputs/secvos_pred'
```

---

## Example Commands

### Baseline (No Optimization)

```bash
python tools/vos_inference_main.py \
    --sam2_model='base+' \
    --dataset='SAV_test' \
    --output_mask_dir='./outputs/baseline'
```

### Window Bypass Only

```bash
python tools/vos_inference_main.py \
    --sam2_model='base+' \
    --dataset='SAV_test' \
    --apply_WB \
    --WB_theta=0.7 \
    --output_mask_dir='./outputs/wb_only'
```

### Window Bypass + Bypass Network

```bash
python tools/vos_inference_main.py \
    --sam2_model='base+' \
    --dataset='SAV_test' \
    --apply_WB \
    --apply_bypass \
    --bypass_type='bottleneck' \
    --bypass_ckpt='./bypass/ckpt/bypass_bottleneck_base.pth' \
    --output_mask_dir='./outputs/wb_bypass'
```

### Full Optimization (Recommended)

```bash
python tools/vos_inference_main.py \
    --sam2_model='base+' \
    --dataset='SAV_test' \
    --apply_WB \
    --apply_bypass \
    --bypass_type='bottleneck' \
    --prune_memory \
    --topk_mask \
    --set_drop_ratio=0.95 \
    --WB_theta=0.7 \
    --Mem_stride=1 \
    --dilate_mask \
    --output_mask_dir='./outputs/full_optimization'
```

### Maximum Speed (All-Layer WB)

```bash
python tools/vos_inference_main.py \
    --sam2_model='large' \
    --dataset='SAV_test' \
    --WB_all_layer \
    --apply_WB \
    --apply_bypass \
    --bypass_type='bottleneck' \
    --selected_layers='[0,1,2,...,43]' \
    --win_sel_layer='[0,3,9]' \
    --fpn_feat_layer='[1,7]' \
    --output_mask_dir='./outputs/all_layer_wb'
```

### Memory Pooling Mode

```bash
python tools/vos_inference_main.py \
    --sam2_model='base+' \
    --dataset='SAV_test' \
    --apply_WB \
    --apply_bypass \
    --pool_memory \
    --pooling_ks=2 \
    --Mem_stride=5 \
    --output_mask_dir='./outputs/memory_pooling'
```

---

## Performance Tips

1. **For Maximum Speed:**
   - Use `--WB_all_layer` with `--WB_theta=0.5`
   - Use `--Mem_stride=5`
   - Use `--set_drop_ratio=0.80`

2. **For Best Accuracy:**
   - Use `--WB_theta=0.9`
   - Use `--Mem_stride=1`
   - Use `--set_drop_ratio=0.99`

3. **Balanced Setting (Recommended):**
   - Use `--WB_theta=0.7`
   - Use `--Mem_stride=1`
   - Use `--set_drop_ratio=0.95`

4. **Memory Constrained:**
   - Enable `--pool_memory`
   - Increase `--Mem_stride`
   - Decrease `--set_drop_ratio`

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce memory usage
--Mem_stride=3
--set_drop_ratio=0.80
--dilate_mask    # Disable if still OOM
```

### Low Accuracy

```bash
# Increase window selection
--WB_theta=0.9

# More frequent memory updates
--Mem_stride=1

# Keep more tokens
--set_drop_ratio=0.99
```

### Slow Inference

```bash
# Increase window selection threshold
--WB_theta=0.5

# Reduce memory updates
--Mem_stride=5

# Prune more tokens
--set_drop_ratio=0.80
```

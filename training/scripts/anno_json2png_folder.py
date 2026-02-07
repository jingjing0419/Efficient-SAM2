import json
import numpy as np
from PIL import Image
import os
import argparse
import glob
from pathlib import Path

def decode_rle_mask(rle_data):
    """
    使用 pycocotools 解码 RLE 掩码数据（COCO 标准格式）
    """
    try:
        # 尝试导入 pycocotools
        try:
            from pycocotools import mask as maskUtils
        except ImportError:
            print("警告: 未安装 pycocotools，尝试安装...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pycocotools'])
            from pycocotools import mask as maskUtils
            print("pycocotools 安装成功")
        
        size = rle_data['size']
        counts = rle_data['counts']
        
        if isinstance(counts, str):
            # 压缩的 RLE 字符串格式
            rle_obj = {
                'counts': counts,
                'size': size
            }
            # 使用官方 pycocotools 解码
            mask = maskUtils.decode(rle_obj)
            return mask
            
        elif isinstance(counts, list):
            # 未压缩的 RLE 计数列表格式
            return decode_rle_counts_fallback(counts, size)
        else:
            raise ValueError(f"不支持的 counts 格式: {type(counts)}")
            
    except Exception as e:
        print(f"    RLE 解码失败: {e}")
        print(f"    使用备用解码方法...")
        return decode_rle_fallback(rle_data)

def decode_rle_counts_fallback(counts, size):
    """
    备用方法：从RLE计数列表创建掩码
    """
    try:
        height, width = size
        total_pixels = height * width
        
        mask_1d = np.zeros(total_pixels, dtype=np.uint8)
        
        pixel_idx = 0
        value = 0  # 从背景开始
        
        for count in counts:
            if pixel_idx + count > total_pixels:
                count = total_pixels - pixel_idx
            
            if value == 1:  # 前景像素
                mask_1d[pixel_idx:pixel_idx + count] = 1
            
            pixel_idx += count
            value = 1 - value  # 交替前景/背景
            
            if pixel_idx >= total_pixels:
                break
        
        mask_2d = mask_1d.reshape((height, width))
        return mask_2d
        
    except Exception as e:
        print(f"    备用RLE计数解码失败: {e}")
        height, width = size
        return np.zeros((height, width), dtype=np.uint8)

def decode_rle_fallback(rle_data):
    """
    最终备用方法：创建空掩码
    """
    try:
        size = rle_data['size']
        height, width = size
        print(f"    创建空掩码 ({height}x{width})")
        return np.zeros((height, width), dtype=np.uint8)
    except:
        return np.zeros((480, 640), dtype=np.uint8)

# def process_sav_video_by_masklet(json_file_path, output_dir, frame_range=None, stride=1):
#     """
#     按 masklet 分组处理 SA-V 视频格式的 JSON 文件
    
#     目录结构:
#     output_dir/
#     └── video_id/
#         ├── 000/  (masklet_0)
#         │   ├── 00000.png  (frame_0)
#         │   ├── 00001.png  (frame_1)
#         │   └── ...
#         ├── 001/  (masklet_1)
#         │   ├── 00000.png
#         │   └── ...
#         └── ...
#     """
#     print(f"开始处理 SA-V 视频文件: {json_file_path}")
    
#     # 读取 JSON 文件
#     try:
#         with open(json_file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#     except Exception as e:
#         print(f"读取 JSON 文件失败: {e}")
#         return
    
#     # 提取视频信息
#     video_id = data.get('video_id', 'unknown')
#     height = int(data.get('video_height', 480))
#     width = int(data.get('video_width', 640))
#     frame_count = int(data.get('video_frame_count', 0))
#     masklet_num = data.get('masklet_num', 0)
    
#     print(f"视频ID: {video_id}")
#     print(f"视频尺寸: {height} x {width}")
#     print(f"总帧数: {frame_count}")
#     print(f"掩码对象数: {masklet_num}")
    
#     # 获取掩码数据
#     masklets = data.get('masklet', [])
#     masklet_ids = data.get('masklet_id', list(range(masklet_num)))
    
#     if not masklets:
#         print("未找到 masklet 数据")
#         return
    
#     print(f"实际可用帧数: {len(masklets)}")
    
#     # 确定处理范围
#     if frame_range:
#         start_frame, end_frame = frame_range
#         start_frame = max(0, start_frame)
#         end_frame = min(len(masklets), end_frame)
#     else:
#         start_frame, end_frame = 0, len(masklets)
    
#     print(f"处理帧范围: {start_frame} - {end_frame}")
    
#     # 创建主输出目录
#     video_output_dir = os.path.join(output_dir, video_id)
#     os.makedirs(video_output_dir, exist_ok=True)
    
#     # 为每个 masklet 创建子目录
#     masklet_dirs = {}
#     for i, masklet_id in enumerate(masklet_ids):
#         masklet_dir_name = f"{i:03d}"  # 000, 001, 002, ...
#         masklet_dir_path = os.path.join(video_output_dir, masklet_dir_name)
#         os.makedirs(masklet_dir_path, exist_ok=True)
#         masklet_dirs[i] = masklet_dir_path
#         print(f"创建目录: {masklet_dir_name} (masklet_id: {masklet_id})")
    
#     # 统计信息
#     total_success = 0
#     total_masks = 0
#     masklet_stats = {i: {'success': 0, 'total': 0, 'pixels': 0} for i in range(len(masklet_ids))}
    
#     # 处理每一帧
#     for frame_idx in range(start_frame, end_frame):
#         if frame_idx % stride != 0:
#             continue
#         frame_masks = masklets[frame_idx]
#         print(f"\n处理帧 {frame_idx}: {len(frame_masks)} 个掩码")
        
#         for mask_idx, mask_data in enumerate(frame_masks):
#             if mask_idx >= len(masklet_ids):
#                 print(f"  警告: 掩码索引 {mask_idx} 超出 masklet_ids 范围")
#                 continue
                
#             masklet_id = masklet_ids[mask_idx]
#             masklet_dir = masklet_dirs[mask_idx]
            
#             try:
#                 print(f"  处理 masklet_{mask_idx:03d} (ID:{masklet_id})...", end=' ')
                
#                 # 使用正确的 COCO RLE 解码
#                 binary_mask = decode_rle_mask(mask_data)
                
#                 # 统计前景像素
#                 foreground_pixels = np.sum(binary_mask)
#                 masklet_stats[mask_idx]['pixels'] += foreground_pixels
                
#                 # 转换为图像 (0=黑色背景, 255=白色前景)
#                 mask_image = (binary_mask * 255).astype(np.uint8)
                
#                 # 保存 PNG - 使用5位数字格式的帧号
#                 output_filename = f"{frame_idx:05d}.png"
#                 output_path = os.path.join(masklet_dir, output_filename)
                
#                 # 保存为灰度图像
#                 Image.fromarray(mask_image, mode='L').save(output_path)
                
#                 print(f"✓ (前景像素: {foreground_pixels})")
#                 total_success += 1
#                 masklet_stats[mask_idx]['success'] += 1
                
#             except Exception as e:
#                 print(f"✗ 失败: {e}")
            
#             total_masks += 1
#             masklet_stats[mask_idx]['total'] += 1
    
#     # 输出统计信息
#     print(f"\n=== 处理完成 ===")
#     print(f"总计: {total_success}/{total_masks} 个掩码成功转换")
#     print(f"输出目录: {video_output_dir}")
    
#     print(f"\n=== 各 Masklet 统计 ===")
#     for mask_idx, stats in masklet_stats.items():
#         masklet_id = masklet_ids[mask_idx] if mask_idx < len(masklet_ids) else mask_idx
#         success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
#         avg_pixels = stats['pixels'] / stats['success'] if stats['success'] > 0 else 0
#         print(f"Masklet {mask_idx:03d} (ID:{masklet_id}): {stats['success']}/{stats['total']} ({success_rate:.1f}%) - 平均前景像素: {avg_pixels:.0f}")
    
#     return video_output_dir

# def process_sav_video_by_masklet(json_file_path, output_dir, frame_range=None, stride=1, max_masklet_num=1):
#     """
#     按 masklet 分组处理 SA-V 视频格式的 JSON 文件
#     添加首帧过滤功能：跳过首帧前景像素为0或大于总像素30%的masklet
    
#     目录结构:
#     output_dir/
#     └── video_id/
#         ├── 000/  (masklet_0)
#         │   ├── 00000.png  (frame_0)
#         │   ├── 00001.png  (frame_1)
#         │   └── ...
#         ├── 001/  (masklet_1)
#         │   ├── 00000.png
#         │   └── ...
#         └── ...
#     """
#     print(f"开始处理 SA-V 视频文件: {json_file_path}")
    
#     # 读取 JSON 文件
#     try:
#         with open(json_file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#     except Exception as e:
#         print(f"读取 JSON 文件失败: {e}")
#         return
    
#     # 提取视频信息
#     video_id = data.get('video_id', 'unknown')
#     height = int(data.get('video_height', 480))
#     width = int(data.get('video_width', 640))
#     frame_count = int(data.get('video_frame_count', 0))
#     masklet_num = data.get('masklet_num', 0)
    
#     print(f"视频ID: {video_id}")
#     print(f"视频尺寸: {height} x {width}")
#     print(f"总帧数: {frame_count}")
#     print(f"掩码对象数: {masklet_num}")
    
#     # 获取掩码数据
#     masklets = data.get('masklet', [])
#     masklet_ids = data.get('masklet_id', list(range(masklet_num)))
    
#     if not masklets:
#         print("未找到 masklet 数据")
#         return
    
#     print(f"实际可用帧数: {len(masklets)}")
    
#     # 确定处理范围
#     if frame_range:
#         start_frame, end_frame = frame_range
#         start_frame = max(0, start_frame)
#         end_frame = min(len(masklets), end_frame)
#     else:
#         start_frame, end_frame = 0, len(masklets)
    
#     print(f"处理帧范围: {start_frame} - {end_frame}")
    
#     # ===== 新增：首帧过滤逻辑 =====
#     print(f"\n=== 首帧质量检查 ===")
#     total_pixels = height * width
#     pixel_threshold = total_pixels * 0.2  # 30%阈值
#     valid_masklets = set()  # 存储通过过滤的masklet索引
    
#     if len(masklets) > start_frame:
#         first_frame_masks = masklets[start_frame]
#         print(f"首帧包含 {len(first_frame_masks)} 个掩码")
        
#         for mask_idx, mask_data in enumerate(first_frame_masks):
#             if mask_idx >= len(masklet_ids):
#                 continue
                
#             masklet_id = masklet_ids[mask_idx]
            
#             try:
#                 # 解码首帧掩码
#                 binary_mask = decode_rle_mask(mask_data)
#                 foreground_pixels = np.sum(binary_mask)
                
#                 # 检查过滤条件
#                 if foreground_pixels == 0:
#                     print(f"  Masklet {mask_idx:03d} (ID:{masklet_id}): ✗ 跳过 (首帧无前景像素)")
#                 elif foreground_pixels > pixel_threshold:
#                     percentage = (foreground_pixels / total_pixels) * 100
#                     print(f"  Masklet {mask_idx:03d} (ID:{masklet_id}): ✗ 跳过 (首帧前景像素过多: {percentage:.1f}%)")
#                 else:
#                     percentage = (foreground_pixels / total_pixels) * 100
#                     print(f"  Masklet {mask_idx:03d} (ID:{masklet_id}): ✓ 通过 (首帧前景像素: {foreground_pixels}, {percentage:.1f}%)")
#                     valid_masklets.add(mask_idx)
                    
#             except Exception as e:
#                 print(f"  Masklet {mask_idx:03d} (ID:{masklet_id}): ✗ 跳过 (首帧解码失败: {e})")
    
#     print(f"\n通过首帧过滤的 masklet 数量: {len(valid_masklets)}/{len(masklet_ids)}")
    
#     if not valid_masklets:
#         print("警告: 没有 masklet 通过首帧过滤，退出处理")
#         return video_output_dir
    
#     # 创建主输出目录
#     video_output_dir = os.path.join(output_dir, video_id)
#     os.makedirs(video_output_dir, exist_ok=True)
    
#     # 只为通过过滤的 masklet 创建子目录
#     masklet_dirs = {}
#     for i, masklet_id in enumerate(masklet_ids):
#         if len(masklet_dirs) >= max_masklet_num:
#             break
#         if i in valid_masklets:  # 只处理通过过滤的masklet
#             masklet_dir_name = f"{i:03d}"  # 000, 001, 002, ...
#             masklet_dir_path = os.path.join(video_output_dir, masklet_dir_name)
#             os.makedirs(masklet_dir_path, exist_ok=True)
#             masklet_dirs[i] = masklet_dir_path
#             print(f"创建目录: {masklet_dir_name} (masklet_id: {masklet_id}) ✓")
            
    
#     # 统计信息
#     total_success = 0
#     total_masks = 0
#     total_skipped = 0
#     masklet_stats = {i: {'success': 0, 'total': 0, 'pixels': 0} for i in valid_masklets}
    
#     # 处理每一帧
#     for frame_idx in range(start_frame, end_frame):
#         if frame_idx % stride != 0:
#             continue
#         frame_masks = masklets[frame_idx]
#         valid_masks_in_frame = sum(1 for i in range(len(frame_masks)) if i in valid_masklets)
#         print(f"\n处理帧 {frame_idx}: {len(frame_masks)} 个掩码 (有效: {valid_masks_in_frame})")
        
#         for mask_idx, mask_data in enumerate(frame_masks):
#             if mask_idx >= len(masklet_ids):
#                 print(f"  警告: 掩码索引 {mask_idx} 超出 masklet_ids 范围")
#                 continue
            
#             # 检查是否为有效的masklet
#             if mask_idx not in valid_masklets:
#                 total_skipped += 1
#                 continue
                
#             masklet_id = masklet_ids[mask_idx]
#             masklet_dir = masklet_dirs[mask_idx]
            
#             try:
#                 print(f"  处理 masklet_{mask_idx:03d} (ID:{masklet_id})...", end=' ')
                
#                 # 使用正确的 COCO RLE 解码
#                 binary_mask = decode_rle_mask(mask_data)
                
#                 # 统计前景像素
#                 foreground_pixels = np.sum(binary_mask)
#                 masklet_stats[mask_idx]['pixels'] += foreground_pixels
                
#                 # 转换为图像 (0=黑色背景, 255=白色前景)
#                 mask_image = (binary_mask * 255).astype(np.uint8)
                
#                 # 保存 PNG - 使用5位数字格式的帧号
#                 output_filename = f"{frame_idx:05d}.png"
#                 output_path = os.path.join(masklet_dir, output_filename)
                
#                 # 保存为灰度图像
#                 Image.fromarray(mask_image, mode='L').save(output_path)
                
#                 print(f"✓ (前景像素: {foreground_pixels})")
#                 total_success += 1
#                 masklet_stats[mask_idx]['success'] += 1
                
#             except Exception as e:
#                 print(f"✗ 失败: {e}")
            
#             total_masks += 1
#             masklet_stats[mask_idx]['total'] += 1
    
#     # 输出统计信息
#     print(f"\n=== 处理完成 ===")
#     print(f"首帧过滤: 跳过了 {len(masklet_ids) - len(valid_masklets)} 个质量不佳的 masklet")
#     print(f"处理过程: 跳过了 {total_skipped} 个被过滤的掩码")
#     print(f"成功转换: {total_success}/{total_masks} 个有效掩码")
#     print(f"输出目录: {video_output_dir}")
    
#     print(f"\n=== 有效 Masklet 统计 ===")
#     for mask_idx, stats in masklet_stats.items():
#         masklet_id = masklet_ids[mask_idx] if mask_idx < len(masklet_ids) else mask_idx
#         success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
#         avg_pixels = stats['pixels'] / stats['success'] if stats['success'] > 0 else 0
#         print(f"Masklet {mask_idx:03d} (ID:{masklet_id}): {stats['success']}/{stats['total']} ({success_rate:.1f}%) - 平均前景像素: {avg_pixels:.0f}")
    
#     with open("train.txt", "a", encoding="utf-8") as f:
#         f.write(f"{video_id}\n")
    
#     return video_output_dir

# def process_sav_video_by_masklet(json_file_path, output_dir, frame_range=None, stride=4, max_masklet_num=1):
#     """
#     按 masklet 分组处理 SA-V 视频格式的 JSON 文件
#     添加首帧过滤功能：跳过首帧前景像素为0或大于总像素30%的masklet
#     限制最多保留 max_masklet_num 个满足条件的masklet
    
#     目录结构:
#     output_dir/
#     └── video_id/
#         ├── 000/  (masklet_0)
#         │   ├── 00000.png  (frame_0)
#         │   ├── 00001.png  (frame_1)
#         │   └── ...
#         ├── 001/  (masklet_1)
#         │   ├── 00000.png
#         │   └── ...
#         └── ...
#     """
#     print(f"开始处理 SA-V 视频文件: {json_file_path}")
#     print(f"最大保留 masklet 数量: {max_masklet_num}")
    
#     # 读取 JSON 文件
#     try:
#         with open(json_file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#     except Exception as e:
#         print(f"读取 JSON 文件失败: {e}")
#         return None
    
#     # 提取视频信息
#     video_id = data.get('video_id', 'unknown')
#     height = int(data.get('video_height', 480))
#     width = int(data.get('video_width', 640))
#     frame_count = int(data.get('video_frame_count', 0))
#     masklet_num = data.get('masklet_num', 0)
    
#     print(f"视频ID: {video_id}")
#     print(f"视频尺寸: {height} x {width}")
#     print(f"总帧数: {frame_count}")
#     print(f"掩码对象数: {masklet_num}")
    
#     # 获取掩码数据
#     masklets = data.get('masklet', [])
#     masklet_ids = data.get('masklet_id', list(range(masklet_num)))
    
#     if not masklets:
#         print("未找到 masklet 数据")
#         return None
    
#     print(f"实际可用帧数: {len(masklets)}")
    
#     # 确定处理范围
#     if frame_range:
#         start_frame, end_frame = frame_range
#         start_frame = max(0, start_frame)
#         end_frame = min(len(masklets), end_frame)
#     else:
#         start_frame, end_frame = 0, len(masklets)
    
#     print(f"处理帧范围: {start_frame} - {end_frame}")
    
#     # ===== 新增：首帧过滤逻辑 =====
#     print(f"\n=== 首帧质量检查 ===")
#     total_pixels = height * width
#     pixel_threshold = total_pixels * 0.2  # 20%阈值
#     valid_masklets = []  # 改为列表，保持顺序
    
#     if len(masklets) > start_frame:
#         first_frame_masks = masklets[start_frame]
#         print(f"首帧包含 {len(first_frame_masks)} 个掩码")
        
#         # 用于存储候选masklet及其质量评分
#         masklet_candidates = []
        
#         for mask_idx, mask_data in enumerate(first_frame_masks):
#             if mask_idx >= len(masklet_ids):
#                 continue
                
#             masklet_id = masklet_ids[mask_idx]
            
#             try:
#                 # 解码首帧掩码
#                 binary_mask = decode_rle_mask(mask_data)
#                 foreground_pixels = np.sum(binary_mask)
                
#                 # 检查过滤条件
#                 if foreground_pixels == 0:
#                     print(f"  Masklet {mask_idx:03d} (ID:{masklet_id}): ✗ 跳过 (首帧无前景像素)")
#                 elif foreground_pixels > pixel_threshold:
#                     percentage = (foreground_pixels / total_pixels) * 100
#                     print(f"  Masklet {mask_idx:03d} (ID:{masklet_id}): ✗ 跳过 (首帧前景像素过多: {percentage:.1f}%)")
#                 else:
#                     percentage = (foreground_pixels / total_pixels) * 100
#                     print(f"  Masklet {mask_idx:03d} (ID:{masklet_id}): ✓ 候选 (首帧前景像素: {foreground_pixels}, {percentage:.1f}%)")
                    
#                     # 将候选masklet添加到列表中，包含质量评分
#                     # 这里使用前景像素数作为质量评分，可以根据需要调整评分标准
#                     masklet_candidates.append({
#                         'index': mask_idx,
#                         'id': masklet_id,
#                         'pixels': foreground_pixels,
#                         'percentage': percentage
#                     })
                    
#             except Exception as e:
#                 print(f"  Masklet {mask_idx:03d} (ID:{masklet_id}): ✗ 跳过 (首帧解码失败: {e})")
        
#         # 根据质量评分排序并选择前N个
#         if masklet_candidates:
#             # 按前景像素数排序（可以根据需要调整排序标准）
#             # 这里选择前景像素数适中的masklet（不要太少也不要太多）
#             masklet_candidates.sort(key=lambda x: x['pixels'], reverse=True)
            
#             # 选择前max_masklet_num个
#             selected_candidates = masklet_candidates[:max_masklet_num]
            
#             print(f"\n=== 选择结果 ===")
#             for i, candidate in enumerate(selected_candidates):
#                 valid_masklets.append(candidate['index'])
#                 print(f"  选择 Masklet {candidate['index']:03d} (ID:{candidate['id']}) - "
#                       f"前景像素: {candidate['pixels']}, 占比: {candidate['percentage']:.1f}%")
    
#     print(f"\n通过首帧过滤的 masklet 数量: {len(valid_masklets)}/{len(masklet_ids)}")
#     print(f"实际选择的 masklet 数量: {min(len(valid_masklets), max_masklet_num)}")
    
#     if not valid_masklets:
#         print("警告: 没有 masklet 通过首帧过滤，退出处理")
#         return None
    
#     # 创建主输出目录
#     video_output_dir = os.path.join(output_dir, video_id)
#     os.makedirs(video_output_dir, exist_ok=True)
    
#     # 只为选择的 masklet 创建子目录
#     masklet_dirs = {}
#     for i, masklet_id in enumerate(masklet_ids):
#         if i in valid_masklets:  # 只处理选择的masklet
#             masklet_dir_name = f"{i:03d}"  # 000, 001, 002, ...
#             masklet_dir_path = os.path.join(video_output_dir, masklet_dir_name)
#             os.makedirs(masklet_dir_path, exist_ok=True)
#             masklet_dirs[i] = masklet_dir_path
#             print(f"创建目录: {masklet_dir_name} (masklet_id: {masklet_id}) ✓")
    
#     # 统计信息
#     total_success = 0
#     total_masks = 0
#     total_skipped = 0
#     masklet_stats = {i: {'success': 0, 'total': 0, 'pixels': 0} for i in valid_masklets}
    
#     # 处理每一帧
#     for frame_idx in range(start_frame, end_frame):
#         # if frame_idx % stride != 0:
#         #     continue
#         frame_masks = masklets[frame_idx]
#         valid_masks_in_frame = sum(1 for i in range(len(frame_masks)) if i in valid_masklets)
#         print(f"\n处理帧 {frame_idx}: {len(frame_masks)} 个掩码 (有效: {valid_masks_in_frame})")
        
#         for mask_idx, mask_data in enumerate(frame_masks):
#             if mask_idx >= len(masklet_ids):
#                 print(f"  警告: 掩码索引 {mask_idx} 超出 masklet_ids 范围")
#                 continue
            
#             # 检查是否为选择的masklet
#             if mask_idx not in valid_masklets:
#                 total_skipped += 1
#                 continue
                
#             masklet_id = masklet_ids[mask_idx]
#             masklet_dir = masklet_dirs[mask_idx]
            
#             try:
#                 print(f"  处理 masklet_{mask_idx:03d} (ID:{masklet_id})...", end=' ')
                
#                 # 使用正确的 COCO RLE 解码
#                 binary_mask = decode_rle_mask(mask_data)
                
#                 # 统计前景像素
#                 foreground_pixels = np.sum(binary_mask)
#                 masklet_stats[mask_idx]['pixels'] += foreground_pixels
                
#                 # 转换为图像 (0=黑色背景, 255=白色前景)
#                 mask_image = (binary_mask * 255).astype(np.uint8)
                
#                 # 保存 PNG - 使用5位数字格式的帧号
#                 output_filename = f"{frame_idx:05d}.png"
#                 output_path = os.path.join(masklet_dir, output_filename)
                
#                 # 保存为灰度图像
#                 Image.fromarray(mask_image, mode='L').save(output_path)
                
#                 print(f"✓ (前景像素: {foreground_pixels})")
#                 total_success += 1
#                 masklet_stats[mask_idx]['success'] += 1
                
#             except Exception as e:
#                 print(f"✗ 失败: {e}")
            
#             total_masks += 1
#             masklet_stats[mask_idx]['total'] += 1
    
#     # 输出统计信息
#     print(f"\n=== 处理完成 ===")
#     print(f"首帧过滤: 跳过了 {len(masklet_ids) - len(valid_masklets)} 个质量不佳的 masklet")
#     print(f"数量限制: 最终选择了 {len(valid_masklets)} 个 masklet (限制: {max_masklet_num})")
#     print(f"处理过程: 跳过了 {total_skipped} 个被过滤的掩码")
#     print(f"成功转换: {total_success}/{total_masks} 个有效掩码")
#     print(f"输出目录: {video_output_dir}")
    
#     print(f"\n=== 选择的 Masklet 统计 ===")
#     for mask_idx, stats in masklet_stats.items():
#         masklet_id = masklet_ids[mask_idx] if mask_idx < len(masklet_ids) else mask_idx
#         success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
#         avg_pixels = stats['pixels'] / stats['success'] if stats['success'] > 0 else 0
#         print(f"Masklet {mask_idx:03d} (ID:{masklet_id}): {stats['success']}/{stats['total']} ({success_rate:.1f}%) - 平均前景像素: {avg_pixels:.0f}")
    
#     # 只有成功处理了至少一个masklet才记录到train.txt
#     if total_success > 0:
#         with open("train.txt", "a", encoding="utf-8") as f:
#             f.write(f"{video_id}\n")
#         print(f"✓ 视频 {video_id} 已记录到 train.txt")
#         return video_output_dir
#     else:
#         print(f"✗ 视频 {video_id} 处理失败，未记录到 train.txt")
#         return None

def process_sav_video_by_masklet(json_file_path, output_dir, frame_range=None, stride=4, max_masklet_num=1):
    """
    按 masklet 分组处理 SA-V 视频格式的 JSON 文件
    添加首帧过滤功能：跳过首帧前景像素为0或大于总像素30%的masklet
    限制最多保留 max_masklet_num 个满足条件的masklet
    
    参数:
    - stride: 标注的间隔帧数，即每隔多少帧进行一次标注（默认4帧）
    
    目录结构:
    output_dir/
    └── video_id/
        ├── 000/  (masklet_0)
        │   ├── 00000.png  (frame_0)
        │   ├── 00004.png  (frame_4)
        │   ├── 00008.png  (frame_8)
        │   └── ...
        ├── 001/  (masklet_1)
        │   ├── 00000.png
        │   ├── 00004.png
        │   └── ...
        └── ...
    """
    print(f"开始处理 SA-V 视频文件: {json_file_path}")
    print(f"标注间隔: 每 {stride} 帧标注一次")
    print(f"最大保留 masklet 数量: {max_masklet_num}")
    
    # 读取 JSON 文件
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取 JSON 文件失败: {e}")
        return None
    
    # 提取视频信息
    video_id = data.get('video_id', 'unknown')
    height = int(data.get('video_height', 480))
    width = int(data.get('video_width', 640))
    frame_count = int(data.get('video_frame_count', 0))
    masklet_num = data.get('masklet_num', 0)
    
    print(f"视频ID: {video_id}")
    print(f"视频尺寸: {height} x {width}")
    print(f"总帧数: {frame_count}")
    print(f"掩码对象数: {masklet_num}")
    
    # 获取掩码数据
    masklets = data.get('masklet', [])
    masklet_ids = data.get('masklet_id', list(range(masklet_num)))
    
    if not masklets:
        print("未找到 masklet 数据")
        return None
    
    print(f"实际标注帧数: {len(masklets)}")
    print(f"预期标注帧数: {(frame_count + stride - 1) // stride}")
    
    # 确定处理范围（基于masklet数组索引）
    if frame_range:
        start_frame, end_frame = frame_range
        start_frame = max(0, start_frame)
        end_frame = min(len(masklets), end_frame)
    else:
        start_frame, end_frame = 0, len(masklets)
    
    print(f"处理masklet索引范围: {start_frame} - {end_frame}")
    print(f"对应实际帧号范围: {start_frame * stride} - {(end_frame - 1) * stride}")
    
    # ===== 首帧过滤逻辑 =====
    print(f"\n=== 首帧质量检查 ===")
    total_pixels = height * width
    pixel_threshold = total_pixels * 0.2  # 20%阈值
    valid_masklets = []  # 改为列表，保持顺序
    
    if len(masklets) > start_frame:
        first_frame_masks = masklets[start_frame]
        actual_first_frame_num = start_frame * stride
        print(f"首帧 (实际帧号: {actual_first_frame_num}) 包含 {len(first_frame_masks)} 个掩码")
        
        # 用于存储候选masklet及其质量评分
        masklet_candidates = []
        
        for mask_idx, mask_data in enumerate(first_frame_masks):
            if mask_idx >= len(masklet_ids):
                continue
                
            masklet_id = masklet_ids[mask_idx]
            
            try:
                # 解码首帧掩码
                binary_mask = decode_rle_mask(mask_data)
                foreground_pixels = np.sum(binary_mask)
                
                # 检查过滤条件
                if foreground_pixels == 0:
                    print(f"  Masklet {mask_idx:03d} (ID:{masklet_id}): ✗ 跳过 (首帧无前景像素)")
                elif foreground_pixels > pixel_threshold:
                    percentage = (foreground_pixels / total_pixels) * 100
                    print(f"  Masklet {mask_idx:03d} (ID:{masklet_id}): ✗ 跳过 (首帧前景像素过多: {percentage:.1f}%)")
                else:
                    percentage = (foreground_pixels / total_pixels) * 100
                    print(f"  Masklet {mask_idx:03d} (ID:{masklet_id}): ✓ 候选 (首帧前景像素: {foreground_pixels}, {percentage:.1f}%)")
                    
                    # 将候选masklet添加到列表中，包含质量评分
                    masklet_candidates.append({
                        'index': mask_idx,
                        'id': masklet_id,
                        'pixels': foreground_pixels,
                        'percentage': percentage
                    })
                    
            except Exception as e:
                print(f"  Masklet {mask_idx:03d} (ID:{masklet_id}): ✗ 跳过 (首帧解码失败: {e})")
        
        # 根据质量评分排序并选择前N个
        if masklet_candidates:
            # 按前景像素数排序（选择前景像素数适中的masklet）
            masklet_candidates.sort(key=lambda x: x['pixels'], reverse=True)
            
            # 选择前max_masklet_num个
            selected_candidates = masklet_candidates[:max_masklet_num]
            
            print(f"\n=== 选择结果 ===")
            for i, candidate in enumerate(selected_candidates):
                valid_masklets.append(candidate['index'])
                print(f"  选择 Masklet {candidate['index']:03d} (ID:{candidate['id']}) - "
                      f"前景像素: {candidate['pixels']}, 占比: {candidate['percentage']:.1f}%")
    
    print(f"\n通过首帧过滤的 masklet 数量: {len(valid_masklets)}/{len(masklet_ids)}")
    print(f"实际选择的 masklet 数量: {min(len(valid_masklets), max_masklet_num)}")
    
    if not valid_masklets:
        print("警告: 没有 masklet 通过首帧过滤，退出处理")
        return None
    
    # 创建主输出目录
    video_output_dir = os.path.join(output_dir, video_id)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # 只为选择的 masklet 创建子目录
    masklet_dirs = {}
    for i, masklet_id in enumerate(masklet_ids):
        if i in valid_masklets:  # 只处理选择的masklet
            masklet_dir_name = f"{i:03d}"  # 000, 001, 002, ...
            masklet_dir_path = os.path.join(video_output_dir, masklet_dir_name)
            os.makedirs(masklet_dir_path, exist_ok=True)
            masklet_dirs[i] = masklet_dir_path
            print(f"创建目录: {masklet_dir_name} (masklet_id: {masklet_id}) ✓")
    
    # 统计信息
    total_success = 0
    total_masks = 0
    total_skipped = 0
    masklet_stats = {i: {'success': 0, 'total': 0, 'pixels': 0} for i in valid_masklets}
    
    # 处理每一帧
    for masklet_frame_idx in range(start_frame, end_frame):
        # 计算实际的视频帧号
        actual_frame_num = masklet_frame_idx * stride
        
        frame_masks = masklets[masklet_frame_idx]
        valid_masks_in_frame = sum(1 for i in range(len(frame_masks)) if i in valid_masklets)
        print(f"\n处理 masklet[{masklet_frame_idx}] -> 实际帧号 {actual_frame_num}: {len(frame_masks)} 个掩码 (有效: {valid_masks_in_frame})")
        
        for mask_idx, mask_data in enumerate(frame_masks):
            if mask_idx >= len(masklet_ids):
                print(f"  警告: 掩码索引 {mask_idx} 超出 masklet_ids 范围")
                continue
            
            # 检查是否为选择的masklet
            if mask_idx not in valid_masklets:
                total_skipped += 1
                continue
                
            masklet_id = masklet_ids[mask_idx]
            masklet_dir = masklet_dirs[mask_idx]
            
            try:
                print(f"  处理 masklet_{mask_idx:03d} (ID:{masklet_id})...", end=' ')
                
                # 使用正确的 COCO RLE 解码
                binary_mask = decode_rle_mask(mask_data)
                
                # 统计前景像素
                foreground_pixels = np.sum(binary_mask)
                masklet_stats[mask_idx]['pixels'] += foreground_pixels
                
                # 转换为图像 (0=黑色背景, 255=白色前景)
                mask_image = (binary_mask * 255).astype(np.uint8)
                
                # ===== 关键修改：使用实际的视频帧号作为文件名 =====
                output_filename = f"{actual_frame_num:05d}.png"
                output_path = os.path.join(masklet_dir, output_filename)
                
                # 保存为灰度图像
                Image.fromarray(mask_image, mode='L').save(output_path)
                
                print(f"✓ -> {output_filename} (前景像素: {foreground_pixels})")
                total_success += 1
                masklet_stats[mask_idx]['success'] += 1
                
            except Exception as e:
                print(f"✗ 失败: {e}")
            
            total_masks += 1
            masklet_stats[mask_idx]['total'] += 1
    
    # 输出统计信息
    print(f"\n=== 处理完成 ===")
    print(f"首帧过滤: 跳过了 {len(masklet_ids) - len(valid_masklets)} 个质量不佳的 masklet")
    print(f"数量限制: 最终选择了 {len(valid_masklets)} 个 masklet (限制: {max_masklet_num})")
    print(f"处理过程: 跳过了 {total_skipped} 个被过滤的掩码")
    print(f"成功转换: {total_success}/{total_masks} 个有效掩码")
    print(f"输出目录: {video_output_dir}")
    
    print(f"\n=== 选择的 Masklet 统计 ===")
    for mask_idx, stats in masklet_stats.items():
        masklet_id = masklet_ids[mask_idx] if mask_idx < len(masklet_ids) else mask_idx
        success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
        avg_pixels = stats['pixels'] / stats['success'] if stats['success'] > 0 else 0
        print(f"Masklet {mask_idx:03d} (ID:{masklet_id}): {stats['success']}/{stats['total']} ({success_rate:.1f}%) - 平均前景像素: {avg_pixels:.0f}")
    
    # 只有成功处理了至少一个masklet才记录到train.txt
    if total_success > 0:
        with open("train.txt", "a", encoding="utf-8") as f:
            f.write(f"{video_id}\n")
        print(f"✓ 视频 {video_id} 已记录到 train.txt")
        return video_output_dir
    else:
        print(f"✗ 视频 {video_id} 处理失败，未记录到 train.txt")
        return None



def create_masklet_overview(json_file_path, output_dir):
    """
    创建 masklet 概览，显示每个 masklet 的基本信息
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    video_id = data.get('video_id', 'unknown')
    masklets = data.get('masklet', [])
    masklet_ids = data.get('masklet_id', [])
    masklet_types = data.get('masklet_type', [])
    masklet_size_rel = data.get('masklet_size_rel', [])
    masklet_frame_count = data.get('masklet_frame_count', [])
    
    print(f"\n=== {video_id} Masklet 概览 ===")
    print(f"总帧数: {len(masklets)}")
    print(f"Masklet 数量: {len(masklet_ids)}")
    
    print(f"\n{'序号':<4} {'ID':<6} {'类型':<12} {'相对大小':<10} {'帧数':<6} {'目录名'}")
    print("-" * 60)
    
    for i, masklet_id in enumerate(masklet_ids):
        masklet_type = masklet_types[i] if i < len(masklet_types) else 'unknown'
        size_rel = masklet_size_rel[i] if i < len(masklet_size_rel) else 0.0
        frame_cnt = masklet_frame_count[i] if i < len(masklet_frame_count) else 0
        dir_name = f"{i:03d}"
        
        print(f"{i:<4} {masklet_id:<6} {masklet_type:<12} {size_rel:<10.4f} {frame_cnt:<6} {dir_name}")
    
    # 显示每个 masklet 在各帧中的存在情况
    print(f"\n=== Masklet 在各帧中的分布 ===")
    print("(显示前10帧的情况)")
    
    for frame_idx in range(min(10, len(masklets))):
        frame_masks = masklets[frame_idx]
        mask_info = []
        for mask_idx, mask_data in enumerate(frame_masks):
            if mask_idx < len(masklet_ids):
                masklet_id = masklet_ids[mask_idx]
                rle_len = len(mask_data.get('counts', '')) if isinstance(mask_data.get('counts'), str) else len(mask_data.get('counts', []))
                mask_info.append(f"{mask_idx:03d}({masklet_id}):{rle_len}")
        
        print(f"帧 {frame_idx:3d}: {' | '.join(mask_info)}")

def validate_conversion(json_file_path, output_dir, sample_frames=3):
    """
    验证转换结果，显示一些样本帧的统计信息
    """
    print(f"\n=== 验证转换结果 ===")
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    video_id = data.get('video_id', 'unknown')
    video_output_dir = os.path.join(output_dir, video_id)
    
    if not os.path.exists(video_output_dir):
        print(f"输出目录不存在: {video_output_dir}")
        return
    
    # 检查生成的文件
    masklet_dirs = [d for d in os.listdir(video_output_dir) if os.path.isdir(os.path.join(video_output_dir, d))]
    masklet_dirs.sort()
    
    print(f"找到 {len(masklet_dirs)} 个 masklet 目录")
    
    for masklet_dir in masklet_dirs[:3]:  # 只检查前3个
        masklet_path = os.path.join(video_output_dir, masklet_dir)
        png_files = [f for f in os.listdir(masklet_path) if f.endswith('.png')]
        png_files.sort()
        
        print(f"\nMasklet {masklet_dir}: {len(png_files)} 个PNG文件")
        
        # 检查前几个文件的内容
        for png_file in png_files[:sample_frames]:
            png_path = os.path.join(masklet_path, png_file)
            try:
                img = Image.open(png_path)
                img_array = np.array(img)
                
                unique_values = np.unique(img_array)
                foreground_pixels = np.sum(img_array == 255)
                total_pixels = img_array.size
                foreground_ratio = foreground_pixels / total_pixels * 100
                
                print(f"  {png_file}: {img_array.shape}, 值范围: {unique_values}, 前景比例: {foreground_ratio:.2f}%")
                
            except Exception as e:
                print(f"  {png_file}: 读取失败 - {e}")
                


def main():
    parser = argparse.ArgumentParser(description='将 SA-V 视频 JSON 文件按 masklet 分组转换为 PNG (使用正确的COCO RLE解码)')
    parser.add_argument('input_dir', help='输入 JSON 文件路径')
    parser.add_argument('output_dir', help='输出目录路径')
    parser.add_argument('--frames', type=str, help='帧范围，格式: start,end (如: 0,10)')
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--file_pattern', type=str, default='*_manual.json')
    parser.add_argument('--overview', action='store_true', help='显示 masklet 概览')
    parser.add_argument('--validate', action='store_true', help='验证转换结果')
    
    args = parser.parse_args()
    
    # # 检查输入文件
    # if not os.path.exists(args.input):
    #     print(f"输入文件不存在: {args.input}")
    #     return
    
    # # 创建输出目录
    # os.makedirs(args.output, exist_ok=True)
    
    # # 显示概览
    # if args.overview:
    #     create_masklet_overview(args.input, args.output)
    #     return
    
    # # 验证结果
    # if args.validate:
    #     validate_conversion(args.input, args.output)
    #     return
    
    # 解析帧范围
    frame_range = None
    if args.frames:
        try:
            start, end = map(int, args.frames.split(','))
            frame_range = (start, end)
            print(f"设置帧范围: {frame_range}")
        except:
            print("帧范围格式错误，使用默认（处理所有帧）")
    
    # # 处理文件
    # result_dir = process_sav_video_by_masklet(args.input, args.output, frame_range, stride=4)
    
    # if result_dir:
    #     print(f"\n转换完成！可以使用以下命令验证结果:")
    #     print(f"python {__file__} {args.input} {args.output} --validate")
        

    open("train.txt", "w").close()
# def batch_process_sav_videos(input_dir, output_dir, frame_range=None, stride=1, file_pattern="*.json"):
    """
    批量处理文件夹下的所有 SA-V 视频 JSON 文件
    
    参数:
    - input_dir: 输入文件夹路径，包含多个 .json 文件
    - output_dir: 输出根目录
    - frame_range: 帧范围 (start, end)，None 表示处理所有帧
    - stride: 帧间隔，1表示处理每一帧
    - file_pattern: 文件匹配模式，默认 "*.json"
    
    返回:
    - 处理结果统计
    """
    input_dir = args.input_dir
    output_dir = args.output_dir
    file_pattern = args.file_pattern
    stride = args.stride
    
    print(f"=== 批量处理 SA-V 视频文件 ===")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"文件模式: {file_pattern}")
    print(f"帧范围: {frame_range}")
    print(f"帧间隔: {stride}")
    
    # 确保输入目录存在
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        return None
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有匹配的 JSON 文件
    search_pattern = os.path.join(input_dir, file_pattern)
    json_files = glob.glob(search_pattern)
    
    if not json_files:
        print(f"错误: 在 {input_dir} 中未找到匹配 {file_pattern} 的文件")
        return None
    
    print(f"\n找到 {len(json_files)} 个 JSON 文件:")
    for i, file_path in enumerate(json_files, 1):
        print(f"  {i}. {os.path.basename(file_path)}")
    
    # 处理统计
    results = {
        'total_files': len(json_files),
        'successful_files': 0,
        'failed_files': 0,
        'processed_videos': [],
        'failed_videos': [],
        'total_masklets_processed': 0,
        'total_masklets_skipped': 0
    }
    
    # 逐个处理文件
    for i, json_file in enumerate(json_files, 1):
        print(f"\n{'='*60}")
        print(f"处理文件 {i}/{len(json_files)}: {os.path.basename(json_file)}")
        print(f"{'='*60}")
        
        try:
            # 调用单个文件处理函数
            result_dir = process_sav_video_by_masklet(
                json_file_path=json_file,
                output_dir=output_dir,
                frame_range=frame_range,
                stride=stride
            )
            
            if result_dir:
                results['successful_files'] += 1
                results['processed_videos'].append({
                    'file': os.path.basename(json_file),
                    'output_dir': result_dir
                })
                print(f"✓ 文件处理成功: {os.path.basename(json_file)}")
            else:
                results['failed_files'] += 1
                results['failed_videos'].append({
                    'file': os.path.basename(json_file),
                    'error': '处理函数返回空结果'
                })
                print(f"✗ 文件处理失败: {os.path.basename(json_file)}")
                
        except Exception as e:
            results['failed_files'] += 1
            results['failed_videos'].append({
                'file': os.path.basename(json_file),
                'error': str(e)
            })
            print(f"✗ 文件处理异常: {os.path.basename(json_file)} - {e}")
            # exit()
    
    # 输出最终统计
    print(f"\n{'='*60}")
    print(f"=== 批量处理完成 ===")
    print(f"{'='*60}")
    print(f"总文件数: {results['total_files']}")
    print(f"成功处理: {results['successful_files']}")
    print(f"处理失败: {results['failed_files']}")
    print(f"成功率: {results['successful_files']/results['total_files']*100:.1f}%")
    
    if results['processed_videos']:
        print(f"\n=== 成功处理的视频 ===")
        for video in results['processed_videos']:
            print(f"  ✓ {video['file']} -> {video['output_dir']}")
    
    if results['failed_videos']:
        print(f"\n=== 处理失败的视频 ===")
        for video in results['failed_videos']:
            print(f"  ✗ {video['file']}: {video['error']}")
    
    print(f"\n所有输出保存在: {output_dir}")
    
    return results

if __name__ == "__main__":
    main()
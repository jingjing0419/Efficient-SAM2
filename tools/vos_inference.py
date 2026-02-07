# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import enum
import os
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import random
import time
import logging
import sys
sys.path.append("./sam2")
sys.path.append("/home/zhangjing/efficient-sam2/")
from build_model.window_bypass import build_WB_model
from bypass.bypass_modeling import build_bypass_model, build_small_bypass_model

# the PNG palette for DAVIS 2017 dataset
DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"

def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f'Random seed set as {seed}')

def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette


def save_ann_png(path, mask, palette):
    """Save a mask as a PNG file with the given palette."""
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    output_mask = Image.fromarray(mask)
    output_mask.putpalette(palette)
    output_mask.save(path)


def get_per_obj_mask(mask):
    """Split a mask into per-object masks."""
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids > 0].tolist()
    per_obj_mask = {object_id: (mask == object_id) for object_id in object_ids}
    return per_obj_mask


def put_per_obj_mask(per_obj_mask, height, width):
    """Combine per-object masks into a single mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    object_ids = sorted(per_obj_mask)[::-1]
    for object_id in object_ids:
        object_mask = per_obj_mask[object_id]
        object_mask = object_mask.reshape(height, width)
        mask[object_mask] = object_id
    return mask


def load_masks_from_dir(
    input_mask_dir, video_name, frame_name, per_obj_png_file, allow_missing=False
):
    """Load masks from a directory as a dict of per-object masks."""
    if not per_obj_png_file:
        input_mask_path = os.path.join(input_mask_dir, video_name, f"{frame_name}.png")
        if allow_missing and not os.path.exists(input_mask_path):
            return {}, None
        input_mask, input_palette = load_ann_png(input_mask_path)
        per_obj_input_mask = get_per_obj_mask(input_mask)
    else:
        per_obj_input_mask = {}
        input_palette = None
        # each object is a directory in "{object_id:%03d}" format
        for object_name in os.listdir(os.path.join(input_mask_dir, video_name)):
            object_id = int(object_name)
            input_mask_path = os.path.join(
                input_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            if allow_missing and not os.path.exists(input_mask_path):
                continue
            input_mask, input_palette = load_ann_png(input_mask_path)
            per_obj_input_mask[object_id] = input_mask > 0

    return per_obj_input_mask, input_palette


def save_masks_to_dir(
    output_mask_dir,
    video_name,
    frame_name,
    per_obj_output_mask,
    height,
    width,
    per_obj_png_file,
    output_palette,
):
    """Save masks to a directory as PNG files."""
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    if not per_obj_png_file:
        output_mask = put_per_obj_mask(per_obj_output_mask, height, width)
        output_mask_path = os.path.join(
            output_mask_dir, video_name, f"{frame_name}.png"
        )
        save_ann_png(output_mask_path, output_mask, output_palette)
    else:
        for object_id, object_mask in per_obj_output_mask.items():
            object_name = f"{object_id:03d}"
            os.makedirs(
                os.path.join(output_mask_dir, video_name, object_name),
                exist_ok=True,
            )
            output_mask = object_mask.reshape(height, width).astype(np.uint8)
            output_mask_path = os.path.join(
                output_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            save_ann_png(output_mask_path, output_mask, output_palette)


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_inference(
    predictor,
    base_video_dir,
    input_mask_dir,
    output_mask_dir,
    video_name,
    score_thresh=0.0,
    use_all_masks=False,
    per_obj_png_file=False,
):
    """Run VOS inference on a single video with the given predictor."""
    # load the video frames and initialize the inference state on this video
    
    predictor.time_log = {}
    predictor.image_encoder.FW_time_log = []
    predictor.memory_attention.FW_time_log = []
    predictor.memory_encoder.FW_time_log = []
    predictor.sam_mask_decoder.FW_time_log = []
    predictor.image_encoder.trunk.FW_time_log={}
    
    # for blk in predictor.image_encoder.trunk.blocks:
    #     blk.FW_time_log = []
    
    
    video_dir = os.path.join(base_video_dir, video_name)
    frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    inference_state = predictor.init_state(
        video_path=video_dir, async_loading_frames=False
    )
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    input_palette = None

    # fetch mask inputs from input_mask_dir (either only mask for the first frame, or all available masks)
    if not use_all_masks:
        # use only the first video's ground-truth mask as the input mask
        input_frame_inds = [0]
    else:
        # use all mask files available in the input_mask_dir as the input masks
        if not per_obj_png_file:
            input_frame_inds = [
                idx
                for idx, name in enumerate(frame_names)
                if os.path.exists(
                    os.path.join(input_mask_dir, video_name, f"{name}.png")
                )
            ]
        else:
            input_frame_inds = [
                idx
                for object_name in os.listdir(os.path.join(input_mask_dir, video_name))
                for idx, name in enumerate(frame_names)
                if os.path.exists(
                    os.path.join(input_mask_dir, video_name, object_name, f"{name}.png")
                )
            ]
        # check and make sure we got at least one input frame
        if len(input_frame_inds) == 0:
            raise RuntimeError(
                f"In {video_name=}, got no input masks in {input_mask_dir=}. "
                "Please make sure the input masks are available in the correct format."
            )
        input_frame_inds = sorted(set(input_frame_inds))

    # add those input masks to SAM 2 inference state before propagation
    object_ids_set = None
    for i, input_frame_idx in enumerate(input_frame_inds):
        
        try:
            per_obj_input_mask, input_palette = load_masks_from_dir(
                input_mask_dir=input_mask_dir,
                video_name=video_name,
                frame_name=frame_names[input_frame_idx],
                per_obj_png_file=per_obj_png_file,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"In {video_name=}, failed to load input mask for frame {input_frame_idx=}. "
                "Please add the `--track_object_appearing_later_in_video` flag "
                "for VOS datasets that don't have all objects to track appearing "
                "in the first frame (such as LVOS or YouTube-VOS)."
            ) from e
        # get the list of object ids to track from the first input frame
        if object_ids_set is None:
            object_ids_set = set(per_obj_input_mask)
        for object_id, object_mask in per_obj_input_mask.items():
            # check and make sure no new object ids appear only in later frames
            if object_id not in object_ids_set:
                raise RuntimeError(
                    f"In {video_name=}, got a new {object_id=} appearing only in a "
                    f"later {input_frame_idx=} (but not appearing in the first frame). "
                    "Please add the `--track_object_appearing_later_in_video` flag "
                    "for VOS datasets that don't have all objects to track appearing "
                    "in the first frame (such as LVOS or YouTube-VOS)."
                )
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=input_frame_idx,
                obj_id=object_id,
                mask=object_mask,
            )
        
    # check and make sure we have at least one object to track
    if object_ids_set is None or len(object_ids_set) == 0:
        raise RuntimeError(
            f"In {video_name=}, got no object ids on {input_frame_inds=}. "
            "Please add the `--track_object_appearing_later_in_video` flag "
            "for VOS datasets that don't have all objects to track appearing "
            "in the first frame (such as LVOS or YouTube-VOS)."
        )
    # run propagation throughout the video and collect the results in a dict
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    output_palette = input_palette or DAVIS_PALETTE
    video_segments = {}  # video_segments contains the per-frame segmentation results
    st = time.time()
    print('start time:{}'.format(st))
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        per_obj_output_mask = {
            out_obj_id: (out_mask_logits[i] > score_thresh).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        # print(per_obj_output_mask)
        
        video_segments[out_frame_idx] = per_obj_output_mask
        # torch.save(per_obj_output_mask, 'per_obj_output_mask.pth')
        # exit()
    ed = time.time()
    print('end time:{}'.format(ed))
    print('inference time cost:{}'.format(ed-st))
    # write the output masks as palette PNG files to output_mask_dir
    for out_frame_idx, per_obj_output_mask in video_segments.items():
        save_masks_to_dir(
            output_mask_dir=output_mask_dir,
            video_name=video_name,
            frame_name=frame_names[out_frame_idx],
            per_obj_output_mask=per_obj_output_mask,
            height=height,
            width=width,
            per_obj_png_file=per_obj_png_file,
            output_palette=output_palette,
        )

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_inference_wj(
    predictor,
    base_video_dir,
    input_mask_dir,
    output_mask_dir,
    video_name,
    score_thresh=0.0,
    use_all_masks=False,
    per_obj_png_file=False,
):
    """Run VOS inference on a single video with the given predictor."""
    # load the video frames and initialize the inference state on this video
    predictor.time_log = {}
    predictor.image_encoder.FW_time_log = []
    predictor.memory_attention.FW_time_log = []
    predictor.memory_encoder.FW_time_log = []
    predictor.sam_mask_decoder.FW_time_log = []
    predictor.image_encoder.trunk.FW_time_log={}
    
    video_dir = os.path.join(base_video_dir, video_name)
    frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    inference_state = predictor.init_state(
        video_path=video_dir, async_loading_frames=False
    )
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    input_palette = None

    # fetch mask inputs from input_mask_dir (either only mask for the first frame, or all available masks)
    if not use_all_masks:
        # use only the first video's ground-truth mask as the input mask
        input_frame_inds = [0]
    else:
        # use all mask files available in the input_mask_dir as the input masks
        if not per_obj_png_file:
            input_frame_inds = [
                idx
                for idx, name in enumerate(frame_names)
                if os.path.exists(
                    os.path.join(input_mask_dir, video_name, f"{name}.png")
                )
            ]
        else:
            input_frame_inds = [
                idx
                for object_name in os.listdir(os.path.join(input_mask_dir, video_name))
                for idx, name in enumerate(frame_names)
                if os.path.exists(
                    os.path.join(input_mask_dir, video_name, object_name, f"{name}.png")
                )
            ]
        # check and make sure we got at least one input frame
        if len(input_frame_inds) == 0:
            raise RuntimeError(
                f"In {video_name=}, got no input masks in {input_mask_dir=}. "
                "Please make sure the input masks are available in the correct format."
            )
        input_frame_inds = sorted(set(input_frame_inds))

    # add those input masks to SAM 2 inference state before propagation
    object_ids_set = None
    for i, input_frame_idx in enumerate(input_frame_inds):
        
        try:
            per_obj_input_mask, input_palette = load_masks_from_dir(
                input_mask_dir=input_mask_dir,
                video_name=video_name,
                frame_name=frame_names[input_frame_idx],
                per_obj_png_file=per_obj_png_file,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"In {video_name=}, failed to load input mask for frame {input_frame_idx=}. "
                "Please add the `--track_object_appearing_later_in_video` flag "
                "for VOS datasets that don't have all objects to track appearing "
                "in the first frame (such as LVOS or YouTube-VOS)."
            ) from e
        # get the list of object ids to track from the first input frame
        if object_ids_set is None:
            object_ids_set = set(per_obj_input_mask)
        for object_id, object_mask in per_obj_input_mask.items():
            # check and make sure no new object ids appear only in later frames
            if object_id not in object_ids_set:
                raise RuntimeError(
                    f"In {video_name=}, got a new {object_id=} appearing only in a "
                    f"later {input_frame_idx=} (but not appearing in the first frame). "
                    "Please add the `--track_object_appearing_later_in_video` flag "
                    "for VOS datasets that don't have all objects to track appearing "
                    "in the first frame (such as LVOS or YouTube-VOS)."
                )
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=input_frame_idx,
                obj_id=object_id,
                mask=object_mask,
            )
        
    # check and make sure we have at least one object to track
    if object_ids_set is None or len(object_ids_set) == 0:
        raise RuntimeError(
            f"In {video_name=}, got no object ids on {input_frame_inds=}. "
            "Please add the `--track_object_appearing_later_in_video` flag "
            "for VOS datasets that don't have all objects to track appearing "
            "in the first frame (such as LVOS or YouTube-VOS)."
        )
    # run propagation throughout the video and collect the results in a dict
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    output_palette = input_palette or DAVIS_PALETTE
    video_segments = {}  # video_segments contains the per-frame segmentation results
    torch.cuda.synchronize()
    st = time.time()
    # print('start time:{}'.format(st))
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video_wj(
        inference_state
    ):
        per_obj_output_mask = {
            out_obj_id: (out_mask_logits[i] > score_thresh).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        # print(per_obj_output_mask)
        
        video_segments[out_frame_idx] = per_obj_output_mask
        # torch.save(per_obj_output_mask, 'per_obj_output_mask.pth')
        # exit()
    torch.cuda.synchronize()
    ed = time.time()
    # print('end time:{}'.format(ed))
    print('inference time cost:{}'.format(ed-st))
    # write the output masks as palette PNG files to output_mask_dir
    for out_frame_idx, per_obj_output_mask in video_segments.items():
        save_masks_to_dir(
            output_mask_dir=output_mask_dir,
            video_name=video_name,
            frame_name=frame_names[out_frame_idx],
            per_obj_output_mask=per_obj_output_mask,
            height=height,
            width=width,
            per_obj_png_file=per_obj_png_file,
            output_palette=output_palette,
        )


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_separate_inference_per_object(
    predictor,
    base_video_dir,
    input_mask_dir,
    output_mask_dir,
    video_name,
    score_thresh=0.0,
    use_all_masks=False,
    per_obj_png_file=False,
):
    """
    Run VOS inference on a single video with the given predictor.

    Unlike `vos_inference`, this function run inference separately for each object
    in a video, which could be applied to datasets like LVOS or YouTube-VOS that
    don't have all objects to track appearing in the first frame (i.e. some objects
    might appear only later in the video).
    """
    # load the video frames and initialize the inference state on this video
    video_dir = os.path.join(base_video_dir, video_name)
    frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    inference_state = predictor.init_state(
        video_path=video_dir, async_loading_frames=False
    )
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    input_palette = None

    # collect all the object ids and their input masks
    inputs_per_object = defaultdict(dict)
    for idx, name in enumerate(frame_names):
        if per_obj_png_file or os.path.exists(
            os.path.join(input_mask_dir, video_name, f"{name}.png")
        ):
            per_obj_input_mask, input_palette = load_masks_from_dir(
                input_mask_dir=input_mask_dir,
                video_name=video_name,
                frame_name=frame_names[idx],
                per_obj_png_file=per_obj_png_file,
                allow_missing=True,
            )
            for object_id, object_mask in per_obj_input_mask.items():
                # skip empty masks
                if not np.any(object_mask):
                    continue
                # if `use_all_masks=False`, we only use the first mask for each object
                if len(inputs_per_object[object_id]) > 0 and not use_all_masks:
                    continue
                print(f"adding mask from frame {idx} as input for {object_id=}")
                inputs_per_object[object_id][idx] = object_mask

    # run inference separately for each object in the video
    object_ids = sorted(inputs_per_object)
    output_scores_per_object = defaultdict(dict)
    for object_id in object_ids:
        # add those input masks to SAM 2 inference state before propagation
        input_frame_inds = sorted(inputs_per_object[object_id])
        predictor.reset_state(inference_state)
        for input_frame_idx in input_frame_inds:
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=input_frame_idx,
                obj_id=object_id,
                mask=inputs_per_object[object_id][input_frame_idx],
            )

        # run propagation throughout the video and collect the results in a dict
        for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=min(input_frame_inds),
            reverse=False,
        ):
            obj_scores = out_mask_logits.cpu().numpy()
            output_scores_per_object[object_id][out_frame_idx] = obj_scores

    # post-processing: consolidate the per-object scores into per-frame masks
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    output_palette = input_palette or DAVIS_PALETTE
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for frame_idx in range(len(frame_names)):
        scores = torch.full(
            size=(len(object_ids), 1, height, width),
            fill_value=-1024.0,
            dtype=torch.float32,
        )
        for i, object_id in enumerate(object_ids):
            if frame_idx in output_scores_per_object[object_id]:
                scores[i] = torch.from_numpy(
                    output_scores_per_object[object_id][frame_idx]
                )

        if not per_obj_png_file:
            scores = predictor._apply_non_overlapping_constraints(scores)
        per_obj_output_mask = {
            object_id: (scores[i] > score_thresh).cpu().numpy()
            for i, object_id in enumerate(object_ids)
        }
        video_segments[frame_idx] = per_obj_output_mask

    # write the output masks as palette PNG files to output_mask_dir
    for frame_idx, per_obj_output_mask in video_segments.items():
        save_masks_to_dir(
            output_mask_dir=output_mask_dir,
            video_name=video_name,
            frame_name=frame_names[frame_idx],
            per_obj_output_mask=per_obj_output_mask,
            height=height,
            width=width,
            per_obj_png_file=per_obj_png_file,
            output_palette=output_palette,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sam2_cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
        help="SAM 2 model configuration file",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default="./checkpoints/sam2.1_hiera_base_plus.pt",
        help="path to the SAM 2 model checkpoint",
    )
    parser.add_argument(
        "--base_video_dir",
        type=str,
        default='/mnt/data/zhangjing/SAV_dataset/sav_test/JPEGImages_24fps',
        help="directory containing videos (as JPEG files) to run VOS prediction on",
    )
    parser.add_argument(
        "--input_mask_dir",
        type=str,
        default='/mnt/data/zhangjing/SAV_dataset/sav_test/Annotations_6fps/',
        help="directory containing input masks (as PNG files) of each video",
    )
    parser.add_argument(
        "--video_list_file",
        type=str,
        default=None,
        help="text file containing the list of video names to run VOS prediction on",
    )
    parser.add_argument(
        "--output_mask_dir",
        type=str,
        default='./outputs/sav_test_pred_pngs',
        help="directory to save the output masks (as PNG files)",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.0,
        help="threshold for the output mask logits (default: 0.0)",
    )
    parser.add_argument(
        "--use_all_masks",
        action="store_true",
        help="whether to use all available PNG files in input_mask_dir "
        "(default without this flag: just the first PNG file as input to the SAM 2 model; "
        "usually we don't need this flag, since semi-supervised VOS evaluation usually takes input from the first frame only)",
    )
    parser.add_argument(
        "--per_obj_png_file",
        # action="store_true",
        type=bool,
        default=True,
        help="whether use separate per-object PNG files for input and output masks "
        "(default without this flag: all object masks are packed into a single PNG file on each frame following DAVIS format; "
        "note that the SA-V dataset stores each object mask as an individual PNG file and requires this flag)",
    )
    parser.add_argument(
        "--apply_postprocessing",
        action="store_true",
        help="whether to apply postprocessing (e.g. hole-filling) to the output masks "
        "(we don't apply such post-processing in the SAM 2 model evaluation)",
    )
    parser.add_argument(
        "--track_object_appearing_later_in_video",
        action="store_true",
        help="whether to track objects that appear later in the video (i.e. not on the first frame; "
        "some VOS datasets like LVOS or YouTube-VOS don't have all objects appearing in the first frame)",
    )
    parser.add_argument(
        "--use_vos_optimized_video_predictor",
        action="store_true",
        help="whether to use vos optimized video predictor with all modules compiled",
    )
    parser.add_argument(
        '--work_dir',
        type=str,
        default='result/tmp/',
        help='path to save log and result')
    parser.add_argument(
        '--selected_layers',
        type=int,
        nargs='+',
        default=[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
        # default=[13,17,21],
        help='path to save log and result')
    parser.add_argument(
        '--match_layers',
        type=int,
        nargs='+',
        # default=[7,9,14,18],
        # default=[7,8,9],
        # default=[7,8,9,10,11,12,14,15,16,18,19,20],
        default=[7,13],
        # default=[13,17,21],
        help='path to save log and result')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.88,
        help='path to save log and result')
    parser.add_argument(
        "--apply_WB",
        action="store_true")
    
    parser.add_argument(
        "--prune_memory",
        action="store_true")
    
    parser.add_argument(
        "--hiera_WB",
        action="store_true")
    
    parser.add_argument(
        "--apply_bypass",
        action="store_true")
    
    parser.add_argument(
        "--dilate_mask",
        action="store_true")
    
    parser.add_argument(
        '--dilate_kernel_size',
        type=int,
        default=5)
    
    parser.add_argument(
        '--bypass_ckpt',
        type=str,
        default='./bypass/ckpt/bypass_20250701_121031_MPWS_s16_T30_D5_EP3.pth',
        help='path to bypass checkpoint')
    
    parser.add_argument(
        '--small_bypass',
        action='store_true',
        default=False)

    
    
    args = parser.parse_args()

    set_seed(1234)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    
    log_file = os.path.join(args.work_dir, f'{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 设置日志格式
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    args.logger_name = args.work_dir+f'{timestamp}'
    args.output_mask_dir = args.output_mask_dir + f'_{timestamp}'
    logger = logging.getLogger(args.logger_name)
    logger.info('time stamp:{}'.format(timestamp))
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")
        
    # torch.cuda.set_per_process_memory_fraction(0.3, device=0)

    # if we use per-object PNG files, they could possibly overlap in inputs and outputs
    hydra_overrides_extra = [
        "++model.non_overlap_masks=" + ("false" if args.per_obj_png_file else "true")
    ]
    predictor = build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        apply_postprocessing=args.apply_postprocessing,
        hydra_overrides_extra=hydra_overrides_extra,
        vos_optimized=args.use_vos_optimized_video_predictor,
    )
    
    logger.info(predictor)
    if args.apply_WB:
        predictor.init_memory_info(enable_MeP_info=True)
        build_WB_model(args, predictor, selected_layers=args.selected_layers)
    else:
        predictor.init_memory_info(enable_MeP_info=False)
    # logger.info(predictor)
    
    
    if args.apply_bypass:
        build_bypass_model(args, predictor, training=False)
    # predictor.cuda()
    # logger.info(predictor)
    # exit()
        predictor.image_encoder.trunk.blocks[20].bypass_branch.load_state_dict(torch.load(args.bypass_ckpt))
        print('load bypass chackpoint from ', args.bypass_ckpt)
        predictor.image_encoder.trunk.blocks[20].bypass_branch.eval()
        bypass_params = sum(p.numel() for p in predictor.image_encoder.trunk.blocks[20].bypass_branch.parameters())
        print('The parameter of bypass: {}'.format(bypass_params))
    # exit()
    
        # build_bypass_model(args, predictor, training=False)
    # predictor.cuda()
    # logger.info(predictor)
    # exit()
        # predictor.image_encoder.trunk.blocks[20].bypass_branch.load_state_dict(torch.load(args.bypass_ckpt))
        # print('load bypass chackpoint from ', args.bypass_ckpt)
        # predictor.image_encoder.trunk.blocks[20].bypass_branch.eval()
    predictor.dilate_mask = False
    if args.dilate_mask:
        predictor.dilate_mask = True
        predictor.dilate_kernel_size = args.dilate_kernel_size
    
     
    if args.use_all_masks:
        print("using all available masks in input_mask_dir as input to the SAM 2 model")
    else:
        print(
            "using only the first frame's mask in input_mask_dir as input to the SAM 2 model"
        )
    # if a video list file is provided, read the video names from the file
    # (otherwise, we use all subdirectories in base_video_dir)
    if args.video_list_file is not None:
        with open(args.video_list_file, "r") as f:
            video_names = [v.strip() for v in f.readlines()]
    else:
        video_names = [
            p
            for p in os.listdir(args.base_video_dir)
            if os.path.isdir(os.path.join(args.base_video_dir, p))
        ]
        
    logger.info(predictor)
    # predictor.compile()
    # 为了保持两张卡的抽样测试结果一致，直接规定了顺序
    video_names = ['sav_021093', 'sav_017694', 'sav_045541', 'sav_043844', 'sav_001379', 'sav_028748', 'sav_032329', 'sav_024938', 'sav_036919', 'sav_048456', 'sav_048782', 'sav_009307', 'sav_004507', 'sav_039782', 'sav_026821', 'sav_011742', 'sav_011282', 'sav_026969', 'sav_025236', 'sav_041589', 'sav_028579', 'sav_020515', 'sav_044580', 'sav_054217', 'sav_004381', 'sav_023982', 'sav_053239', 'sav_031773', 'sav_003798', 'sav_040157', 'sav_023742', 'sav_042227', 'sav_020688', 'sav_033857', 'sav_003419', 'sav_035939', 'sav_047286', 'sav_008977', 'sav_030621', 'sav_037938', 'sav_012593', 'sav_016333', 'sav_037370', 'sav_026469', 'sav_031273', 'sav_050141', 'sav_051362', 'sav_016568', 'sav_024562', 'sav_010280', 'sav_025082', 'sav_024119', 'sav_001148', 'sav_000818', 'sav_030786', 'sav_031285', 'sav_010164', 'sav_025271', 'sav_027979', 'sav_029809', 'sav_043183', 'sav_044777', 'sav_003418', 'sav_042890', 'sav_021407', 'sav_041748', 'sav_046799', 'sav_007943', 'sav_054565', 'sav_018048', 'sav_015763', 'sav_054282', 'sav_039416', 'sav_004550', 'sav_024568', 'sav_004755', 'sav_049155', 'sav_007311', 'sav_052661', 'sav_022374', 'sav_002789', 'sav_037788', 'sav_002636', 'sav_044104', 'sav_033343', 'sav_043348', 'sav_036300', 'sav_008286', 'sav_030521', 'sav_031901', 'sav_006301', 'sav_023013', 'sav_042517', 'sav_022663', 'sav_037495', 'sav_010923', 'sav_007543', 'sav_020844', 'sav_026800', 'sav_023860', 'sav_012344', 'sav_040678', 'sav_009687', 'sav_026020', 'sav_030325', 'sav_011754', 'sav_051401', 'sav_011787', 'sav_013074', 'sav_052991', 'sav_017171', 'sav_043131', 'sav_042703', 'sav_053716', 'sav_015027', 'sav_054440', 'sav_032237', 'sav_053054', 'sav_045630', 'sav_053269', 'sav_032655', 'sav_054399', 'sav_000208', 'sav_043898', 'sav_010681', 'sav_009146', 'sav_015587', 'sav_010931', 'sav_031935', 'sav_008860', 'sav_013124', 'sav_038325', 'sav_053954', 'sav_031317', 'sav_052793', 'sav_049186', 'sav_050499', 'sav_026256', 'sav_053005', 'sav_027066', 'sav_052215', 'sav_047904', 'sav_000425', 'sav_013624', 'sav_032837', 'sav_021623', 'sav_033373', 'sav_041173', 'sav_003191', 'sav_045675']
    logger.info(f"running VOS prediction on {len(video_names)} videos:\n{video_names}")
    time_log = {}
    FW_time_log = {}
    for n_video, video_name in enumerate(video_names):
        print(f"\n{n_video + 1}/{len(video_names)} - running on {video_name}")
        # if n_video == 0:
        #     continue
        if n_video % 15 != 0:
            continue
        # else:
        #     print(video_name)
        #     continue
        if not args.track_object_appearing_later_in_video:
            if args.apply_WB:
                vos_inference_wj(
                    predictor=predictor,
                    base_video_dir=args.base_video_dir,
                    input_mask_dir=args.input_mask_dir,
                    output_mask_dir=args.output_mask_dir,
                    video_name=video_name,
                    score_thresh=args.score_thresh,
                    use_all_masks=args.use_all_masks,
                    per_obj_png_file=args.per_obj_png_file,
                )
                # print(FW_time_log[video_name]['Hiera_Blks'])
            else:
                vos_inference(
                    predictor=predictor,
                    base_video_dir=args.base_video_dir,
                    input_mask_dir=args.input_mask_dir,
                    output_mask_dir=args.output_mask_dir,
                    video_name=video_name,
                    score_thresh=args.score_thresh,
                    use_all_masks=args.use_all_masks,
                    per_obj_png_file=args.per_obj_png_file,
                )
        else:
            vos_separate_inference_per_object(
                predictor=predictor,
                base_video_dir=args.base_video_dir,
                input_mask_dir=args.input_mask_dir,
                output_mask_dir=args.output_mask_dir,
                video_name=video_name,
                score_thresh=args.score_thresh,
                use_all_masks=args.use_all_masks,
                per_obj_png_file=args.per_obj_png_file,
            )

    print(
        f"completed VOS prediction on {len(video_names)} videos -- "
        f"output masks saved to {args.output_mask_dir}"
    )


if __name__ == "__main__":
    main()

# coding=utf-8
import cv2
import sys
import numpy as np 
import torch 
from segment_anything import SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
import time 

def prepare_image(rgb_img, transform, device):
    rgb_img = transform.apply_image(rgb_img)
    image = torch.as_tensor(rgb_img, device=device) 
    return image.permute(2, 0, 1).contiguous()   # 3hw 深拷贝

# prompt inference sam
def box_only_prompt(sam, image, box_array, box_label, device, inference_size):
    res_label_map = np.zeros_like(image[:,:,0]).astype(np.uint8)
    resize_transform = ResizeLongestSide(inference_size)  # default=sam.image_encoder.img_size=1024
    input_boxes = torch.from_numpy(box_array).to(device=device)  # (nums, 4)
    batched_input = [{'image': prepare_image(image, resize_transform, device),'boxes': resize_transform.apply_boxes_torch(input_boxes, image.shape[:2]), 'original_size': image.shape[:2]}]
    try:
        masks = sam(batched_input, multimask_output=False)[0]  # (batch_size=1so取[0]) x (num_predicted_masks_per_input) x H x W
    except:
        return res_label_map
    print('box prompt~ ')
    mask_labels = masks['masks'].cpu().numpy()
    mask_labels = np.squeeze(mask_labels, 1)  # 去掉冗余的那一维 第二维   -> (num_predicted_masks_per_input) x H x W
    labs = mask_labels.shape[0]
    for i in range(labs):
        res_label_map[mask_labels[i]==True] = box_label[i]
    
    return res_label_map


def points_only_prompt(sam, image, points, point_labels, device, inference_size):
    res_label_map = np.zeros_like(image[:,:,0]).astype(np.uint8)
    resize_transform = ResizeLongestSide(inference_size)   
    points = torch.from_numpy(points).to(device=device)
    lab_info = list(set(point_labels))
    points = points.unsqueeze(0)   
    point_labels = torch.from_numpy(point_labels).to(device=device)
    point_labels = point_labels.unsqueeze(0)   
    batched_input = [{'image': prepare_image(image, resize_transform, device), 'point_coords': resize_transform.apply_coords_torch(points, image.shape[:2]), 'point_labels': point_labels, 'original_size': image.shape[:2]}]
    try:
        masks = sam(batched_input, multimask_output=False)[0]  # (batch_size==1so取[0]) x (num_predicted_masks_per_input) x H x W
    except:
        return res_label_map
    print('points prompt~')
    mask_labels = masks['masks'].cpu().numpy()  # 1x(num_predicted_masks_per_input) x H x W 第2维是冗余的..
    mask_labels = np.squeeze(mask_labels, 1)    # (num_predicted_masks_per_input) x H x W
    labs = mask_labels.shape[0]
    for i in range(labs):
        cur_label = lab_info[i]  # 这里可能出现label对错的问题, 理论上应该是点一处点就出一处mask的! 每个连通域该下cls不麻烦的~ 
        res_label_map[mask_labels[i]==True] = cur_label

    return res_label_map

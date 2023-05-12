# coding=utf-8
# sam_model.py
import cv2
import sys
import numpy as np 
import torch 
from segment_anything import SamAutomaticMaskGenerator, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
import time 

def SAM_mask(rgb_img, sam):
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(rgb_img)   # list, 每个元素是一个分割结果
    return masks

def prepare_image(rgb_img, transform, device):
    rgb_img = transform.apply_image(rgb_img)
    image = torch.as_tensor(rgb_img, device=device) 
    return image.permute(2, 0, 1).contiguous()   # 3hw 深拷贝



# prompt inference sam
def box_only_prompt(sam, image, box_array, device, inference_size):
    res_label_map = np.zeros_like(image[:,:,0]).astype(np.uint8)
    resize_transform = ResizeLongestSide(inference_size)  # default=sam.image_encoder.img_size=1024
    input_boxes = torch.from_numpy(box_array).to(device=device)  # (nums, 4)
    batched_input = [{'image': prepare_image(image, resize_transform, device),'boxes': resize_transform.apply_boxes_torch(input_boxes, image.shape[:2]), 'original_size': image.shape[:2]}]
    try:
        # start = time.time()
        masks = sam(batched_input, multimask_output=True)[0]
        # print(masks['iou_predictions'], masks['low_res_logits'])
        # end = time.time()
        # print('runing time {}'.format(end-start))
    except:
        return res_label_map
    print('box prompt~ ')
    best_mask_inds = [np.argmax(mask_iou) for mask_iou in masks['iou_predictions'].cpu().numpy()]
    mask_labels = masks['masks'].cpu().numpy()
    mask_res = []
    for i, best_id in enumerate(best_mask_inds):
        mask_ = mask_labels[i][best_id]
        mask_res.append(mask_)
    first_mask = mask_res[0]
    for mask_lab in mask_res[1:]:
        first_mask = np.logical_or(first_mask, mask_lab)  # true or false = true 
    res_label_map[first_mask==True] = 255

    return res_label_map


def points_only_prompt(sam, image, points, point_labels, device, inference_size):
    '''
    points: array
    point_labels: array
    '''
    res_label_map = np.zeros_like(image[:,:,0]).astype(np.uint8)
    resize_transform = ResizeLongestSide(inference_size)  # default=sam.image_encoder.img_size=1024
    points = torch.from_numpy(points).to(device=device)
    points = points.unsqueeze(0)   
    point_labels = torch.from_numpy(point_labels).to(device=device)
    point_labels = point_labels.unsqueeze(0)   
    batched_input = [{'image': prepare_image(image, resize_transform, device), 'point_coords': resize_transform.apply_coords_torch(points, image.shape[:2]), 'point_labels': point_labels, 'original_size': image.shape[:2]}]
    masks = sam(batched_input, multimask_output=True)[0]
    try:
        masks = sam(batched_input, multimask_output=True)[0]
    except:
        return res_label_map
    print('points prompt~')
    best_mask_inds = [np.argmax(mask_iou) for mask_iou in masks['iou_predictions'].cpu().numpy()]
    mask_labels = masks['masks'].cpu().numpy()
    mask_res = []
    for i, best_id in enumerate(best_mask_inds):
        mask_ = mask_labels[i][best_id]
        mask_res.append(mask_)
    first_mask = mask_res[0]
    for mask_lab in mask_res[1:]:
        first_mask = np.logical_or(first_mask, mask_lab)  # true or false = true 
    res_label_map[first_mask==True] = 255

    return res_label_map


def points_box_prompt(sam, image, points, point_labels, box_array, device, inference_size):
    res_label_map = np.zeros_like(image[:,:,0]).astype(np.uint8)
    resize_transform = ResizeLongestSide(inference_size)  # default=sam.image_encoder.img_size=1024
    input_boxes = torch.from_numpy(box_array).to(device=device)
    points = torch.from_numpy(points).to(device=device)
    # 模仿sam的源码, 需要对points和label都添加一维.
    points = points.unsqueeze(0)  
    point_labels = torch.from_numpy(point_labels).to(device=device)
    point_labels = point_labels.unsqueeze(0) 
    batched_input = [{'image': prepare_image(image, resize_transform, device), 'boxes': resize_transform.apply_boxes_torch(input_boxes, image.shape[:2]), 'point_coords': resize_transform.apply_coords_torch(points, image.shape[:2]), 'point_labels': point_labels, 'original_size': image.shape[:2]}]
    try:
        masks = sam(batched_input, multimask_output=True)[0]
    except:
        return res_label_map
    print('box+points prompt~')
    best_mask_inds = [np.argmax(mask_iou) for mask_iou in masks['iou_predictions'].cpu().numpy()]
    mask_labels = masks['masks'].cpu().numpy()
    mask_res = []
    for i, best_id in enumerate(best_mask_inds):
        mask_ = mask_labels[i][best_id]
        mask_res.append(mask_)
    first_mask = mask_res[0]
    for mask_lab in mask_res[1:]:
        first_mask = np.logical_or(first_mask, mask_lab)  # true or false = true 
    res_label_map[first_mask==True] = 255

    return res_label_map


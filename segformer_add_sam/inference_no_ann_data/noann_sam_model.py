# coding=utf-8
# noann_sam_model.py   
import cv2
import sys
import numpy as np 
import torch 
from segment_anything import SamPredictor, SamAutomaticMaskGenerator
from segment_anything.utils.transforms import ResizeLongestSide
import time 

def prepare_image(rgb_img, transform, device):
    rgb_img = transform.apply_image(rgb_img)
    image = torch.as_tensor(rgb_img, device=device) 
    return image.permute(2, 0, 1).contiguous()    


def points_box_prompt(sam, image, points, point_labels, box_array, box_label, device, inference_size, multimask_output_flag, is_bbox=False, is_points=True):
    '''
    base是保证给pos+neg points prompt
    '''
    res_label_map = np.zeros_like(image[:,:,0]).astype(np.uint8)
    resize_transform = ResizeLongestSide(inference_size)   
    points = torch.from_numpy(points).to(device=device)
    points = points.unsqueeze(0) 
    point_labels = torch.from_numpy(point_labels).to(device=device)
    point_labels = point_labels.unsqueeze(0)   
    batched_input = {'image': prepare_image(image, resize_transform, device), 'original_size': image.shape[:2]}
    if is_points: 
        print('pos-neg points prompt~')
        batched_input['point_coords'] = resize_transform.apply_coords_torch(points, image.shape[:2])
        batched_input['point_labels'] = point_labels
    if is_bbox:   
        input_boxes = torch.from_numpy(box_array).to(device=device)  # (nums, 4)
        batched_input['boxes'] = resize_transform.apply_boxes_torch(input_boxes, image.shape[:2])
        print('bbox prompt~')
    try:
        masks = sam([batched_input], multimask_output=multimask_output_flag)[0]  
    except:
        return res_label_map
    mask_labels = masks['masks'].cpu().numpy()
    if not multimask_output_flag:
        mask_res = np.squeeze(mask_labels, 1) 
    else:
        best_mask_inds = [np.argmax(mask_iou) for mask_iou in masks['iou_predictions'].cpu().numpy()]
        mask_res = []
        for i, best_id in enumerate(best_mask_inds):
            mask_ = mask_labels[i][best_id]
            mask_res.append(mask_)
    labs = box_array.shape[0]
    for i in range(labs):
        res_label_map[mask_res[i]==True] = box_label[i]
    
    return res_label_map 



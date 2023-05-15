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


def points_box_prompt(sam, image, points, point_labels, box_array, box_label, device, inference_size, is_bbox=False, is_points=True):
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
        masks = sam([batched_input], multimask_output=False)[0]  # (batch_size==1so取[0]) x (num_predicted_masks_per_input) x H x W
    except:
        return res_label_map
    mask_labels = masks['masks'].cpu().numpy()  # 1x(num_predicted_masks_per_input) x H x W 第2维是冗余的..
    mask_labels = np.squeeze(mask_labels, 1)    # (num_predicted_masks_per_input) x H x W
    labs = mask_labels.shape[0]
    for i in range(labs):
        res_label_map[mask_labels[i]==True] = box_label[i]
    
    return res_label_map



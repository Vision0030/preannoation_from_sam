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
def box_only_prompt(sam, image, box_list, device, inference_size):
    res_label_map = np.zeros_like(image[:,:,0]).astype(np.uint8)
    resize_transform = ResizeLongestSide(inference_size)  # default=sam.image_encoder.img_size=1024
    input_boxes = torch.from_numpy(np.array(box_list)).to(device=device)
    batched_input = [{'image': prepare_image(image, resize_transform, device),'boxes': resize_transform.apply_boxes_torch(input_boxes, image.shape[:2]), 'original_size': image.shape[:2]}]
    masks = sam(batched_input, multimask_output=True)[0]
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
    predictor = SamPredictor(sam)
    predictor.set_image(image)   # 获取image embedding
    try:
        masks, _, _ = predictor.predict(
                point_coords=points,
                point_labels=point_labels,
                multimask_output=False,) 
    except:
        return res_label_map
    for mask in masks:
        res_label_map[mask==True] = 255
    print('points prompt~')

    return res_label_map


def points_box_prompt(sam, image, points, point_labels, box_list, device, inference_size):
    res_label_map = np.zeros_like(image[:,:,0]).astype(np.uint8)
    predictor = SamPredictor(sam)
    predictor.set_image(image)   # resizeed image embedding for point prompt
    # resize_transform = ResizeLongestSide(inference_size)
    # input_boxes = torch.from_numpy(np.array(box_list)).to(device=device)
    # input_boxes = resize_transform.apply_boxes_torch(input_boxes, image.shape[:2])
    try:
        masks, _, _ = predictor.predict(
            point_coords=points,   # from segformer 
            point_labels=point_labels,   
            box=np.array(box_list),      # ann or other 
            multimask_output=False,)
    except:
        return res_label_map 
    for mask in masks:
        res_label_map[mask==True] = 255
    print('box+points prompt~')

    return res_label_map

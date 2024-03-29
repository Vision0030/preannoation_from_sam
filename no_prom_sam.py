# coding=utf-8
# sam_model.py 导入sam的pretrain: https://github.com/facebookresearch/segment-anything#model-checkpoints
# 依次写: 不给prompt, box prompt, point prompt等function
import cv2
import sys
import numpy as np 
import torch 
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
​
sam_checkpoint = "./sam_vit_h_4b8939.pth"
model_type = "default"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(sam)
 
# no prompt
def SAM_mask(rgb_img):
    masks = mask_generator.generate(rgb_img)   # list, 每个元素是一个分割结果
    return masks
 
# box prompt
def prepare_image(rgb_img, transform):
    rgb_img = transform.apply_image(rgb_img)
    image = torch.as_tensor(rgb_img, device='cuda') 
    return image.permute(2, 0, 1).contiguous()   # 3hw 深拷贝
def box_prompt_mask(rgb_img, input_boxes):
    input_boxes = torch.from_numpy(np.array(input_boxes)).to(device='cuda')
    batched_input = [{'image': prepare_image(rgb_img, resize_transform),'boxes': resize_transform.apply_boxes_torch(input_boxes, rgb_img.shape[:2]), 'original_size': rgb_img.shape[:2]}]
    batched_output = sam(batched_input, multimask_output=False)
 
    return batched_output
 
# random box内points and label
def point_prompt_mask(image, points, point_labels):
    points = np.array(points)
    point_labels = np.array(point_labels)
    predictor.set_image(image)
    masks, _, _ = predictor.predict(
    point_coords=points,
    point_labels=point_labels,
    multimask_output=False,)
    return masks
 
# train model and give points and point_labels
def mask_prompt(image, points, point_labels):
    points = np.array(points)
    point_labels = np.array(point_labels)
    predictor.set_image(image)
    masks, _, _ = predictor.predict(
    point_coords=points,   # train自己的任务数据模型, 给一个初步predict mask 
    point_labels=point_labels,   # 对应mask的label index
    multimask_output=False,)
    return masks
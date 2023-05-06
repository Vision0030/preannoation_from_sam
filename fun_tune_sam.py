# coding=utf-8
# ref: https://colab.research.google.com/drive/1F6uRommb3GswcRlPZWpkAQRMVNdVH7Ww?usp=sharing#scrollTo=lz7B4NDoJRxJ

import os
import os.path as osp 
import numpy as np 
import cv2 
import torch 
from statistics import mean
from tqdm import tqdm
from torch.nn.functional import threshold, normalize
from collections import defaultdict
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import SamPredictor, sam_model_registry
from read_xml import getimages


def getdirs(root_dir):
    sub_dirs = os.listdir(root_dir)
    if len(sub_dirs[0].split('.')) < 2:
        for sub_dir in sub_dirs:
            getdirs(osp.join(root_dir, sub_dir))
    else:
        for sub_dir in sub_dirs:
            if '.xml' in sub_dir:
                xmls.append(osp.join(root_dir, sub_dir))

# 1. base load 
model_type = 'default'
checkpoint = './sam_vit_h_4b8939.pth'
device = 'cuda:0'
sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
sam_model.to(device)
sam_model.train()


# 2. box prompt
bbox_coords = {}
root_dir = '/mnt/data/jiachen/pre_ann_data/test'
xmls = []
getdirs(root_dir)
for xml_path in xmls:
    _, xml_ann = getimages(xml_path)
    box_list = []
    for ann in xml_ann:
        xml_box = ann[:4]
        box_list.append(xml_box)
    bbox_coords[xml_path[:-3]] = np.array(box_list)
# 没有box的prompt的话, 可以用mask级别的gt找contours. 
# bbox_coords = {}
# for f in sorted(Path('ground-truth-maps/ground-truth-maps/').iterdir())[:100]:
#   k = f.stem[:-3]
#   if k not in stamps_to_exclude:
#     im = cv2.imread(f.as_posix())
#     gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#     contours, hierarchy = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
#     if len(contours) > 1:
#       x,y,w,h = cv2.boundingRect(contours[0])
#       height, width, _ = im.shape
#       bbox_coords[k] = np.array([x, y, x + w, y + h])


# 3. fine tune, so we need ground True mask
ground_truth_masks = {}
for k in bbox_coords.keys():
  gt_grayscale = cv2.imread('{}png'.format(k), cv2.IMREAD_GRAYSCALE)
  ground_truth_masks[k] = (gt_grayscale == 0)

# 4. transform data 
transformed_data = defaultdict(dict)
for k in bbox_coords.keys():
    image = cv2.imread('{}bmp'.format(k))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    input_image = transform.apply_image(image)
    input_image_torch = torch.as_tensor(input_image, device=device)
    transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

    input_image = sam_model.preprocess(transformed_image)
    original_image_size = image.shape[:2]
    input_size = tuple(transformed_image.shape[-2:])

    transformed_data[k]['image'] = input_image
    transformed_data[k]['input_size'] = input_size
    transformed_data[k]['original_image_size'] = original_image_size

# 5. Set up the optimizer, hyperparameter tuning will improve performance here
lr = 1e-4
wd = 0
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=lr, weight_decay=wd)
loss_fn = torch.nn.MSELoss()
# loss_fn = torch.nn.BCELoss()
keys = list(bbox_coords.keys())

# 6. train 
num_epochs = 100
losses = []
for epoch in range(num_epochs):
  epoch_losses = []
  # Just train on the first 20 examples
  # for k in keys[:20]:
  for k in keys:
    input_image = transformed_data[k]['image'].to(device)
    input_size = transformed_data[k]['input_size']
    original_image_size = transformed_data[k]['original_image_size']
    # No grad here as we don't want to optimise the encoders
    with torch.no_grad():
      image_embedding = sam_model.image_encoder(input_image)
      prompt_box = bbox_coords[k]
      box = transform.apply_boxes(prompt_box, original_image_size)
      box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
      box_torch = box_torch[None, :]
      sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
          points=None,
          boxes=box_torch,
          masks=None,
      )
    low_res_masks, iou_predictions = sam_model.mask_decoder(
      image_embeddings=image_embedding,
      image_pe=sam_model.prompt_encoder.get_dense_pe(),
      sparse_prompt_embeddings=sparse_embeddings,
      dense_prompt_embeddings=dense_embeddings,
      multimask_output=False,
    )

    upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
    binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

    gt_mask_resized = torch.from_numpy(np.resize(ground_truth_masks[k], (1, 1, ground_truth_masks[k].shape[0], ground_truth_masks[k].shape[1]))).to(device)
    gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)
    
    loss = loss_fn(binary_mask, gt_binary_mask)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch_losses.append(loss.item())
  losses.append(epoch_losses)
  print(f'EPOCH: {epoch}')
  print(f'Mean loss: {mean(epoch_losses)}')
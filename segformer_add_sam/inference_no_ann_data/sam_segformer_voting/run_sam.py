# coding=utf-8 
'''
把sam和segformer并行run吧, 各自都落盘下来, 加速pipeline~ 
'''
import os 
import os.path as osp 
import numpy as np 
import cv2  
import argparse
from segment_anything import sam_model_registry
from sam_model import SAM_mask


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='default')
    parser.add_argument('--sam_checkpoint', type=str, default='./sam_vit_h_4b8939.pth')
    parser.add_argument('--device_sam', type=str, default='cuda:6')
    parser.add_argument('--sam_mask_savedir', type=str, default='/home/jia.chen/worshop/big_model/SAM/sam_mask_dir')
    args = parser.parse_args()

    all_img_paths = np.load('./all_img_paths.npy')

    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device_sam)

    for img_path in all_img_paths:
        mask_list = []
        basename = osp.basename(img_path)
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        print('sam {} ~~~'.format(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = SAM_mask(image, sam)
        for ind, mask_dict in enumerate(masks):
            mask_map = mask_dict['segmentation']
            mask_list.append(mask_map)
        np.save(osp.join(args.sam_mask_savedir, basename[:-3]+'npy'), np.array(mask_list))


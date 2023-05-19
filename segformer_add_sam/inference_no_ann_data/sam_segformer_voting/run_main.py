# coding=utf-8
'''
sam 和 segformer的结合, sam出好的边缘, 找Segformer的结果做voting~
这个规则设计需要挺细节的.. 还需再完善..

'''

import os
import os.path as osp 
import numpy as np 
import argparse
import cv2 
import copy 
import json 
from subprocess import run
from segment_anything import sam_model_registry
from sam_model import SAM_mask
from bk_cvat_upload_ann.samres2citycapse import generate_gtFine, check_instance_id


def vis_label_map(label_map, col_map):
    color_map1 = np.zeros_like(label_map)
    color_map2 = np.zeros_like(label_map)
    color_map3 = np.zeros_like(label_map)
    for ind, col in enumerate(col_map):  
        color_map1[label_map==ind+1] = col[0]
        color_map2[label_map==ind+1] = col[1]
        color_map3[label_map==ind+1] = col[2]
    resimg = cv2.merge([color_map1, color_map2, color_map3])

    return resimg



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='default')
    parser.add_argument('--sam_checkpoint', type=str, default='./sam_vit_h_4b8939.pth')
    parser.add_argument('--device_sam', type=str, default='cuda:0')
    parser.add_argument('--sam_segformer_voing_thre', type=float, default=0.7)
    parser.add_argument('--mask_area_thres', type=int, default=50) 
    parser.add_argument('--segformer_maskdir', type=str, default='/home/jia.chen/worshop/big_model/SAM/former_dir')
    parser.add_argument('--data_dir', type=str, default='/mnt/data/jiachen/noann_data/data/2023-5-12')
    parser.add_argument('--out_dir', type=str, default='/mnt/data/jiachen/noann_data/preann_res/gtFine/default')
    parser.add_argument('--vis_dir', type=str, default='/mnt/data/jiachen/noann_data/preann_res/gtFine/vis')
    parser.add_argument('--haitian_needlabels', type=dict, default={1: 'Rug', 2: 'Cable', 3: 'Tissue', 4: 'Poop', 5: 'Liquid'})   
    args = parser.parse_args()

    col_map = [[0, 255, 0], [0, 255, 255], [200, 100, 200], [0, 0, 255], [255,0,0]]
    # 1. init sam
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device_sam)
    img_paths = [osp.join(args.data_dir, a) for a in os.listdir(args.data_dir)]

    # 2. run segformer
    # 调用cmd run segformer的inference, 我装的这俩环境有点冲突, so分开run..
    run('/home/jia.chen/miniconda3/envs/open-mmlab/bin/python run_segformer.py', shell=True)

    for img_path in img_paths:
        basename = osp.basename(img_path)
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        print('sam processing {} ~~~'.format(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # run sam 
        masks = SAM_mask(image, sam)
        segformer_mask = cv2.imread(osp.join(args.segformer_maskdir, basename[:-3]+'png'), cv2.IMREAD_UNCHANGED)
        sam_add_segformer = np.zeros_like(segformer_mask).astype(np.uint8)
        for ind, mask_dict in enumerate(masks):
            mask_map = mask_dict['segmentation']
            tmp = copy.deepcopy(segformer_mask)
            # 针对sam分割出的每一个bin, 对应去取segformer里的pred_lab_value
            tmp[mask_map==False] = 0  
            labs = np.unique(tmp)
            if len(labs) == 1:  # [0]
                continue 
            # [0, lab1, lab2, ...]
            all_areas = len(np.where(mask_map==True)[0])  # sam这块bin的总像素个数 
            lab_rate = [0]
            for lab in labs[1:]:  # 考虑sam的边缘优先级好于segformer, so, tmp里出现多个lab_i的话, 我们按像素点多的那个类去赋值label_value
                cur_lab_count = len(np.where(tmp==lab)[0])
                cur_lab_count /= float(all_areas)
                lab_rate.append(cur_lab_count)
            max_rate = max(lab_rate)
            if max_rate > args.sam_segformer_voing_thre:  # 需要卡个阈值, 避免segformer过检导致image内到处都是object
                sam_add_segformer[mask_map==True] = labs[lab_rate.index(max_rate)]
        # vis ~ 
        color_res = vis_label_map(sam_add_segformer, col_map)
        labeluint8, instanceuint16, color_map = generate_gtFine(mask_area_thres=args.mask_area_thres, sam_label_res=sam_add_segformer, sam_color_map=color_res)
        print(osp.join(args.out_dir, basename[:-4]+'_gtFine_labelIds.png'))
        cv2.imwrite(osp.join(args.out_dir, basename[:-4]+'_gtFine_labelIds.png'), labeluint8)
        cv2.imwrite(osp.join(args.out_dir, basename[:-4]+'_gtFine_instanceIds.png'), instanceuint16)
        cv2.imwrite(osp.join(args.out_dir, basename[:-4]+'_gtFine_color.png'), color_map)
        vis_img = np.concatenate([color_map, image[:,:,::-1]], axis=0)
        cv2.imwrite(osp.join(args.vis_dir, basename[:-4]+'_vis.png'), vis_img)
    check_instance_id(args.out_dir)

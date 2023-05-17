# coding=utf-8
'''
segformer出mask然后找bbox, pos_neg_points 给sam~ 
但效果看起来没有太好.. 
后续考虑方案: 不给任何prompt直接sam出mask, 各个instance抽出feature, 和segformer的各个类别的feature做对比(voting)
实现instance分类. 主要想用起来sam的优秀边缘!

'''
import os
import os.path as osp 
import argparse
import time
import numpy as np 
import cv2 
from read_xml import getimages
from segment_anything import sam_model_registry
from noann_sam_model import points_box_prompt
from bk_cvat_upload_ann.samres2citycapse import generate_gtFine, check_instance_id
from segformer_points_prompt import get_segformer_mask, segformer_points_bbox 


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

def run_prompt_sam(img_name, sam, image):
    res_label_map = np.zeros_like(image[:,:,0]).astype(np.uint8)
    segformer_mask = get_segformer_mask(img_name, args.segformer_checkpoint, args.segformer_config, args.device_segformer) 
    print('segformer_mask decoder: ', np.sum(segformer_mask))
    points, pointslabels, box_array, box_label = segformer_points_bbox(segformer_mask, args.segformer2haitian, point_num=args.point_num, mask_area_thres=args.mask_area_thres,ner_kernel_size=args.ner_kernel_size)
    if args.prompt_format == 'segformer_points' and points.shape[0]:
        res_label_map = points_box_prompt(sam, image, points, pointslabels, box_array, box_label, args.device_sam, args.inference_size)
    if args.prompt_format == 'segformer_box' and box_array.shape[0]:
        res_label_map = points_box_prompt(sam, image, points, pointslabels, box_array, box_label, args.device_sam, args.inference_size, is_bbox=True, is_points=False)
    if args.prompt_format in ['segformer_points_bbox', 'persam_segformer_points_bbox']:
        if points.shape[0]:
            res_label_map = points_box_prompt(sam, image, points, pointslabels, box_array, box_label, args.device_sam, args.inference_size, is_bbox=True)
            if args.prompt_format == 'persam_segformer_points_bbox':
                # 手动做refinment, 和persam略有不同~  [不一定比segformer_points_bbox好...]
                res_label_map[res_label_map!=0] = 1
                print('first sam mask decoder res: ', np.sum(res_label_map))
                points, pointslabels, box_array, box_label  = segformer_points_bbox(res_label_map, args.segformer2haitian, point_num=args.point_num, mask_area_thres=args.mask_area_thres,ner_kernel_size=args.ner_kernel_size)
                if points.shape[0]:
                    res_label_map = points_box_prompt(sam, image, points, pointslabels, box_array, box_label, args.device_sam, args.inference_size, is_bbox=True)
    return res_label_map


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='default')
    parser.add_argument('--sam_checkpoint', type=str, default='./sam_vit_h_4b8939.pth')
    parser.add_argument('--segformer_config', type=str, default='./segformer_config.py')
    parser.add_argument('--segformer_checkpoint', type=str, default='/home/jia.chen/worshop/big_model/SegFormer/work_dirs/iter_160000.pth')
    parser.add_argument('--device_sam', type=str, default='cuda:0')
    parser.add_argument('--device_segformer', type=str, default='cuda:3')  # segformer+sam一起可能显存不够,so分开run
    parser.add_argument('--inference_size', type=int, default=600)
    parser.add_argument('--prompt_format', type=str, default='persam_segformer_points_bbox')  # 'segformer_points', 'segformer_box', 'segformer_points_bbox', 'persam_segformer_points_bbox'
    parser.add_argument('--ner_kernel_size', type=int, default=5)  # 邻域范围内, neg,pos点均要满足一致
    parser.add_argument('--mask_area_thres', type=int, default=30)  # 小于mask_area_thres面积的滤掉
    parser.add_argument('--point_num', type=int, default=2)  # 每个instance出promp point: pos, neg各point_num个
    parser.add_argument('--data_dir', type=str, default='/mnt/data/jiachen/pre_ann_data/2023-5-12')
    parser.add_argument('--out_dir', type=str, default='/mnt/data/jiachen/pre_ann_data/2023-5-12_bbox_points_persam')
    parser.add_argument('--vis_dir', type=str, default='/mnt/data/jiachen/pre_ann_data/2023-5-12_bbox_points_persam/vis')
    parser.add_argument('--segformer2haitian', type=dict, default= {'1':1,'2':2,'3':3,'4':4,'5':5})   
    args = parser.parse_args()

    jiachen_cls_index = {'Rug': 1, 'Cable': 2, 'Tissue': 3, 'Poop': 4, 'Liquid': 5}
    col_map = [[0, 255, 0], [0, 255, 255], [200, 100, 200], [0, 0, 255], [255,0,0]] 
    # 1. init sam
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device_sam)
    img_paths = [osp.join(args.data_dir, a) for a in os.listdir(args.data_dir)]
    for img_path in img_paths:
        basename = osp.basename(img_path)
        # 2. get sam mask
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        print('processing {} ~~~'.format(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 3. prompt run sam ~~~
        sam_label = run_prompt_sam(img_path, sam, image)
        color_res = vis_label_map(sam_label, col_map)
        # 4. 生成instance.png, 过滤一些小area mask, 对应把label.png, colormap都refine下~
        labeluint8, instanceuint16, color_map = generate_gtFine(mask_area_thres=args.mask_area_thres, sam_label_res=sam_label, sam_color_map=color_res)
        # 5. cv2.imwrite as citycapse format
        cv2.imwrite(osp.join(args.out_dir, basename[:-4]+'_gtFine_labelIds.png'), labeluint8)
        cv2.imwrite(osp.join(args.out_dir, basename[:-4]+'_gtFine_instanceIds.png'), instanceuint16)
        cv2.imwrite(osp.join(args.out_dir, basename[:-4]+'_gtFine_color.png'), color_map)
        vis_img = np.concatenate([color_map, image[:,:,::-1]], axis=0)
        cv2.imwrite(osp.join(args.vis_dir, basename[:-4]+'_vis.png'), vis_img)
    # 10. 检查下instance.png的index连续性
    check_instance_id(args.out_dir)

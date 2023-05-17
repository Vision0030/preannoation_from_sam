# coding=utf-8
'''
segformer出points prompt
ann-box; ann-box+segformer-points; 2种形式prompt
结论: 
    ann-box baseline
    ann-box+segformer-points:
            1. 更好的pos neg points: segformer_pos_neg_points_in_box
            2. 是不是points设计错了? 整体弄下来, 只用box不用points更好...??? 

'''
import os
import os.path as osp 
import argparse
import time
import numpy as np 
import cv2 
import json 
from read_xml import getimages
from segment_anything import sam_model_registry
from sam_model import box_only_prompt, points_only_prompt, points_box_prompt 
from bk_cvat_upload_ann.samres2citycapse import generate_gtFine, check_instance_id
from segformer_points_prompt import get_segformer_mask, segformer_pos_neg_points_in_box


def getdirs(root_dir):
    sub_dirs = os.listdir(root_dir)
    if len(sub_dirs[0].split('.')) < 2:
        for sub_dir in sub_dirs:
            getdirs(osp.join(root_dir, sub_dir))
    else:
        for sub_dir in sub_dirs:
            if '.xml' in sub_dir:
                xmls.append(osp.join(root_dir, sub_dir))


def rewrite_img_2_citycapse(xml, img_save_name):
    image = cv2.imread(xml[:-3]+'bmp', cv2.IMREAD_UNCHANGED)
    cv2.imwrite(img_save_name, image)
    
    return img_save_name

def pre_data_process(xml_path, image_save_name):
    _, xml_ann = getimages(xml_path)
    box_list = []
    labels = []
    for ann in xml_ann:
        cls_name = ann[4]
        if cls_name in clses:
            labels.append(cls_name)
            xml_box = ann[:4]
            box_list.append(xml_box)
            # image rename成citycapse格式且保存为jpg
            if not osp.exists(image_save_name):
                rewrite_img_2_citycapse(xml_path, image_save_name)
                
    return labels, box_list 


def give_cls_index(res_label_map, jiachen_cls_index, labels, box_list):
    sam_label = np.zeros_like(res_label_map).astype(np.uint8)
    for ind, box in enumerate(box_list):
        tmp = res_label_map[box[1]:box[3],box[0]:box[2]]
        if np.sum(tmp) != 0:
            tmp[tmp!=0] = jiachen_cls_index[labels[ind]]
            sam_label[box[1]:box[3],box[0]:box[2]] = tmp 

    return sam_label


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

def run_prompt_sam(sam, image, box_list, img_save_name):
    res_label_map = np.zeros_like(image[:,:,0]).astype(np.uint8)
    box_array = np.array(box_list)
    if args.prompt_format == 'box-only':
        res_label_map = box_only_prompt(sam, image, box_array, args.device_sam, args.inference_size, args.multimask_output)
    elif args.prompt_format == 'box_and_segformerpoints':
        segformer_mask = get_segformer_mask(img_save_name, args.segformer_checkpoint, args.segformer_config, args.device_segformer)   
        # 基于ann-box, 让segformer出更准的points(pos, neg都包含, 且邻域检查~)
        points, point_labels = segformer_pos_neg_points_in_box(segformer_mask, args.segformer2haitian, box_list, point_num=args.point_num, mask_area_thres=args.mask_area_thres,pos_ner_kernel_size=args.pos_ner_kernel_size, neg_ner_kernel_size=args.neg_ner_kernel_size, find_times=args.find_times)
        if points.shape[0]:
            res_label_map = points_box_prompt(sam, image, points, point_labels, box_array, args.device_sam, args.inference_size, args.multimask_output)
        else:
            res_label_map = box_only_prompt(sam, image, box_array, args.device_sam, args.inference_size, args.multimask_output)

    return res_label_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='default')
    parser.add_argument('--sam_checkpoint', type=str, default='./sam_vit_h_4b8939.pth')
    parser.add_argument('--segformer_config', type=str, default='./segformer_config.py')
    parser.add_argument('--segformer_checkpoint', type=str, default='/home/jia.chen/worshop/big_model/SegFormer/work_dirs/iter_160000.pth')
    parser.add_argument('--device_sam', type=str, default='cuda:0')
    parser.add_argument('--device_segformer', type=str, default='cuda:2')  # segformer+sam一起可能显存不够,so分开run
    parser.add_argument('--inference_size', type=int, default=600)
    parser.add_argument('--multimask_output', type=bool, default=True)
    parser.add_argument('--prompt_format', type=str, default='box_and_segformerpoints')  # 'box-only', 'box_and_segformerpoints' 
    parser.add_argument('--mask_area_thres', type=int, default=50)  # 小于mask_area_thres面积的滤掉
    parser.add_argument('--pos_ner_kernel_size', type=int, default=7)  # 5x5winds邻域内都是正样本or负样本
    parser.add_argument('--neg_ner_kernel_size', type=int, default=11)
    parser.add_argument('--point_num', type=int, default=1)  # 每个instance出point_num个promp point(pos, neg均point_num个)
    parser.add_argument('--find_times', type=int, default=50)  # 50次寻找pos,neg点的限制次数
    parser.add_argument('--data_dir', type=str, default='/mnt/data/jiachen/pre_ann_data/test')
    parser.add_argument('--out_dir', type=str, default='/mnt/data/jiachen/sam_preann_haitian/gtFine/default')
    parser.add_argument('--img_save_path', type=str, default='/mnt/data/jiachen/sam_preann_haitian/imgsFine/leftImg8bit/default')
    parser.add_argument('--vis_dir', type=str, default='/mnt/data/jiachen/sam_preann_haitian/gtFine/vis')
    parser.add_argument('--segformer2haitian', type=dict, default= {'4':2, '2':1})  # segformer的4是便便对应海天的2, segformer的2是cable线, 对应海天的1
    args = parser.parse_args()

    # bianbian2  xian1 
    jiachen_cls_index = {'fabric': 1, 'wire': 1, 'shoes': 3, 'door': 4, 'bed': 5, 'trash can': 6, 'feces': 2, 'range hood': 8, 'toilet': 9, 'showerhead': 10, 'electric water heater': 11, 'tvmonitor': 12, 'end table': 13, 'chair': 14, 'bedside table': 15, 'wardrobe': 16, 'cupboard': 17, 'tv cabinet': 18, 'sofa': 19, 'dining table': 20, 'base': 21, 'power strip': 1, 'bathroom scale': 23, 'MISSING': 24}
    col_map = [[200,200,0], [70,70,70]]  # 只处理感兴趣的xian, bianbian 
    clses = {'wire': 1, 'feces': 2, 'power strip': 3}

    # init sam
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device_sam)
    # load or pre_process data 
    xml_file_name = './xmls.txt'
    if osp.exists(xml_file_name):
        xmls = open(xml_file_name, 'r').readlines()
        xmls = [a[:-1] for a in xmls]
    else:
        xmls = []
        getdirs(args.data_dir)
        xml_files = open(xml_file_name, 'w')
        for xml in xmls:
            xml_files.write(xml+'\n')
    #  prompt sam ~ 
    for xml_path in xmls:
        path_dir = osp.dirname(xml_path)
        image_save_name = osp.join(args.img_save_path, '{}_{}_{}'.format(path_dir.split('/')[-2], path_dir.split('/')[-1], osp.basename(xml_path)[:-4]+'.jpg'))
        basename = osp.basename(image_save_name)
        labels, box_list = pre_data_process(xml_path, image_save_name)
        if len(labels) == 0:
            continue 
        image = cv2.imread(image_save_name, cv2.IMREAD_UNCHANGED)  # 直接读取上一步rename好的jpg~
        print('processing {} ~~~'.format(image_save_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # prompt sam ~~~
        res_label_map = run_prompt_sam(sam, image, box_list, image_save_name)
        sam_label = give_cls_index(res_label_map, jiachen_cls_index, labels, box_list)
        print(np.unique(res_label_map))
        color_res = vis_label_map(sam_label, col_map)
        # 生成instance.png, 过滤一些小area mask, 对应把label.png, colormap都refine下~
        labeluint8, instanceuint16, color_map = generate_gtFine(mask_area_thres=args.mask_area_thres, sam_label_res=sam_label, sam_color_map=color_res)
        # imwrite as citycapse format
        cv2.imwrite(osp.join(args.out_dir, basename[:-4]+'_gtFine_labelIds.png'), labeluint8)
        cv2.imwrite(osp.join(args.out_dir, basename[:-4]+'_gtFine_instanceIds.png'), instanceuint16)
        cv2.imwrite(osp.join(args.out_dir, basename[:-4]+'_gtFine_color.png'), color_map)
        # visualize check 
        for xml_box in box_list:
            cv2.rectangle(color_map, (xml_box[0],xml_box[1]), (xml_box[2],xml_box[3]), (0,255,0), 2)
            cv2.rectangle(image, (xml_box[0],xml_box[1]), (xml_box[2],xml_box[3]), (0,255,0), 2)
        vis_img = np.concatenate([color_map, image[:,:,::-1]], axis=0)  # image bgr2rgb
        cv2.imwrite(osp.join(args.vis_dir, basename[:-4]+'_vis.png'), vis_img)
    #  检查下instance.png的index连续性
    check_instance_id(args.out_dir)
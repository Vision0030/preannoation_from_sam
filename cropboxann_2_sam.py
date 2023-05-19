# coding=utf-8
'''
根据box annotation信息, 把roi crop出来, 上下左右给点冗余
然后再加上ann-box prompt 给到sam~ (因为sam给你seg出everything, 没有box prompt的话, 我们取不出想要的具体object啊~)
又快又准!, sam要推理的图也小了, 背景的干扰也少了~ 

'''

import os
import os.path as osp 
import numpy as np 
import os.path as osp 
import argparse
import cv2 
import json 
from read_xml import getimages
from segment_anything import sam_model_registry
from sam_model import box_only_prompt
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

def getdirs(root_dir):
    sub_dirs = os.listdir(root_dir)
    if len(sub_dirs[0].split('.')) < 2:
        for sub_dir in sub_dirs:
            getdirs(osp.join(root_dir, sub_dir))
    else:
        for sub_dir in sub_dirs:
            if '.xml' in sub_dir:
                xmls.append(osp.join(root_dir, sub_dir))

def pre_data_process(xml_path, image_save_name):
    _, xml_ann, h, w = getimages(xml_path)    
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
                try:
                    image = cv2.imread(xml_path[:-3]+'bmp', cv2.IMREAD_UNCHANGED)
                    cv2.imwrite(image_save_name, image) 
                except:
                    print(image_save_name)

    return labels, box_list

def modifiy_box(box, h, w):
    box_w, box_h = box[2]-box[0], box[3]-box[1]
    crop_box = [0,0, min(w, box[2]+args.buff_size), min(h,box[3]+args.buff_size)] 
    new_box = [0]*4
    for ind in range(2):
        if box[ind]-args.buff_size>0:
            crop_box[ind] = box[ind]-args.buff_size
            new_box[ind] = args.buff_size
    new_box[2] = new_box[0]+box_w
    new_box[3] = new_box[0]+box_h

    return crop_box, np.array(new_box), max(crop_box[3]-crop_box[1], crop_box[2]-crop_box[0])
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='default')
    parser.add_argument('--sam_checkpoint', type=str, default='./sam_vit_h_4b8939.pth')
    parser.add_argument('--device_sam', type=str, default='cuda:2')
    parser.add_argument('--buff_size', type=int, default=20)  # 根据boxann去crop roi然后送给sam, 上下左右各设置20pixels的冗余buff
    parser.add_argument('--data_dir', type=str, default='/mnt/data/jiachen/boxann_data/data/0518')
    parser.add_argument('--out_dir', type=str, default='/mnt/data/jiachen/boxann_data/preann_res/0518/gtFine/default')
    parser.add_argument('--img_save_path', type=str, default='/mnt/data/jiachen/boxann_data/preann_res/0518/imgsFine/leftImg8bit/default')
    parser.add_argument('--vis_dir', type=str, default='/mnt/data/jiachen/boxann_data/preann_res/0518/gtFine/vis')
    parser.add_argument('--multimask_output', type=bool, default=True)
    parser.add_argument('--mask_area_thres', type=int, default=50)
    parser.add_argument('--segformer2haitian', type=dict, default= {'4':2, '2':1})  
    args = parser.parse_args()

    col_map = [[200,200,0], [70,70,70]]   
    clses = {'wire': 1, 'feces': 2, 'power strip': 1}

    # init sam
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device_sam)
    # load or pre_process data 
    xml_file_name = './xmls_0518.txt'
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
        if len(labels) == 0 or not osp.exists(image_save_name):
            continue 
        image = cv2.imread(image_save_name, cv2.IMREAD_UNCHANGED)   
        h, w = image.shape[:2]  # y在前x在后
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res_label_map = np.zeros_like(image[:,:,0]).astype(np.uint8)
        print('croped {} ~~~'.format(image_save_name))
        for ind, box in enumerate(box_list):
            label_value = clses[labels[ind]]
            crop_box, box_array, box_size = modifiy_box(box, h, w)
            roi_img = image[crop_box[1]:crop_box[3],crop_box[0]:crop_box[2],:]
            # cv2.imwrite('roi_img.jpg', roi_img)
            roi_res = res_label_map[crop_box[1]:crop_box[3],crop_box[0]:crop_box[2]]
            # 还是要给box prompt的, 不然sam给你seg出everything, 但我们想要的目标, 取不出具体的啊~~~
            sam_roi_res = box_only_prompt(sam, roi_img, box_array, args.device_sam, box_size, args.multimask_output)
            roi_res[sam_roi_res==255] = label_value
            res_label_map[crop_box[1]:crop_box[3],crop_box[0]:crop_box[2]] = roi_res
        color_res = vis_label_map(res_label_map, col_map)

        labeluint8, instanceuint16, color_map = generate_gtFine(mask_area_thres=args.mask_area_thres, sam_label_res=res_label_map, sam_color_map=color_res)
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
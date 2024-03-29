# coding=utf-8
# box_prompt_pipeline.py
import os
import os.path as osp 
import numpy as np 
import cv2 
from read_xml import getimages
from sam_model import box_prompt_mask
from bk_cvat_upload_ann.samres2citycapse import generate_gtFine, check_instance_id
import time 

def give_cls_index(res_label_map, jiachen_cls_index, labels, box_list):
    sam_label = np.zeros_like(res_label_map).astype(np.uint8)
    for ind, box in enumerate(box_list):
        tmp = res_label_map[box[1]:box[3],box[0]:box[2]]
        if np.sum(tmp) != 0:
            # sam_label[box[1]:box[3],box[0]:box[2]]=jiachen_cls_index[labels[ind]]
            tmp[tmp!=0] = jiachen_cls_index[labels[ind]]
            sam_label[box[1]:box[3],box[0]:box[2]] = tmp 
    if len(sam_label.shape) > 2:
        sam_label = sam_label[:,:,0]
        
    return sam_label


def vis_label_map(label_map):
    col_map = [[0,0,50],[0,0,100],[0,0,150],[0,0,250],[0,255,0],[0,155,0],[0,55,0],[255,255,0],[100,100,100],[0,0,10],[0,0,50],[0,0,80],[0,0,190],[0,125,0],[0,105,0],[0,15,0],[200,200,0],[70,70,70], [0,0,30],[0,0,110],[0,65,0],[0,35,0],[0,8,0],[20,10,0]]
    color_map1 = np.zeros_like(label_map)
    color_map2 = np.zeros_like(label_map)
    color_map3 = np.zeros_like(label_map)
    for ind, col in enumerate(col_map):   # label_index_range
        # color_map1[label_map==ind+33] = col[0]
        # color_map2[label_map==ind+33] = col[1]
        # color_map3[label_map==ind+33] = col[2]
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


def rewrite_img_2_citycapse(xml, img_save_path):
    image = cv2.imread(xml[:-3]+'bmp', cv2.IMREAD_UNCHANGED)
    basename = osp.basename(xml)
    path_dir = osp.dirname(xml)
    cv2.imwrite(osp.join(img_save_path, '{}_{}_{}'.format(path_dir.split('/')[-2], path_dir.split('/')[-1], basename[:-4]+'.jpg')), image)


if __name__ == "__main__":

    jiachen_cls_index ={'feces': 1, 'base': 2, 'chair': 3, 'shoes': 4, 'bed': 5, 'bedside table': 6, 'door': 7, 'wire': 8, 'wardrobe': 9, 'cupboard': 10, 'range hood': 11, 'toilet': 12, 'showerhead': 13, 'tvmonitor': 14, 'tv cabinet': 15, 'end table': 16, 'sofa': 17, 'trash can': 18, 'dining table': 19, 'fabric': 20, 'electric water heater': 21, 'power strip': 22, 'bathroom scale': 23, 'MISSING': 24}
    root_dir = '/mnt/data/jiachen/pre_ann_data/test'
    mask_area_thres = 20  # sam出的mask, 面积<mask_area_thres的就过滤掉~ 
    # 1. xml process 
    xmls = []
    getdirs(root_dir)
    out_dir = '/mnt/data/jiachen/sam_preann_haitian/gtFine/default'
    img_save_path = '/mnt/data/jiachen/sam_preann_haitian/imgsFine/leftImg8bit/default'
    vis_dir = '/mnt/data/jiachen/sam_preann_haitian/gtFine/vis'
    for xml_path in xmls:
        # 2. image rename成citycapse格式~ 且保存为jpg
        rewrite_img_2_citycapse(xml_path, img_save_path)
        basename = osp.basename(xml_path)
        _, xml_ann = getimages(xml_path)
        box_list = []
        labels = []
        for ann in xml_ann:
            xml_box = ann[:4]
            box_list.append(xml_box)
            labels.append(ann[4])
        # 3. get sam mask
        image = cv2.imread(xml_path[:-3]+'bmp', cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res_label_map = np.zeros_like(image[:,:,0]).astype(np.uint8)
        flag = 1
        try:
            start = time.time()
            masks =  box_prompt_mask(image, box_list)[0]   
            # print(masks['iou_predictions'])
            # print(masks['low_res_logits'])
            end = time.time()
            # print('need time: {}'.format(end-start))
        except:
            flag = 0
        if flag:
            print('sam okk~ ')
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
        sam_label = give_cls_index(res_label_map, jiachen_cls_index, labels, box_list)
        color_res = vis_label_map(sam_label)
        # 4. 生成instance.png, 过滤一些小area mask, 对应把label.png, colormap都refine下~
        labeluint8, instanceuint16, color_map = generate_gtFine(mask_area_thres=mask_area_thres, sam_label_res=sam_label, sam_color_map=color_res)
        # 5. xml dir process 
        path_dir = osp.dirname(xml_path)
        if not osp.exists(path_dir):
            os.makedirs(path_dir)
        # 6. cv2.imwrite as citycapse format
        cv2.imwrite(osp.join(out_dir, '{}_{}_{}'.format(path_dir.split('/')[-2], path_dir.split('/')[-1], basename[:-4]+'_gtFine_labelIds.png')), labeluint8)
        cv2.imwrite(osp.join(out_dir, '{}_{}_{}'.format(path_dir.split('/')[-2], path_dir.split('/')[-1], basename[:-4]+'_gtFine_instanceIds.png')), instanceuint16)
        cv2.imwrite(osp.join(out_dir, '{}_{}_{}'.format(path_dir.split('/')[-2], path_dir.split('/')[-1], basename[:-4]+'_gtFine_color.png')), color_map)
        # 7. visualize check 
        for xml_box in box_list:
            cv2.rectangle(color_map, (xml_box[0],xml_box[1]), (xml_box[2],xml_box[3]), (0,255,0), 2)
            cv2.rectangle(image, (xml_box[0],xml_box[1]), (xml_box[2],xml_box[3]), (0,255,0), 2)
        vis_img = np.concatenate([color_map, image], axis=0)
        cv2.imwrite(osp.join(vis_dir, '{}_{}_{}'.format(path_dir.split('/')[-2], path_dir.split('/')[-1], basename[:-4]+'_vis.png')), vis_img)
    # 8. 检查下instance.png的index连续性
    check_instance_id(out_dir)

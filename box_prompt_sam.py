# codding=utf-8
# box prompt sam, change to citycapse dataset
import os
import os.path as osp 
import numpy as np 
import cv2 
from read_xml import getimages
from sam_model import box_prompt_mask

# from imantics import Polygons, Mask
# ...
# polygons = Mask(sam_mask).polygons()
# points = polygons.points[0]  # save points to COCO json format, which can be read by CVAT.


def give_cls_index(res_label_map, jiachen_cls_index, labels, box_list):
    sam_label = np.zeros_like(res_label_map)
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
        color_map1[label_map==ind+33] = col[0]
        color_map2[label_map==ind+33] = col[1]
        color_map3[label_map==ind+33] = col[2]
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
    image = cv2.imread(xml[:-3]+'bmp')
    print(image.shape)
    basename = osp.basename(xml)
    path_dir = osp.dirname(xml)
    cv2.imwrite(osp.join(img_save_path, '{}_{}_{}'.format(path_dir.split('/')[-2], path_dir.split('/')[-1], basename[:-4]+'_leftImg8bit.bmp')), image)


if __name__ == "__main__":
    jiachen_cls_index = {'feces': 33, 'base': 34, 'chair': 35, 'shoes': 36, 'bed': 37, 'bedside table': 38, 'door': 39, 'wire': 40, 'wardrobe': 41, 'cupboard': 42, 'range hood': 43, 'toilet': 44, 'showerhead': 45, 'tvmonitor': 46, 'tv cabinet': 47, 'end table': 48, 'sofa': 49, 'trash can': 50, 'dining table': 51, 'fabric': 52, 'electric water heater': 53, 'power strip': 54, 'bathroom scale': 55, 'MISSING': 56}
    root_dir = '/mnt/data/jiachen/pre_ann_data/test'
    xmls = []
    getdirs(root_dir)
    out_vis_dir = '/home/jia.chen/worshop/big_model/SAM/citycapse/gtFine/default'
    img_save_path = '/home/jia.chen/worshop/big_model/SAM/citycapse/imgsFine/leftImg8bit/default'
    for xml_path in xmls:

        # 把image rename成citycapse格式~ 
        rewrite_img_2_citycapse(xml_path, img_save_path)

        basename = osp.basename(xml_path)
        _, xml_ann = getimages(xml_path)
        box_list = []
        labels = []
        for ann in xml_ann:
            xml_box = ann[:4]
            box_list.append(xml_box)
            labels.append(ann[4])
        # get sam mask
        image = cv2.imread(xml_path[:-3]+'bmp')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res_label_map = np.zeros_like(image[:,:,0])
        flag = 1
        try:
            masks =  box_prompt_mask(image, box_list)[0]   # batch ==1
            # print(masks['iou_predictions'])
            # print(masks['low_res_logits'])
        except:
            flag = 0
        if flag:
            mask_labels = masks['masks'].cpu().numpy().squeeze(axis=1)
            first_mask = mask_labels[0]
            for mask_lab in mask_labels[1:]:
                first_mask = np.logical_or(first_mask, mask_lab)  # true or false = true 
            res_label_map[first_mask==True] = 255
        # 给各个sam seg出的mask团, class label
        sam_label = give_cls_index(res_label_map, jiachen_cls_index, labels, box_list)
        path_dir = osp.dirname(xml_path)
        # if not osp.exists(path_dir):
        #     os.makedirs(path_dir)
        print(osp.join(out_vis_dir, '{}_{}_{}'.format(path_dir.split('/')[-2], path_dir.split('/')[-1], basename[:-4]+'_gtFine_labelIds.png')))
        cv2.imwrite(osp.join(out_vis_dir, '{}_{}_{}'.format(path_dir.split('/')[-2], path_dir.split('/')[-1], basename[:-4]+'_gtFine_labelIds.png')), sam_label)
        cv2.imwrite(osp.join(out_vis_dir, '{}_{}_{}'.format(path_dir.split('/')[-2], path_dir.split('/')[-1], basename[:-4]+'_gtFine_instanceIds.png')), sam_label)
        # 做可视化
        res = vis_label_map(sam_label)
        for xml_box in box_list:
            cv2.rectangle(res, (xml_box[0],xml_box[1]), (xml_box[2],xml_box[3]), (0,255,0), 2)
            cv2.rectangle(image, (xml_box[0],xml_box[1]), (xml_box[2],xml_box[3]), (0,255,0), 2)
        vis = np.concatenate([res, image], axis=0)
        cv2.imwrite(osp.join(out_vis_dir, '{}_{}_{}'.format(path_dir.split('/')[-2], path_dir.split('/')[-1], basename[:-4]+'_gtFine_color.png')), vis)


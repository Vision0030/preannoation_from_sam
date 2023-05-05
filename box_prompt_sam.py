# codding=utf-8
import os
import os.path as osp 
import numpy as np 
import cv2 
from read_xml import getimages
from sam_model import box_prompt_mask

def give_cls_index(res_label_map, jiachen_cls_index, labels, box_list):
    sam_label = np.zeros_like(res_label_map)
    for ind, box in enumerate(box_list):
        tmp = res_label_map[box[1]:box[3],box[0]:box[2]]
        if np.sum(tmp) != 0:
            sam_label[box[1]:box[3],box[0]:box[2]]=jiachen_cls_index[labels[ind]]

    return sam_label


def vis_label_map(label_map):
    col_map = [[0,0,50],[0,0,100],[0,0,150],[0,0,250],[0,255,0],[0,155,0],[0,55,0],[255,255,0],[100,100,100]]
    color_map1 = np.zeros_like(label_map)
    color_map2 = np.zeros_like(label_map)
    color_map3 = np.zeros_like(label_map)
    
    for ind, col in enumerate(col_map):   # label_index_range
        color_map1[label_map==ind+33] = col[0]
        color_map2[label_map==ind+33] = col[1]
        color_map3[label_map==ind+33] = col[2]
    resimg = cv2.merge([color_map1, color_map2, color_map3])

    return resimg




if __name__ == "__main__":
    jiachen_cls_index = {'shoes': 33, 'bathroom scale': 34, 'door':35, 'wire':36, 'bed':37, 'fabric':38, 'bedside table':39, 'feces':40, 'wardrobe':41}
    xml_dir = '/home/jia.chen/worshop/big_model/SAM/bedroom'
    out_vis_dir = '/home/jia.chen/worshop/big_model/SAM/bedroom_vis'
    xmls = [osp.join(xml_dir, a) for a in os.listdir(xml_dir) if '.xml' in a]
    for xml_path in xmls:
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
        res = vis_label_map(sam_label)
        for xml_box in box_list:
            cv2.rectangle(res, (xml_box[0],xml_box[1]), (xml_box[2],xml_box[3]), (0,255,0), 2)
            cv2.rectangle(image, (xml_box[0],xml_box[1]), (xml_box[2],xml_box[3]), (0,255,0), 2)
        vis = np.concatenate([res, image], axis=0)
        cv2.imwrite(osp.join(out_vis_dir, osp.basename(xml_path)[:-3]+'jpg'), vis)
        
# codding=utf-8
import os
import os.path as osp 
import numpy as np 
import cv2 
from read_xml import getimages
from sam_model import SAM_mask

def give_cls_index(label_map, cls_index, res_label_map):
    res_label_map[label_map==True] = cls_index

    return res_label_map

def sam_predict_pair_ann_box(masks, xml_path, box_diff_thres=0.1):
    # load box-level ann 
    _, xml_ann = getimages(xml_path)
    res_label_map = np.zeros_like(image[:,:,0])
    box_list = []
    for ann in xml_ann:
        xml_box = ann[:4]
        box_list.append(xml_box)
        xlm_h, xlm_w = xml_box[2]-xml_box[0], xml_box[3]-xml_box[1]
        xlm_cls = ann[4]
        # sam predict出的box结果, 和ann中的box, 四个点diff不超过15%, 就认为分割准确, 拿这个mask来用.
        for mask_dirt in masks:
            bbox = mask_dirt['bbox']
            bbox = bbox[:2] + [bbox[0]+bbox[2], bbox[1]+bbox[3]]
            if abs(bbox[0]-xml_box[0]) < xlm_h*box_diff_thres and abs(bbox[2]-xml_box[2]) < xlm_h*box_diff_thres and \
            abs(bbox[1]-xml_box[1]) < xlm_w*box_diff_thres and abs(bbox[3]-xml_box[3]) < xlm_w*box_diff_thres:
                label_mask = mask_dirt['segmentation']
                cls_index = jiachen_cls_index[xlm_cls]
                res_label_map = give_cls_index(label_mask, cls_index, res_label_map)

    return res_label_map, box_list
    

if __name__ == "__main__":
    jiachen_cls_index = {'shoes': 33, 'bathroom scale': 34, 'door':35, 'wire':36, 'bed':37, 'fabric':38, 'bedside table':39, 'feces':40, 'wardrobe':41}
    xml_dir = '/home/jia.chen/worshop/big_model/SAM/bedroom'
    out_vis_dir = '/home/jia.chen/worshop/big_model/SAM/bedroom_vis'
    xmls = [osp.join(xml_dir, a) for a in os.listdir(xml_dir) if '.xml' in a]
    for xml_path in xmls:
        # get sam mask
        image = cv2.imread(xml_path[:-3]+'bmp')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = SAM_mask(image)
        res_label_map, box_list = sam_predict_pair_ann_box(masks, xml_path, box_diff_thres=0.15)
        # vis sam-seg-result 
        res_label_map[res_label_map!=0] = 255
        res = cv2.merge([res_label_map,res_label_map,res_label_map])
        for xml_box in box_list:
            cv2.rectangle(res, (xml_box[0],xml_box[1]), (xml_box[2],xml_box[3]), (0,255,0), 2)
            cv2.rectangle(image, (xml_box[0],xml_box[1]), (xml_box[2],xml_box[3]), (0,255,0), 2)
        vis = np.concatenate([res, image], axis=0)
        cv2.imwrite(osp.join(out_vis_dir, osp.basename(xml_path)[:-3]+'jpg'), vis)
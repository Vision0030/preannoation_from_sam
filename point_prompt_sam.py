# codding=utf-8
# point prompt + label 信息. 甚至后面可以给自己数据train的model predict作为mask_input
import os
import os.path as osp 
import numpy as np 
import cv2 
from read_xml import getimages
from sam_model import point_prompt_mask
import random

def point_pointlabel(ann, jiachen_cls_index, first_point_rate=0.5, h_half=500, w_half=500):
    box = [int(a) for a in ann[:4]]
    offset_rate = random.random()*first_point_rate
    label_index = jiachen_cls_index[ann[4]]
    h, w = box[2]-box[0], box[3]-box[1]
    xs, ys = [box[0]+h*offset_rate], [box[1]+w*offset_rate]
    if h//h_half != 0:
        for i in range(h//h_half):
            new_x = min(xs[0]+h_half*(i+1), box[2])
    if w//w_half != 0:
        for i in range(w//w_half):
            new_y = min(ys[0]+w_half*(i+1), box[3])
    points = []
    point_labels = []
    for x in xs:
        for y in ys:
            points.append([x,y])
            point_labels.append(label_index)

    return points, point_labels





if __name__ == "__main__":

    jiachen_cls_index = {'shoes': 33, 'bathroom scale': 34, 'door':35, 'wire':36, 'bed':37, 'fabric':38, 'bedside table':39, 'feces':40, 'wardrobe':41}
    xml_dir = '/home/jia.chen/worshop/big_model/SAM/bedroom'
    out_vis_dir = '/home/jia.chen/worshop/big_model/SAM/bedroom_vis'
    xmls = [osp.join(xml_dir, a) for a in os.listdir(xml_dir) if '.xml' in a]
    for xml_path in xmls:
        _, xml_ann = getimages(xml_path)
        points = []
        point_labels = []
        box_list = []
        for ann in xml_ann:
            xml_box = ann[:4]
            box_list.append(xml_box)
            ps, labs = point_pointlabel(ann, jiachen_cls_index)
            points.extend(ps)
            point_labels.extend(labs)
        # get sam mask
        image = cv2.imread(xml_path[:-3]+'bmp')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res_label_map = np.zeros_like(image[:,:,0])
        masks =  point_prompt_mask(image, points, point_labels)
        for mask in masks:
            res_label_map[mask==True] = 255
        res = cv2.merge([res_label_map,res_label_map,res_label_map])
        for xml_box in box_list:
            cv2.rectangle(res, (xml_box[0],xml_box[1]), (xml_box[2],xml_box[3]), (0,255,0), 2)
            cv2.rectangle(image, (xml_box[0],xml_box[1]), (xml_box[2],xml_box[3]), (0,255,0), 2)
        vis = np.concatenate([res, image], axis=0)
        cv2.imwrite(osp.join(out_vis_dir, osp.basename(xml_path)[:-3]+'jpg'), vis)
    
# coding=utf-8
# samres2citycapse.py
import os
import os.path as osp
import numpy as np 
import cv2 
from PIL import Image

# datumaro中介 让citycapse转coco 
# ht下包含label_color.txt gtFine
# datum convert --input-format cityscapes --input-path ht --output-format coco 
# cvat平台上传coco格式的ann-json
 
# img = cv2.imread(path_, cv2.IMREAD_UNCHANGED)
# label_ids = np.unique(img)  # 打印不重复元素
# print(label_ids) 

def to16int(path):
    for im in os.listdir(path):
        img = cv2.imread(os.path.join(path, im))
        img16 = img.astype(np.uint16)[:,:,0]
        imgOut = Image.fromarray(img16)
        imgOut.save(os.path.join(path, im))
        img = cv2.imread(os.path.join(path, im),  cv2.CV_16UC1)
        print(type(img[0,0]))


def single_cls_instanceId(map_, label_id, mask, pre_ind, uint8_label, color_map, instance_area_thres):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(map_, connectivity=8)
    # x,y,h,w,s = stats
    instance_id = 0
    for i in range(1, num_labels):
        if stats[i][-1] >= instance_area_thres:
            instance_id += 1
            mask[labels==i] = instance_id + label_id*1000 + pre_ind 
            print(instance_id + label_id*1000 + pre_ind)
        else:
            uint8_label[labels==i] = 0  # 这些就置为背景啦~
            color_map[labels==i] = [0,0,0]
    pre_ind += instance_id
    return pre_ind, mask, uint8_label, color_map

def check_instance_id(def_dir):
    # 保证instance的id是连续的~
    ins = [a for a in os.listdir(def_dir) if 'instance' in a]
    for im in ins:
        img = cv2.imread(osp.join(def_dir, im), cv2.IMREAD_UNCHANGED)
        label_ids = np.unique(img)  # 打印不重复元素
        if len(label_ids) == 1:
            continue
        label_ids=[a%1000 for a in label_ids[1:]]
        assert len(label_ids) == label_ids[-1]
                
def generate_gtFine(path=None, path1=None, mask_area_thres=None, sam_label_res=None, sam_color_map=None):
    if path:  # labels已落盘, 来做refine
        labels = [a for a in os.listdir(path) if 'label' in a]
        for ins in labels:
            img = cv2.imread(os.path.join(path, ins), cv2.IMREAD_UNCHANGED)
            color_map = cv2.imread(os.path.join(path, ins[:-12]+'color.png'), cv2.IMREAD_UNCHANGED) 
            uint8_label = img.astype(np.uint8)
            pre_ind = 0
            instance_map = np.zeros_like(img).astype(np.uint16)
            label_ids = np.unique(img)   # img中有哪些value
            for i in label_ids:
                if i == 0:
                    continue
                map_ = np.zeros_like(img).astype(np.uint8)
                map_[img==i] = 1
                # area太小的连通域滤掉
                pre_ind, instance_map, uint8_label, color_map = single_cls_instanceId(map_, i, instance_map, pre_ind, uint8_label, color_map, mask_area_thres)
            print('=========')
            cv2.imwrite(os.path.join(path1, ins), uint8_label)
            cv2.imwrite(os.path.join(path1, ins[:-12]+'instanceIds.png'), instance_map)
            cv2.imwrite(os.path.join(path1, ins[:-12]+'color.png'), color_map)

            return 1,2,3
    else:  # 直接label image形式给到 初步生成instance color等结果
        uint8_label = sam_label_res.astype(np.uint8)  # label.png
        pre_ind = 0
        instance_map = np.zeros_like(sam_label_res).astype(np.uint16)  # instance.png
        label_ids = np.unique(sam_label_res)    
        for i in label_ids:
            if i == 0:
                continue
            map_ = np.zeros_like(sam_label_res).astype(np.uint8)
            map_[sam_label_res==i] = 1
            # area太小的连通域滤掉
            pre_ind, instance_map, uint8_label, sam_color_map = single_cls_instanceId(map_, i, instance_map, pre_ind, uint8_label, sam_color_map, mask_area_thres)
        print('**********')

        return uint8_label, instance_map, sam_color_map


if __name__ == "__main__":
    path = '/home/jia.chen/worshop/big_model/SAM/bk_cvat_upload_ann/ht/gtFine/default'
    path1 = path # '/home/jia.chen/worshop/big_model/SAM/bk_cvat_upload_ann/ht/gtFine/d'
    mask_area_thres = 20
    a, b, c = generate_gtFine(path=path, path1=path1, mask_area_thres=mask_area_thres, sam_label_res=None, sam_color_map=None)
    check_instance_id(path1)
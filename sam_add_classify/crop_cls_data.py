# coding=utf-8
import os 
import os.path as osp 
import cv2 
import numpy as np  

# 标注好的seg数据, 去找label.png的连通域, 把面积过阈值的bins保存下来作为cls分类数据
train_label = '/mnt/data/jiachen/small_fov_seg/train/vis/seg_label'
train_img = '/mnt/data/jiachen/small_fov_seg/train/vis/img'
img_paths = [osp.join(train_label, a) for a in os.listdir(train_label)]
instance_cls_savedir = '/mnt/data/jiachen/small_fov_seg/train/sam_cls_data'

# 统计各个类别的平均面积 320x1024
cls_areas = [[] for _ in range(6)]   

for img_path in img_paths:
    basename = osp.basename(img_path)
    label_map = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    label_ids = np.unique(label_map)
    if len(label_ids) > 1:
        image = cv2.imread(osp.join(train_img, basename[:-3]+'jpg'))
        for lab_ind in label_ids[1:]:
            save_dir = osp.join(instance_cls_savedir, str(lab_ind))
            if not osp.exists(save_dir):
                os.makedirs(save_dir)
            tmp = copy.deepcopy(label_map)
            tmp[tmp!=lab_ind] = 0  # 仅当前一个lab_ind类别
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(tmp, connectivity=8)
            for i in range(1, num_labels):
                x,y,h,w,s = stats[i][:5]
                if s >= 2500:  # 连通域的面积阈值
                    roi_label = labels[y:y+w, x:x+h]
                    roi1 = image[y:y+w, x:x+h,0]
                    roi2 = image[y:y+w, x:x+h,1]
                    roi3 = image[y:y+w, x:x+h,2]
                    roi1[roi_label!=i] = 0  # 屏蔽box内背景干扰. 
                    roi2[roi_label!=i] = 0
                    roi3[roi_label!=i] = 0
                    cv2.imwrite(osp.join(save_dir, '{}_{}.jpg'.format(basename[:-4], i)), cv2.merge([roi1, roi2, roi3]))
                    cls_areas[lab_ind].append(s) 
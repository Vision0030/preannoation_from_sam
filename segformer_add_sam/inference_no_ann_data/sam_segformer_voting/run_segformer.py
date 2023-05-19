# coding=utf-8
import os
import numpy as np 
import os.path as osp 
import argparse
import cv2 


def ger_segformer_connecte_center(img_name, image, mask_area_thres):
    from segformer_points_prompt import get_segformer_mask
    # segformer_centroids = []
    res_label_map = np.zeros_like(image[:,:,0]).astype(np.uint8)
    segformer_mask = get_segformer_mask(img_name, args.segformer_checkpoint, args.segformer_config, args.device_segformer) 
    # predicted_labs = np.unique(segformer_mask)
    # if len(predicted_labs) > 1:
    #     for predicted_lab in predicted_labs:
    #         single_cls_centroids = []
    #         tmp = np.zeros_like(segformer_mask).astype(np.uint8)
    #         tmp[segformer_mask==predicted_lab] = 1
    #         num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(tmp, connectivity=8)
    #         for ins_id in range(1, num_labels):
    #             if stats[ins_id][-1] > mask_area_thres:
    #                 single_cls_centroids.append(centroids[ins_id])
    #     segformer_centroids.append(single_cls_centroids)

    return segformer_mask   # segformer_centroids



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--device_segformer', type=str, default='cuda:1') 
    parser.add_argument('--segformer_config', type=str, default='./segformer_config.py')
    parser.add_argument('--mask_area_thres', type=int, default=50) 
    parser.add_argument('--segformer_checkpoint', type=str, default='/home/jia.chen/worshop/big_model/SegFormer/work_dirs/iter_160000.pth')
    parser.add_argument('--data_dir', type=str, default='/mnt/data/jiachen/noann_data/data/2023-5-12')
    parser.add_argument('--segformer_maskdir', type=str, default='/home/jia.chen/worshop/big_model/SAM/former_dir')
    args = parser.parse_args()

    img_paths = [osp.join(args.data_dir, a) for a in os.listdir(args.data_dir)]

    for img_path in img_paths:
        print('segformer inference {} ~~~'.format(img_path))
        basename = osp.basename(img_path)
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        segformer_centroids = ger_segformer_connecte_center(img_path, image, args.mask_area_thres)
        cv2.imwrite(osp.join(args.segformer_maskdir, basename[:-3]+'png'), segformer_centroids)

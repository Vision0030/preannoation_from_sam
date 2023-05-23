# coding=utf-8 
import pandas as pd 
import os 
import argparse
import numpy as np 
import os.path as osp 
from subprocess import run
from segment_anything import sam_model_registry
from sam_model import SAM_mask
from bk_cvat_upload_ann.samres2citycapse import generate_gtFine 

'''
算法方案: source activate groupvit 
1. segformer和sam各inference一遍待标注数据, 两个mask结果voting对比, 得到预标注结果 
2. 分割结果按citycapse保存, 转成coco并上传cvat

cvat处理:
1. 新建三个分类级别标签:  1. perfect; 2. good; 3. bad 分别代表
    1. 便便地毯 几乎不需要修标注, 但需要子明check下是否漏了别的类别(漏了就一会修掉~)
    2. 纸巾液体, 检查一些漏检, 修细小边缘, 或 一些没有缠绕的线, 预标注结果不错, 可简单修下~ 
    3. 线, 预标注很差, 需ziming做大修or重新标的那些 (3就不要求ziming修了..)
2. 在cvat上check预标注结果, 按以上标准, 给出: 1,2,3标签 
3. 把123标签写入总数据的pkl中(need_human_annotation: 1,2:False, 3:True), 1,2标签可不外发标注, 3标签得外发标注~

'''

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='default')
    parser.add_argument('--sam_checkpoint', type=str, default='./sam_vit_h_4b8939.pth')
    parser.add_argument('--device_sam', type=str, default='cuda:4')
    parser.add_argument('--sam_segformer_voing_thre', type=float, default=0.01)  # or 50.0 
    parser.add_argument('--mask_area_thres', type=int, default=50) 
    parser.add_argument('--sam_mask_savedir', type=str, default='/home/jia.chen/worshop/big_model/SAM/sam_mask_dir')
    parser.add_argument('--data_pkl', type=str, default='/mnt/DATABASE/dvc_repos/small_fov_dataset/out_source/small_fov/倍赛/small_fov_file_info.pkl')
    parser.add_argument('--out_dir', type=str, default='/mnt/data/jiachen/haitian_outdir/default')
    parser.add_argument('--vis_dir', type=str, default='/mnt/data/jiachen/haitian_outdir/vis')
    parser.add_argument('--haitian_needlabels', type=dict, default={1: 'Rug', 2: 'Cable', 3: 'Tissue', 4: 'Poop', 5: 'Liquid'})   
    args = parser.parse_args()

    col_map = [[0, 255, 0], [0, 255, 255], [200, 100, 200], [0, 0, 255], [255,0,0]]
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device_sam)

    # read data.pkl in py3.8, 给到的参数是: pkl的路径, 和所有imgs的路径,npy形式save和load
    run(' /home/jia.chen/miniconda3/envs/py3.8/bin/python read3.9pkl.py --img_paths_npy=./all_img_paths.npy --data_pkl=/mnt/DATABASE/dvc_repos/small_fov_dataset/out_source/small_fov/倍赛/small_fov_file_info.pkl', shell=True)
    all_img_paths = np.load('./all_img_paths.npy')

    # run segformer_points_bbox_sam.py方法
    # run('/home/jia.chen/miniconda3/envs/open-mmlab/bin/python segformer_points_bbox_sam.py --img_paths=./all_img_paths.npy', shell=True)

    # run sam和segformer compare voting方法
    # 先segformer过一遍数据
    run('/home/jia.chen/miniconda3/envs/open-mmlab/bin/python run_segformer.py --segformer_maskdir=./former_dir --img_paths=./all_img_paths.npy --device_segformer=cuda:5', shell=True)
    # run sam, then compare 
    for img_path in all_img_paths:
        basename = osp.basename(img_path)
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        print('sam processing {} ~~~'.format(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # run sam (和segformer一起并行已经落盘好mask了~)
        # masks = SAM_mask(image, sam)
        masks = np.load(osp.join(args.sam_mask_savedir, basename[:-3]+'npy'))
        segformer_mask = cv2.imread(osp.join(args.segformer_maskdir, basename[:-3]+'png'), cv2.IMREAD_UNCHANGED)
        sam_add_segformer = np.zeros_like(segformer_mask).astype(np.uint8)
        for ind, mask_map in enumerate(masks):
            # mask_map = mask_dict['segmentation']
            tmp = copy.deepcopy(segformer_mask)
            # 针对sam分割出的每一个bin, 对应去取segformer里的pred_lab_value
            tmp[mask_map==False] = 0  
            labs = np.unique(tmp)
            if len(labs) == 1:  # [0]
                continue 
            # [0, lab1, lab2, ...]

            # voting1:  
            all_areas = len(np.where(mask_map==True)[0])  # sam这块bin的总像素个数 
            lab_rate = [0]
            for lab in labs[1:]:   # 考虑sam的边缘优先级好于segformer, so, tmp里出现多个lab_i的话, 我们按像素点多的那个类去赋值label_value
                cur_lab_count = len(np.where(tmp==lab)[0])
                if args.sam_segformer_voing_thre <= 1:  # 是segformer/sam的比例
                    cur_lab_count /= float(all_areas)
                lab_rate.append(cur_lab_count) 
            max_rate = max(lab_rate)
            if max_rate > args.sam_segformer_voing_thre:  # 卡个阈值, 避免segformer过检导致image内到处都是object. 
                sam_add_segformer[mask_map==True] = labs[lab_rate.index(max_rate)]

        # vis ~ 
        color_res = vis_label_map(sam_add_segformer, col_map)
        labeluint8, instanceuint16, color_map = generate_gtFine(mask_area_thres=args.mask_area_thres, sam_label_res=sam_add_segformer, sam_color_map=color_res)
        dir_name = osp.dirname(img_path)
        date_info = dir_name.split('/')[-1]
        out_path = osp.join(args.out_dir, date_info)
        if not osp.exists(out_path):
            os.makedirs(out_path)
        cv2.imwrite(osp.join(out_path, basename[:-4]+'_gtFine_labelIds.png'), labeluint8)
        cv2.imwrite(osp.join(out_path, basename[:-4]+'_gtFine_instanceIds.png'), instanceuint16)
        cv2.imwrite(osp.join(out_path, basename[:-4]+'_gtFine_color.png'), color_map)
        vis_img = np.concatenate([color_map, image[:,:,::-1]], axis=0)
        vis_dir = osp.join(args.vis_dir, date_info)
        if not osp.exists(vis_dir):
            os.makedirs(vis_dir)
        cv2.imwrite(osp.join(vis_dir, basename[:-3]+'png'), vis_img)
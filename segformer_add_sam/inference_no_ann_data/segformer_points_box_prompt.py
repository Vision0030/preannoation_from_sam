# coding=utf-8
# segformer_points_prompt.py
'''
img = cv2.imread('/..._gtFine_labelIds.png', cv2.IMREAD_UNCHANGED)
img1 = cv2.imread('/..._gtFine_color.png', cv2.IMREAD_UNCHANGED)
# label_ids = np.unique(img)
# print(label_ids)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
for bin_id in range(1, num_labels):
    x,y,h,w,s = stats[bin_id][:5]
    if s > 30:  
        cv2.rectangle(img1, (x,y),(x+h, y+w), (0,255,0), 2) 
        for i in range(10):
            random_x, random_y = random.randint(x,x+h), random.randint(y, y+w)
            cv2.circle(img1, (random_x, random_y), 1, (0,0,255), 4)
        cv2.imwrite('1.jpg', img1)
test random neg points 

'''
import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor
import numpy as np 
import random
import cv2 
from shapely import geometry
class LoadImage:
    """A simple pipeline to load image."""
    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results
def init_segmentor(config, checkpoint=None, device='cuda:0'):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = checkpoint['meta']['PALETTE']
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model
def inference_segmentor(model, img):
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]
    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result

def get_segformer_mask(im_name, checkpoint, config, device):
    model = init_segmentor(config, checkpoint, device=device)
    segformer_res_mask = inference_segmentor(model, im_name)[0]
    return segformer_res_mask

def ner_count(cur_x, cur_y, ner_kernel_size, labels, bin_id, is_pos=True):
    tmp_count = 0
    for i in range(ner_kernel_size):
        for j in range(ner_kernel_size):
            ner_x, ner_y = cur_x+i, cur_y+j 
            if is_pos and labels[ner_x, ner_y]==bin_id:
                tmp_count += 1
            elif not is_pos and labels[ner_x, ner_y]!=bin_id:
                tmp_count += 1
    return tmp_count


def segformer2pointsprompt(segformer_mask, segformer2haitian, point_num=None, mask_area_thres=None, ner_kernel_size=None):
    instance_points = []
    instance_point_label = []
    if np.sum(segformer_mask) == 0:
        return np.array(instance_points), np.array(instance_point_label)
    need_labinds = [int(a) for a in segformer2haitian]   
    for need_ind in need_labinds:
        haitian_lab = segformer2haitian[str(need_ind)]
        if np.sum(segformer_mask==need_ind) > 0:
            tmp = np.zeros_like(segformer_mask).astype(np.uint8)
            tmp[segformer_mask==need_ind] = 1
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(tmp, connectivity=8)
            for bin_id in range(1, num_labels):
                x,y,h,w,s = stats[bin_id][:5]
                if s > mask_area_thres:   
                    point_coords = np.where(labels==bin_id)
                    for _ in range(point_num):   
                        random_ind = random.randint(0, len(point_coords[0])-1)  # 
                        # 随机选点pos点, pos需满足ner_kernel_size邻域点均在mask上
                        cur_x, cur_y = point_coords[0][random_ind], point_coords[1][random_ind]
                        tmp_count = ner_count(cur_x, cur_y, ner_kernel_size, labels, bin_id, is_pos=True)
                        if tmp_count == ner_kernel_size*ner_kernel_size: # 这里也可以放宽阈值条件
                            instance_points.append([cur_x, cur_y])
                            instance_point_label.append(haitian_lab) 
                    # 开始筛neg点
                    random_x, random_y = random.randint(y, y+w), random.randint(x,x+h)
                    if labels[random_x, random_y]!=bin_id: # 不在mask,so我找到neg点了, 继续看邻域也是neg就好了~
                        tmp_count = ner_count(cur_x, cur_y, ner_kernel_size, labels, bin_id, is_pos=False)
                        if tmp_count == ner_kernel_size*ner_kernel_size:  
                            instance_points.append([cur_x, cur_y])
                            instance_point_label.append(0)  # neg点label=0
    points = np.array(instance_points)
    pointslabels = np.array(instance_point_label)
    return points, pointslabels

def if_inPoly(polygon, Points):
    line = geometry.LineString(polygon)
    point = geometry.Point(Points)
    polygon = geometry.Polygon(line)
    return polygon.contains(point)
    
def nerpoints(ner_kernel_size, cur_x, cur_y, labels, bin_id, rect, is_pos=True):
    # cur_x, cur_y的ner_kernel_size邻域是否均满足在mask上且某个ann-box内
    for i in range(ner_kernel_size):
        for j in range(ner_kernel_size):
            ner_x, ner_y = cur_x+i, cur_y+j 
            if is_pos:
                if labels[ner_x, ner_y] == bin_id:  # ner也在cur_mask上
                    if not if_inPoly(rect, (ner_x, ner_y)):   # 也在此box-ann内
                        return False 
            else:  # 筛选neg点
                if labels[ner_x, ner_y] != bin_id:  # ner在mask上
                    if not if_inPoly(rect, (ner_x, ner_y)):    
                        return False 
    return True 


def segformer_pos_neg_points_in_box(segformer_mask, segformer2haitian, box_list, point_num=None, mask_area_thres=None, ner_kernel_size=None):
    # 基于ann-box信息, 给出更准备的pos neg points
    '''
    1. pos在box内且是seg_mask_res上, 且满足: cur_point的ner_kernel_size邻域都是满足box内且mask上.
    2. neg在box内但不在seg_mask_res上, 且满足: cur_point的ner_kernel_size邻域都是满足box内非mask上.
    3. segformer_maskbox_annbox_iou_thres 先打印iou看看这个值大概什么范围..
    '''
    instance_points = []
    instance_point_label = []
    if np.sum(segformer_mask) == 0:
        return np.array(instance_points), np.array(instance_point_label)
    need_labinds = [int(a) for a in segformer2haitian]   
    for need_ind in need_labinds:
        haitian_lab = segformer2haitian[str(need_ind)]
        if np.sum(segformer_mask==need_ind) > 0:
            tmp = np.zeros_like(segformer_mask).astype(np.uint8)
            tmp[segformer_mask==need_ind] = 1
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(tmp, connectivity=8)
            for bin_id in range(1, num_labels):
                if stats[bin_id][-1] > mask_area_thres:  
                    point_coords = np.where(labels==bin_id)
                    for _ in range(point_num):  
                        random_ind = random.randint(0, len(point_coords[0])-1)
                        cur_x, cur_y = point_coords[0][random_ind], point_coords[1][random_ind]  # mask上的某个点
                        for box_ in box_list:
                            rect = [(box_[0],box_[1]), (box_[2],box_[1]), (box_[2],box_[3]), (box_[0],box_[3])]
                            if if_inPoly(rect, (cur_x, cur_y)):  # 在某个ann-box内
                                if nerpoints(ner_kernel_size, cur_x, cur_y, labels, bin_id, rect, is_pos=True):  # 选pos点
                                    instance_points.append([cur_x, cur_y])
                                    instance_point_label.append(haitian_lab)   
                    for _ in range(point_num):  # 筛选neg点
                        for box_ in box_list:
                            rect = [(box_[0],box_[1]), (box_[2],box_[1]), (box_[2],box_[3]), (box_[0],box_[3])]
                            cur_x, cur_y = random.randint(int(box_[1]), int(box_[3])), random.randint(int(box_[0]), int(box_[2])) # 点在box内
                            if labels[cur_x, cur_y] != bin_id:  # 点不在mask上, so是neg点
                                # 继续看cur_x, cur_y是否也在box内但不在mask上
                                if nerpoints(ner_kernel_size, cur_x, cur_y, labels, bin_id, rect, is_pos=False):
                                    instance_points.append([cur_x, cur_y])
                                    instance_point_label.append(0) # neg点labelid=0   
    points = np.array(instance_points)
    pointslabels = np.array(instance_point_label)

    return points, pointslabels


def segformer2boxsprompt(segformer_mask, segformer2haitian, mask_area_thres=None):
    # Segformet inference一遍待标注数据, mask结果找连通域然后给box
    box_list = []
    box_label = []
    if np.sum(segformer_mask) == 0:
        return np.array(box_list), box_label
    need_labinds = [int(a) for a in segformer2haitian]   
    for need_ind in need_labinds:
        haitian_lab = segformer2haitian[str(need_ind)]
        if np.sum(segformer_mask==need_ind) > 0:
            tmp = np.zeros_like(segformer_mask).astype(np.uint8)
            tmp[segformer_mask==need_ind] = 1
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(tmp, connectivity=8)
            for bin_id in range(1, num_labels):
                x,y,h,w,s = stats[bin_id][:5]
                if s > mask_area_thres:   
                    box_list.append([x,y,x+h,y+w])
                    box_label.append(haitian_lab) 
    box_arry = np.array(box_list)

    return box_arry, box_label


if __name__ == '__main__':
    pass 
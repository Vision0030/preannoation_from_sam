# coding=utf-8
# segformer_points_prompt.py
'''
def if_inPoly(polygon, Points):
    line = geometry.LineString(polygon)
    point = geometry.Point(Points)
    polygon = geometry.Polygon(line)
    return polygon.contains(point)

img = cv2.imread('/mnt/data/jiachen/pre_ann_data/2023-5-12_bbox/河北_三线城市_THSJ小区_002_障碍物_线_IMG_20230509_153932_00_982_0_ht_gtFine_labelIds.png', cv2.IMREAD_UNCHANGED)
img1 = cv2.imread('/mnt/data/jiachen/pre_ann_data/2023-5-12_bbox/河北_三线城市_THSJ小区_002_障碍物_线_IMG_20230509_153932_00_982_0_ht_gtFine_color.png', cv2.IMREAD_UNCHANGED)
label_ids = np.unique(img)
# print(np.unique(img))
img[img!=0] = 1
img2 = cv2.merge([img,img,img])
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
for bin_id in range(1, num_labels):
    x,y,h,w,s = stats[bin_id][:5]
    box_ = [x,y,x+h, y+w]
    if s > 400:
        print(labels[100, 481], 'mask point~')
        print(box_)
        point_coords = np.where(labels==bin_id)  # point_coords[0], point_coords[1]
        # cv2.rectangle(img1, (box_[0],box_[1]),(box_[2], box_[3]), (0,255,0), 2)  
        coordinates = [(box_[0],box_[1]), (box_[2],box_[1]), (box_[2],box_[3]), (box_[0],box_[3])]
        for i, coor in enumerate(coordinates):
            cv2.circle(img1, (int(coor[0]),int(coor[1])), 1, (0,255,255), 1)
            cv2.imwrite('{}.jpg'.format(i), img1)
        cv2.circle(img1, (481, 100), 1, (255,0,0), 1)
        cv2.imwrite('{}.jpg'.format(4), img1)
        print(if_inPoly(coordinates, (481, 100)))

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
    h, w = labels.shape[:2]
    tmp_count = 0
    for i in range(ner_kernel_size):
        for j in range(ner_kernel_size):
            ner_x, ner_y = min(cur_x+i, h-1), min(cur_y+j, w-1)
            if is_pos:
                if labels[ner_x, ner_y]==bin_id:
                    tmp_count += 1
            else:
                if labels[ner_x, ner_y]==0:
                    tmp_count += 1

    return tmp_count


def segformer_points_bbox(segformer_mask, segformer2haitian, point_num=None, mask_area_thres=None, ner_kernel_size=None,find_times=None):
    tmp1 = np.zeros_like(segformer_mask).astype(np.uint8)
    tmp1[segformer_mask!=0] = 200 
    three_segformer_mask = cv2.merge([tmp1,tmp1,tmp1])
    '''
    无任何ann, 基于segformer出mask, 找bbox, 找邻域pos,neg points.
    '''
    instance_points = []
    instance_point_label = []
    box_list = []
    box_label = []
    if np.sum(segformer_mask) == 0:
        return np.array(instance_points), np.array(instance_point_label), np.array(box_list), box_label
    need_labinds = [int(a) for a in segformer2haitian]   
    for need_ind in need_labinds:
        haitian_lab = segformer2haitian[str(need_ind)]
        if np.sum(segformer_mask==need_ind) > 0:
            tmp = np.zeros_like(segformer_mask).astype(np.uint8)
            tmp[segformer_mask==need_ind] = 1
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(tmp, connectivity=8)
            # 找面积最大的那个instance
            instance_areas = [stat[4] for stat in stats[1:]]
            bin_id = instance_areas.index(max(instance_areas))+1
            # for bin_id in range(1, num_labels):
            x,y,h,w,s = stats[bin_id][:5]
            if s > mask_area_thres:    
                instance_find_times = 0
                goted_points = 0
                box_list.append([x,y,x+h,y+w])   # 满足mask_area_thres即可加入box_list
                box_label.append(haitian_lab) 
                point_coords = np.where(labels==bin_id)
                while goted_points < point_num:
                    while instance_find_times < find_times:
                        random_ind = random.randint(0, len(point_coords[0])-1)   # 随机选点pos点, pos需满足ner_kernel_size邻域点均在mask上
                        cur_x, cur_y = point_coords[0][random_ind], point_coords[1][random_ind]
                        assert labels[cur_x, cur_y] == bin_id
                        tmp_count = ner_count(cur_x, cur_y, ner_kernel_size, labels, bin_id, is_pos=True)
                        instance_find_times += 1
                        if tmp_count == ner_kernel_size*ner_kernel_size: # 这里也可以放宽阈值条件
                            instance_points.append([cur_x, cur_y])
                            instance_point_label.append(haitian_lab) 
                            goted_points += 1
                            # cv2.rectangle(three_segformer_mask, (x,y),(x+h,y+w), (0,255,0), 2)  
                            # cv2.circle(three_segformer_mask, (cur_y, cur_x), 10, (255,0,0), 1)
                            # cv2.imwrite('{}.jpg'.format(33), three_segformer_mask)  
                            if goted_points >= point_num:
                                break
                    goted_points += 1   # 找了限定次数没找到, 就给你虚假计数吧~ 不继续找了..
                print('goted_points: ', goted_points)
                # 筛neg点
                neg_instance_find_times = 0
                neg_goted_points = 0
                # cv2.rectangle(three_segformer_mask, (x,y),(x+h,y+w), (0,255,0), 2)  
                # cv2.imwrite('{}.jpg'.format(44), three_segformer_mask)  
                while neg_goted_points < point_num:
                    while neg_instance_find_times < find_times:
                        cur_x, cur_y = random.randint(y, y+w-1), random.randint(x,x+h-1)
                        if labels[cur_x, cur_y]==0:  
                            neg_instance_find_times += 1
                            tmp_count = ner_count(cur_x, cur_y, ner_kernel_size, labels, bin_id, is_pos=False)
                            if tmp_count == ner_kernel_size*ner_kernel_size:  
                                instance_points.append([cur_x, cur_y])
                                instance_point_label.append(0)  
                                neg_goted_points += 1
                                # cv2.rectangle(three_segformer_mask, (x,y),(x+h,y+w), (0,255,0), 2)  
                                # cv2.circle(three_segformer_mask, (cur_y, cur_x), 10, (0,255,0), 1)
                                # cv2.imwrite('{}.jpg'.format(44), three_segformer_mask)  
                                if neg_goted_points >= point_num:
                                    break
                    neg_goted_points += 1
                    print('neg_goted_points:', neg_goted_points)
    points = np.array(instance_points)
    pointslabels = np.array(instance_point_label)
    box_array = np.array(box_list)

    return points, pointslabels, box_array, box_label

def if_inPoly(polygon, Points):
    line = geometry.LineString(polygon)
    point = geometry.Point(Points)
    polygon = geometry.Polygon(line)
    return polygon.contains(point)
    
def nerpoints(ner_kernel_size, cur_x, cur_y, labels, bin_id, rect, is_pos=True):
    # cur_x, cur_y的ner_kernel_size邻域是否均满足在mask上且某个ann-box内
    h, w = labels.shape[:2]
    for i in range(ner_kernel_size):
        for j in range(ner_kernel_size):
            ner_x, ner_y = min(cur_x+i, h-1), min(cur_y+j, w-1)
            if is_pos:
                if labels[ner_x, ner_y] == bin_id:  # ner也在cur_mask上
                    if not if_inPoly(rect, (ner_y, ner_x)):   # 也在此box-ann内
                        return False 
            else:  # 筛选neg点
                if labels[ner_x, ner_y] == 0:  # ner在mask上
                    if not if_inPoly(rect, (ner_y, ner_x)):    
                        return False 
    return True 


def segformer_pos_neg_points_in_box(segformer_mask, segformer2haitian, box_list, point_num=None, mask_area_thres=None, pos_ner_kernel_size=None, neg_ner_kernel_size=None, find_times=None):
    tmp1 = np.zeros_like(segformer_mask).astype(np.uint8)
    tmp1[segformer_mask!=0] = 200 
    three_segformer_mask = cv2.merge([tmp1,tmp1,tmp1])
    # 基于ann-box信息, 给出更准备的pos neg points
    '''
    1. pos在box内且是seg_mask_res上, 且满足: cur_point的ner_kernel_size邻域都是满足box内且mask上.
    2. neg在box内但不在seg_mask_res上, 且满足: cur_point的ner_kernel_size邻域都是满足box内非mask上.
    3. segformer_maskbox_annbox_iou_thres  打印iou看看这个值大概什么范围? 

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
            # 找面积最大的那个instance
            instance_areas = [stat[4] for stat in stats[1:]]
            bin_id = instance_areas.index(max(instance_areas))+1
            # for bin_id in range(1, num_labels):
            if stats[bin_id][-1] > mask_area_thres:  
                point_coords = np.where(labels==bin_id)
                goted_point = 0
                found_time = 0
                while goted_point < point_num:
                    while found_time < find_times:
                        random_ind = random.randint(0, len(point_coords[0])-1)
                        cur_x, cur_y = point_coords[0][random_ind], point_coords[1][random_ind] 
                        assert labels[cur_x, cur_y] == bin_id
                        found_time += 1  
                        for box_ in box_list:
                            rect = [(box_[0],box_[1]), (box_[2],box_[1]), (box_[2],box_[3]), (box_[0],box_[3])]
                            if if_inPoly(rect, (cur_y, cur_x)) and nerpoints(pos_ner_kernel_size, cur_x, cur_y, labels, bin_id, rect, is_pos=True):   
                                instance_points.append([cur_x, cur_y])
                                instance_point_label.append(haitian_lab) 
                                # 检查下找到的这个pos点 
                                cv2.rectangle(three_segformer_mask, (box_[0],box_[1]),(box_[2], box_[3]), (0,255,0), 2)  
                                cv2.circle(three_segformer_mask, (cur_y, cur_x), 10, (255,0,0), 1)
                                # cv2.imwrite('{}.jpg'.format(33), three_segformer_mask)  
                                goted_point += 1
                                if goted_point >= point_num:
                                    break
                        break
                    break
                    goted_point += 1
                print(goted_point, 'goted_point')
                # for neg points:
                neg_goted_point = 0
                neg_found_time = 0
                while neg_goted_point < point_num:
                    while neg_found_time < find_times:
                        neg_found_time += 1
                        for box_ in box_list:
                            rect = [(box_[0],box_[1]), (box_[2],box_[1]), (box_[2],box_[3]), (box_[0],box_[3])]
                            cur_x, cur_y = random.randint(int(box_[1]), int(box_[3])-1), random.randint(int(box_[0]), int(box_[2])-1) 
                            if labels[cur_x, cur_y] == 0 and nerpoints(neg_ner_kernel_size, cur_x, cur_y, labels, bin_id, rect, is_pos=False):
                                instance_points.append([cur_x, cur_y])
                                instance_point_label.append(0) # neg点labelid=0 
                                neg_goted_point += 1
                                cv2.rectangle(three_segformer_mask, (box_[0],box_[1]),(box_[2], box_[3]), (0,255,0), 2)  
                                # # 检查下找到的这个neg点 
                                cv2.circle(three_segformer_mask, (cur_y, cur_x), 10, (0,0,0), 1)
                                cv2.imwrite('{}.jpg'.format(33), three_segformer_mask)  
                                if neg_goted_point >= point_num:
                                    break
                        break
                    break
                    neg_goted_point += 1
                print(neg_goted_point, 'neg_goted_point')
    points = np.array(instance_points)
    pointslabels = np.array(instance_point_label)

    return points, pointslabels



if __name__ == '__main__':
    pass 
    
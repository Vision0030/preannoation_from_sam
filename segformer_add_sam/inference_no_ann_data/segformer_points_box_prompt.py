# coding=utf-8
import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor
import numpy as np 
import random
import cv2 

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

def segformer2pointsprompt(segformer_mask, segformer2haitian, point_num=None, mask_area_thres=None):

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
                if stats[bin_id][-1] > mask_area_thres:  # 太细碎的连通域不要..
                    point_coords = np.where(labels==bin_id)
                    for _ in range(point_num):  # 每个连通域内找两个点
                        random_ind = random.randint(0, len(point_coords[0])-1)
                        instance_points.append([point_coords[0][random_ind], point_coords[1][random_ind]])
                        instance_point_label.append(haitian_lab)   
    points = np.array(instance_points)
    pointslabels = np.array(instance_point_label)

    return points, pointslabels


def segformer2boxsprompt(segformer_mask, segformer2haitian, mask_area_thres=None):
    # Segformet inference一遍待标注数据, mask结果找连通域然后给box
    box_list = []
    box_label = []
    if np.sum(segformer_mask) == 0:
        return np.array(box_list), box_label
    need_labinds = [int(a) for a in segformer2haitian]  # 2 4 便便, 线, 俩类
    for need_ind in need_labinds:
        haitian_lab = segformer2haitian[str(need_ind)]
        if np.sum(segformer_mask==need_ind) > 0:
            tmp = np.zeros_like(segformer_mask).astype(np.uint8)
            tmp[segformer_mask==need_ind] = 1
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(tmp, connectivity=8)
            for bin_id in range(1, num_labels):
                x,y,h,w,s = stats[bin_id][:5]
                if s > mask_area_thres:  # 太细碎的连通域不要..
                    box_list.append([x,y,x+h,y+w])
                    box_label.append(haitian_lab) 
    box_arry = np.array(box_list)

    return box_arry, box_label


if __name__ == '__main__':
    pass 
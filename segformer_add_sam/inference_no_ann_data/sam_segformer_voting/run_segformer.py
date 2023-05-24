# coding=utf-8
import os
import os.path as osp 
import argparse
import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor
import numpy as np 
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
    # print('segformer to device ~', device)
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
    data = collate([data], samples_per_gpu=10)    
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]
    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def get_segformer_mask(im_name, model):
    segformer_res_mask = inference_segmentor(model, im_name)[0]
    return segformer_res_mask


def ger_segformer_connecte_center(img_name, mask_area_thres):
    # segformer_mask = get_segformer_mask(img_name, args.segformer_checkpoint, args.segformer_config, args.device_segformer) 
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
    pass 
    # return segformer_centroids   # segformer_centroids



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--device_segformer', type=str, default='cuda:7') 
    parser.add_argument('--segformer_config', type=str, default='./segformer_config.py')
    parser.add_argument('--mask_area_thres', type=int, default=50) 
    parser.add_argument('--segformer_checkpoint', type=str, default='/home/jia.chen/worshop/big_model/SegFormer/work_dirs/iter_160000.pth')
    parser.add_argument('--img_paths', type=str, default='./all_img_paths.npy')
    parser.add_argument('--segformer_maskdir', type=str, default='./')
    args = parser.parse_args()

    # load一次model就好..~
    model = init_segmentor(args.segformer_config, args.segformer_checkpoint, device=args.device_segformer)
    # 每次都load, inference要7s, 只load一次然后inference,只要0.1s~ 

    # img_paths = [osp.join(args.data_dir, a) for a in os.listdir(args.data_dir)]
    img_paths = np.load(args.img_paths)
    for img_path in img_paths:
        print('segformer inference {} ~~~'.format(img_path))
        basename = osp.basename(img_path)
        # segformer_centroids = ger_segformer_connecte_center(img_path, args.mask_area_thres)
        segformer_centroids = get_segformer_mask(img_path, model) 
        cv2.imwrite(osp.join(args.segformer_maskdir, basename[:-3]+'png'), segformer_centroids.astype(np.uint8))

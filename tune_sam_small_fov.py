
# coding=utf-8
# ref: https://colab.research.google.com/drive/1F6uRommb3GswcRlPZWpkAQRMVNdVH7Ww?usp=sharing#scrollTo=lz7B4NDoJRxJ
# ref： https://github.com/bowang-lab/MedSAM/blob/main/train.py

import os
import os.path as osp 
import numpy as np 
import matplotlib.pyplot as plt
import cv2 
import torch 
from torch.utils.data import Dataset, DataLoader
import monai
from tqdm import tqdm
from statistics import mean
from torch.nn.functional import threshold, normalize

# torch.distributed.init_process_group(backend='nccl')
# local_rank = torch.distributed.get_rank()
# torch.cuda.set_device(local_rank)
# device = torch.device("cuda", local_rank)

from collections import defaultdict
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import SamPredictor, sam_model_registry

class jiachenDataset(Dataset): 
    def __init__(self, img_dir, seg_label_dir, image_size=(331, 331)):
        self.img_dir = img_dir
        self.seg_label_dir = seg_label_dir
        self.image_size = image_size
        self.gts = []
        self.input_image = []
        # self.org_size = None
    
        for gt_path in os.listdir(self.seg_label_dir):
            gt_img = cv2.imread(osp.join(self.seg_label_dir, gt_path))[:,:,0]
            # self.org_size = gt_img.shape
            gt_img = cv2.resize(gt_img, self.image_size)
            self.gts.append(gt_img)
        # self.gts = np.stack(gts, axis=0)
        for im_path in os.listdir(self.img_dir):
            img = cv2.imread(osp.join(self.img_dir, im_path))
            img = cv2.resize(img, self.image_size)
            self.input_image.append(img)
    
    def __len__(self):
        return len(self.gts)

    def __getitem__(self, index):
        img = self.input_image[index]
        img = torch.as_tensor(img, device='cuda:7').float()
        gt2D = self.gts[index]
        gt_mask = torch.as_tensor(gt2D[None, :,:], device='cuda:7').float()

        # 二值化找轮廓
        gt2D[gt2D!=0] = 255
        contours, hierarchy = cv2.findContours(gt2D,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]

        if len(contours) > 1:
            x,y,w,h = cv2.boundingRect(contours[0])
            box_list = [x, y, x+w, y+h]
        else:
            box_list = [0,0,0,0]
        bboxes = np.array(box_list)
        box_torch = torch.as_tensor(bboxes, device='cuda:7').float()

        return img.permute(2, 0, 1).contiguous(), gt_mask, box_torch


if __name__ == "__main__":
    img_dir = '/mnt/data/jiachen/small_fov_seg/val/real/vis/img'
    ann_dir = '/mnt/data/jiachen/small_fov_seg/val/real/vis/seg_label'
    train_dataset = jiachenDataset(img_dir, ann_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    model_type = 'vit_b'  #  'default'  
    checkpoint = 'sam_vit_b_01ec64.pth'   # './sam_vit_h_4b8939.pth'   # 3090单卡都放不下一张..
    device = 'cuda:7'
    sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
    sam_model.to(device)
    sam_model.train()
    lr = 1e-5
    wd = 0
    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=lr, weight_decay=wd)

    # 6. train 
    num_epochs = 100
    losses = []
    best_loss = 1e-3
    for epoch in range(num_epochs):
        epoch_loss = 0
        for step, (image, gt2D, box_torch) in enumerate(tqdm(train_dataloader)):
            image_embedding = sam_model.image_encoder(image) 
            # resize来匹配sam pretrain
            # 这里size写si了.. 不优雅...
            sam_trans = ResizeLongestSide(331)
            box_torch = sam_trans.apply_boxes_torch(box_torch, (320, 1024))  # 后面的是origin img size
            sam_trans = ResizeLongestSide(256)
            gt2D = sam_trans.apply_image_torch(gt2D)  

            # No grad here as we don't want to optimise the encoders
            with torch.no_grad():
                if len(box_torch.shape) == 2:
                    box_torch = box_torch[:, None, :] # (B, 1, 4)
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                                    points=None,
                                    boxes=box_torch,
                                    masks=None,)
            # 只train sam_model.mask_decoder()
            low_res_masks, iou_predictions = sam_model.mask_decoder(
                image_embeddings=image_embedding, # (B, 256, 64, 64)
                image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                multimask_output=False,)

            loss = seg_loss(low_res_masks, gt2D)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= step
        losses.append(epoch_loss)
        print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
        losses.append(epoch_loss)
        if epoch % 50 == 0:
            torch.save(sam_model.state_dict(), './iter{}_sam_small_fov.pth'.format(epoch))
        # save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(sam_model.state_dict(), './sam_model_best.pth')

    # %% plot loss
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('./train_loss.png')
    plt.close()

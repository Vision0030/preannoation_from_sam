#coding=utf-8 
import os 
import os.path as osp 
import numpy as np 
import cv2 
import torch
from PIL import Image
import torchvision
from torchvision import transforms
from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator


transform = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def img_transform(img_rgb, transform=None):
    img_t = transform(img_rgb)

    return img_t


sam = sam_model_registry['default'](checkpoint='./sam_vit_h_4b8939.pth')
device = 'cuda:4'
sam.to(device=device)

def SAM_mask(rgb_img, sam):
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(rgb_img)   
    return masks

# inference bins
device = torch.device('cuda:0')
model = torch.load('./resnet50_smallfov_cls.pth')
model.to(device)

def get_sam_roi_imgs():
    im_path = '20230506_四川省_成都_锦江区三单元5楼20号_颗粒物_IMG_20230504_140043_00_344_0.jpg'
    image = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam_mask_res = np.zeros_like(image[:,:,0]).astype(np.uint8)
    masks = SAM_mask(image, sam)
    for ind, mask_dict in enumerate(masks):
        mask_map = mask_dict['segmentation']
        tmp = np.zeros_like(mask_map).astype(np.uint8)
        tmp[mask_map==True] = 1 
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(tmp, connectivity=8)
        for i in range(1, num_labels):
            x,y,h,w,s = stats[i][:5]
            if s >= 2500:
                roi = image[y:y+w, x:x+h, :]
                img_tensor = img_transform(Image.fromarray(roi), transform)
                img_tensor.unsqueeze_(0)  
                img_tensor = img_tensor.to(device)  
                outputs = model(img_tensor)
                _, pred_int = torch.max(outputs.data, 1)
                pred_int = pred_int.cpu().numpy()[0]
                sam_mask_res[labels==i] = 255 
                print(pred_int)
                cv2.imwrite('/home/jia.chen/worshop/big_model/SAM/sam_bins/{}_{}.jpg'.format(ind, i), roi)
    cv2.imwrite('/home/jia.chen/worshop/big_model/SAM/111.jpg', cv2.merge([sam_mask_res,sam_mask_res,sam_mask_res]))
get_sam_roi_imgs()

# inference bins
# device = torch.device('cuda:0')
# model = torch.load('./resnet50_smallfov_cls.pth')
# model.to(device)
# roi_path = '/home/jia.chen/worshop/big_model/SAM/sam_bins'
# img_names = [osp.join(roi_path, a) for a in os.listdir(roi_path)]
# print(img_names)
# res = {}
# with torch.no_grad():
#     for idx, img_name in enumerate(img_names):
#         
#         res[osp.basename(img_name)] = pred_int
# print(res)




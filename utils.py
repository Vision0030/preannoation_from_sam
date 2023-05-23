import json 

# 删除background类别, citycapse2coco时候自动添加的, 但不需要这个类别.
data = json.load(open('./instances_default.json', 'r'))
anns = data['annotations']
slim_ann = []
for ann in anns:
    if ann['category_id'] != 1:
        ann['category_id'] -= 1
        slim_ann.append(ann)
data['annotations'] = slim_ann
data['categories'] = {'id': 1, 'name': 'wire', 'supercategory': ''}, {
    'id': 2, 'name': 'feces', 'supercategory': ''}
with open('./0518.json', 'w') as f:
    json.dump(data, f)



# read pandas .pkl in python>=3.8 
run('/home/jia.chen/miniconda3/envs/py3.8/bin/python read3.9pkl.py --img_paths_npy=./all_img_paths.npy --data_pkl=/mnt/DATABASE/dvc_repos/small_fov_dataset/out_source/small_fov/倍赛/small_fov_file_info.pkl', shell=True)
all_img_paths = np.load('./all_img_paths.npy')

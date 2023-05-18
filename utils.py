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

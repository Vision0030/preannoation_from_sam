# coding=utf-8
import os
import xml.etree.ElementTree as ET

def getimages(xmlname):
    sig_xml_box = []
    tree = ET.parse(xmlname)
    root = tree.getroot()
    images = {}
    for i in root:  # 遍历一级节点
        if i.tag == 'filename':
            file_name = i.text  # 0001.jpg
            # print('image name: ', file_name)
            images['file_name'] = file_name
        if i.tag == 'size':
            for j in i:
                if j.tag == 'width':
                    width = j.text
                    images['width'] = width
                if j.tag == 'height':
                    height = j.text
                    images['height'] = height
        if i.tag == 'object':
            for j in i:
                if j.tag == 'name':
                    cls_name = j.text
                if j.tag == 'bndbox':
                    bbox = []
                    xmin = 0
                    ymin = 0
                    xmax = 0
                    ymax = 0
                    for r in j:
                        if r.tag == 'xmin':
                            xmin = eval(r.text)
                        if r.tag == 'ymin':
                            ymin = eval(r.text)
                        if r.tag == 'xmax':
                            xmax = eval(r.text)
                        if r.tag == 'ymax':
                            ymax = eval(r.text)
                    bbox.append(xmin)
                    bbox.append(ymin)
                    bbox.append(xmax)
                    bbox.append(ymax)
                    bbox.append(cls_name)
                    # anno area
                    bbox.append((xmax - xmin) * (ymax - ymin) - 10.0)   # bbox的ares
                    # coco中的ares数值是 < w*h 的, 因为它其实是按segmentation的面积算的,所以我-10.0一下...
                    sig_xml_box.append(bbox)
                    # print('bbox', xmin, ymin, xmax - xmin, ymax - ymin, 'id', id, 'cls_id', cat_id)
    # print ('sig_img_box', sig_xml_box)
    return images, sig_xml_box
    
import os.path as osp 
if __name__ == "__main__":
    classes = []
    xml_dir = '/home/jia.chen/worshop/big_model/SAM/bedroom'
    xmls = [osp.join(xml_dir, a) for a in os.listdir(xml_dir) if '.xml' in a]
    for xml_file in xmls:
        _, sig_xml_bbox = getimages(xml_file)
        for single in sig_xml_bbox:
            cls_ = single[4]
            if cls_ not in classes:
                classes.append(cls_)
    print(classes)

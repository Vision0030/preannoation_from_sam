# get_get_ann_from_sam
用segment anything做自己任务数据的半自动标注


segformer出points prompt
ann-box; segformer-points; ann-box+segformer-points; segformer-box 4种形式prompt
结论: 
    ann-box最好, 
    ann-box+segformer-points反而有些漏检  不太好的points prompt影响好的ann-box prompt
    segformer-box形式比segformer-points形式好!
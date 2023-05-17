# get_get_ann_from_sam
用segment anything做下游任务数据预标注


segformer出predict mask 
ann-box; segformer-points; ann-box+segformer-points; segformer-box 4种形式prompt
结论: 
    ann-box较好,
    ann-box+segformer-points一些类别上优于ann-box 
    segformer-box形式比segformer-points形式好!
    segformer出的mask怎么点中好的point作为prompt需要好好设计!
    [找最大面积那个连通域,且考虑邻域点pos,neg属性的一致性~!]
![1](1.PNG)  
![2](2.PNG)
![3](3.PNG)
![4](4.PNG)
![5](5.PNG) 
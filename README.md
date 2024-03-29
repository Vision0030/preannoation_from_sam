# get_get_ann_from_sam
1. segment anything做下游任务数据预标注
    1. segformer train一个下游任务数据的分割模型, 精度到还不错的水平.
    2. sam接受的prompt形式: 
        1. annotation-box; 
        2. annotation-box+segformer-points; 
        3. segformer-points; 
        4. segformer-box
2. 结论: 
    1. annotation-box表现比较稳定
    2. annotation-box+segformer-points, 较之1提升效果不稳定, 非常依赖points的质量, 是否"命中"目标 
    3. 毫无annotation的冷启动数据标注:
        1. segformer-box形式比segformer-points形式好. box形式的prompt好于points.  
           [质量不稳定的一些零星点还是不如把这些点团成一个连通域找bbox呢~]
           [点给的越多,sam对显存的要求越大,玩不起哦~]
    4. segformer出的mask怎么点中好的points?
        1. 目前的做法: 1. 找area最大的那个连通域, 2. 考虑pos,neg邻域内属性一致
            1. pos的邻域可小些, 因为已经是在area最大的连通域上找pos了, 不至于random得太离谱point到物体的边界上
            2. neg的领域设置大一些, 保证这个点大概率是准的(对于有缠绕的线, 想要找到线之间的那些小镂空挺难的~)
            3. 搭配一个10 or 20次的find_point_times参数, 即找了find_point_times次后还没有指定个数的pos,neg点, 那就放弃不给points prompt了~  
                [至少保底出一个annotation-box的sam标注结果,不让不好的points prompt破坏sam的分割效果~]
    5. 训测同步非常重要!!! segformer训练的image size和待标注数据的size要保持一致~ 尽可能的让segformer出的mask准些吧~!
3. 更中肯的结论:
    1. 有box-annotation的数据, 直接给box prompt就不错了, 错的很少(我的落地场景,标注效率提升3倍~!). 用segformer辅助出points prompt反而恶化, [优化了蛮久但还是很费劲..点point-prompt就是很费劲!!!]
    2. 没有任何annotation的数据, 就用segformer出mask然后找连通域的bbox给sam做. 
    3. 打算尝试的新方案: 
        1. 有box-ann的, 可根据box信息把roi扣出来(4个边界留点冗余~), 再辅助上annbox-prompt给到sam出分割结果[又快又好,详见cropboxann_2_sam.py]
        2. 没有box-ann的(即毫无annotation冷启动预标注), sam, segformer各inference一遍, 然后个两个结果做配对, segformer有一定的提供cls信息的能力. 想到的方案: 
            1. 还是segformer先出一遍mask_res, 找连通域的质心(也就是cv2.connectedComponentsWithStats的centroids),同样sam也过一遍原图出mask_res, 用segformer的质心去选择sam中的分割块. 被命中分割块作为预标注结果. 
            ![center](center.PNG) 
            [找连通域的质量心并不是好的做法, 质心点很可能就没点中object, 如红圈圈中的那个.]
            还是想把sam的好边缘优势用起来, 但怎么做sam和Segformer之间的compare(or voting), 得好好设计~
            2. segformer推理一遍待标注数据, 得到mask_res然后找连通域(设置个面积阈值筛掉些零碎的分割块哦~)出bounding box(以下简称bbox), bbox作为annbox prompt, 走crop_annbox_sam.py pipeline. 
            局限: 考验segformer的recall object能力, bbox的个数, 大小可能都不完全准确~ 在bbox内把sam的好边缘利用起来!  [这个晚点code,代码基本可复用,就先不缝合轮子了~~~]
            3. 考虑sam的边缘更好, 所以拿各个sam的mask块去取segformer_mask中的值. 这样就可获取类别信息了. 取segformer_mask内像素个数最多的那个类别作为检出结果的label_value(信任sam没有把不同类别miss到同一个连通域内). 另外要卡一个segformer/sam的阈值(lab_rate), 防止segformer把背景检为目标, 然后在voting配对时候, 把sam检出的背景赋值label_value了~
            4. sam后面接个分类模型得了.[狗头]
            ![8](8.png) 

4. 可视化结果:
    ![1](1.PNG)  
    ![2](2.PNG)
    ![3](3.PNG)
    ![4](4.PNG)
    ![5](5.PNG) 
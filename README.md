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
    1. 有box-annotation的数据, 直接给box prompt就不错了, 错的很少. 用segformer辅助出points prompt反而恶化, [优化了蛮久但还是很费劲..点point-prompt就是很费劲!!!]
    2. 没有任何annotation的数据, 就用segformer出mask然后找连通域的bbox给sam做. 
    3. 打算尝试的新方案: sam直接出分割结果, 用box annotation去框sam后的结果, annotation里也已经有cls信息了~ 

4. 可视化结果:
    ![1](1.PNG)  
    ![2](2.PNG)
    ![3](3.PNG)
    ![4](4.PNG)
    ![5](5.PNG) 
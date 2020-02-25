1.在计算model ppf 时，数据筛选边界条件在TODO1 处定义，可能存在后续问题，记录

```c++
d_distance= diameter * tau_d  	//
diameter=点云模型的棋盘最大距离（不是欧式最大距离）
F1 = int(dis / d_distance)=(dis/diameter)/tau_d
max_nDistance=(1/tau_d)+5 
    
if (F1 > this->max_nDistance)
    return -1;
所以现在很可能存在 模型中的点云无法采集的情况(空间中的四个角)
```

2. 目前采用的是线程数大于points，额外的线程也计算，但是输入值为0，计算值也不采纳。4096为大于当前model points的CUDA填充线程数(CUDA每个block塞满会更快，更利于地址对齐)。所以后期写一个计算函数，计算对于当前点云数分配的线程数的 函数。
3. 发现在调用PCL库中的法线计算函数时，最后一个点的法向量计算不出来！为NAN
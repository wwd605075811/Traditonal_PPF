```c++
d_distance= diameter * tau_d  	//
diameter=点云模型的棋盘最大距离（不是欧式最大距离）
F1 = int(dis / d_distance)=(dis/diameter)/tau_d
max_nDistance=(1/tau_d)+5 
    
if (F1 > this->max_nDistance)
    return -1;
所以现在很可能存在 模型中的点云无法采集的情况(空间中的四个角)
```


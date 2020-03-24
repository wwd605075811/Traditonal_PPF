1.NOTE:在调用PCL库中的法线计算函数时，噪声点法向量为NAN，在SurfaceMatching.cpp 的52行去除(不太好，后面改进).

2.参数列表：

```c++
//模型点降采样率
model_leaf_size = 1.5f;   
//场景点降采样率
scene_leaf_size = 1.8f;	  
//model采样步长(选计算ppf的点，1代表每个model点)
int refPointDownsampleFactor = 1;  
//scene采样步长(5代表每5个点取一个计算ppf)
int sceneStep =5;	
//对于空间距离的缩小率？(没整清楚)
float tau_d = 0.05; 
//聚类时判定是否可以划分为一个类时的标准,即angle和dis小于某个设定阈值
CreateTranformtion_HCluster(float angle_thresh=0.2, float dis_thresh=0.5);
//设置icp迭代时最多可移动距离, 初次在结果周围选择 其平方范围内的点(具体在库里)
icp.setMaxCorrespondenceDistance(8);
//计算评分
double score=icp.getFitnessScore(5);
```

​	
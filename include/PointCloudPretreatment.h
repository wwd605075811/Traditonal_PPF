#ifndef PPF_CLOUDPOINTINIT_H
#define PPF_CLOUDPOINTINIT_H
#pragma once
#include <fstream>
#include<sstream>
#include <string>
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/visualization/pcl_visualizer.h>
using namespace std;
typedef pcl::PointNormal PointNT;   //using PointNT = pcl::PointNormal;
typedef pcl::PointCloud<PointNT> PointCloudNT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;

class PointCloudPretreatment {

public:
    PointCloudNT::Ptr modelDownsampled_;
    PointCloudNT::Ptr sceneDownsampled_;

    PointCloudPretreatment();
    /**
    * \简介 初始化model scene点云，输入6个参数。model 点云需要提前计算好法向量
    * scene点云目前只能分割一个平面，若scene平面已经分割，则去平面参数可设置为0
    * \参数[in] 
    *  1.model 点云文件路径
    *  2.model 降采样系数(PCL库)
    *  3.scene 点云文件路径
    *  4.去平面阈值参数(PCL库)
    *  5.法向量计算阈值(PCL库)
    * \生命周期
    */
    PointCloudPretreatment(const string &modelFilePath, float modelLeafSize, const string &sceneFilePath,
                   float backgroundThreshold, float normalThreshold, float sceneLeafSize);
    
    ~PointCloudPretreatment();

private:
    //场景点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene;
    //去掉平面的场景点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene_without_plane;
    //去掉平面且计算了法向量的点云
    PointCloudNT::Ptr sceneWithNormals;

    float modelLeafSize_;
    float backgroundThreshold;
    float normalThreshold;
    float sceneLeafSize;
    string modelFilePath_;
    string sceneFilePath;

    /**
     * \简介 得到降采样的model 点云，若文件路径下已经有当前所需的下采样规模的
     * model时，直接调用(没有完全实现，所以if分支为 0)
     */
    void setModel();
    /**
     * \简介 若本地没有降采样过的model，则对原始model降采样
     */
    void initModel();
    /**
     * \简介 判断输入文件为pcd or ply，保存scnen点云并调用-去背景-算法向量-降采样
     * todo判断文件名可以使用std::vector下的函数，更稳定
     */
    void setScene();
    /**
     * \简介 PCL库例子 去背景平面并显示结果
     */
    void segmentBackground(pcl::PointCloud<pcl::PointXYZ>::Ptr scene);
    /**
     * \简介 PCL库例子 计算scene法向量，当阈值设置的小时，法向量更准确
     * 目前使用的为OMP加速法向量计算，也不够快
     * todo：用有组织的点云计算法向量 pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal>
     */
    void estimateNormal();
    /**
     * \简介 scene点云降采样
     */
    void downSample();
};
#endif //PPF_CLOUDPOINTINIT_H

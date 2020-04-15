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
    PointCloudPretreatment(const string &modelFilePath, float modelLeafSize, const string &sceneFilePath,
                   float backgroundThreshold, float normalThreshold, float sceneLeafSize);
    ~PointCloudPretreatment();

private:
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene;
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene_without_plane;
    PointCloudNT::Ptr sceneWithNormals;

    float modelLeafSize_;
    float backgroundThreshold;
    float normalThreshold;
    float sceneLeafSize;
    string modelFilePath_;
    string sceneFilePath;

    void setModel();
    void initModel();
    void setScene();
    void segmentBackground(pcl::PointCloud<pcl::PointXYZ>::Ptr scene);
    void estimateNormal();
    void downSample();
};
#endif //PPF_CLOUDPOINTINIT_H

#include <fstream>
#include <iostream>
#include <string>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>

#include "../include/SurfaceMatching.h"
#include "../include/Model.h"
using namespace std;
using namespace pcl;

typedef pcl::PointNormal PointNT;   //using PointNT = pcl::PointNormal;
typedef pcl::PointCloud<PointNT> PointCloudNT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;

int main (int argc, char *argv[])
{
      ///The file to read from.
    string model_file_path = "../data/component.pcd";
    string scene_file_path = "../dataTest/s.pcd"; //s1.pcd

    ///The file to output to.
    string outfile = "../dataTest/output.pcd";

    pcl::PointCloud<pcl::PointXYZ>::Ptr scene(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::Normal>::Ptr scene_normals(new pcl::PointCloud<pcl::Normal>());
    PointCloudNT::Ptr model_with_normals(new PointCloudNT());
    PointCloudNT::Ptr scene_with_normals(new PointCloudNT());
    PointCloudNT::Ptr scene_downsampled(new PointCloudNT());
    PointCloudNT::Ptr model_downsampled(new PointCloudNT());

    pcl::io::loadPCDFile(model_file_path, *model_with_normals);
    cout << "model file loaded: " << model_file_path << endl;
    cout << "model_with_normals size:" << model_with_normals->size() << endl;

    // downsample model
    const float model_leaf_size = 1.5f;
    pcl::VoxelGrid<PointNT> sor;
    sor.setInputCloud(model_with_normals);
    sor.setLeafSize(model_leaf_size, model_leaf_size, model_leaf_size);
    sor.filter(*model_downsampled);
    cout << "model_downsampled size:" << model_downsampled->size() << endl;

    /*ColorHandlerT green(model_downsampled, 0,scene size: 255, 0);
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer());
    viewer->addPointCloud<PointNT>(model_downsampled, green, "result_cloud_icp");
    viewer->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
            4, "result_cloud_icp");

    while (!viewer->wasStopped()) {
        viewer->spinOnce();
    }*/
    pcl::io::loadPCDFile(scene_file_path, *scene_downsampled);
    cout << "scene file loaded:" << scene_file_path << endl;
    cout << "scene size:" << scene_downsampled->size() << endl;

    /*/// estimate normals of scene
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_scene(new pcl::search::KdTree<pcl::PointXYZ>());
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne_scene;
    ne_scene.setInputCloud(scene);
    ne_scene.setSearchMethod(tree_scene);
    ne_scene.setRadiusSearch(10.0f);
    ne_scene.compute(*scene_normals);
    pcl::concatenateFields(*scene, *scene_normals, *scene_with_normals);
    cout << "scene_with_normals size:" << scene_with_normals->size() << endl;

    //downsample scene
    const float scene_leaf_size = 1.2f;
    pcl::VoxelGrid<PointNT> sor_scene;
    sor_scene.setInputCloud(scene_with_normals);
    sor_scene.setLeafSize(scene_leaf_size, scene_leaf_size, scene_leaf_size);
    sor_scene.filter(*scene_downsampled);
    cout << "scene_downsampled size:" << scene_downsampled->size() << endl;*/

   /////// pcl::PCDWriter writer;writer.write<PointNT> (outfile.c_str (), *scene_downsampled, false);

    /*///caculate background
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    // Optional
    seg.setOptimizeCoefficients (true);
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (3);

    seg.setInputCloud (scene);
    seg.segment (*inliers, *coefficients);
    cout << "Background plane size: " << inliers->indices.size() << endl;
    cout << "Size of Foreground points:  " << scene->size() - inliers->indices.size() << endl;

    ///segment the background
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene_without_plane(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::ExtractIndices<pcl::PointXYZ> eifilter (true);
    eifilter.setInputCloud (scene);
    eifilter.setIndices (inliers);
    eifilter.setNegative (true);
    eifilter.filter (*scene_without_plane);
    */
    /*
    /// estimate normals of scene (too many ways, why chose this?)
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_scene(new pcl::search::KdTree<pcl::PointXYZ>());
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne_scene;
    ne_scene.setInputCloud(scene);
    ne_scene.setSearchMethod(tree_scene);   //Kd tree to find neighbouring points
    ne_scene.setRadiusSearch(10.0f);  //set the sphere radius,more small more specific
    ne_scene.compute(*scene_normals);
    pcl::concatenateFields(*scene, *scene_normals, *scene_with_normals);
    cout << "scene_with_normals size:" << scene_with_normals->size() << endl;
*/
    //downsample scene
    /*const float scene_leaf_size = 2.5f;
    pcl::VoxelGrid<PointNT> sor_scene;
    sor_scene.setInputCloud(scene_downsampled);
    sor_scene.setLeafSize(scene_leaf_size, scene_leaf_size, scene_leaf_size);
    sor_scene.filter(*scene_downsampled);
    cout << "scene_downsampled size:" << scene_downsampled->size() << endl;*/


    /// parameter
    int refPointDownsampleFactor = 1;
    int sceneStep = 5;
    float tau_d = 0.05;
    float clusterAngleThresh = 0.5;
    float clusterDisThresh = 50;

    pcl::PointNormal minpt, maxpt;
    pcl::getMinMax3D(*model_downsampled, minpt, maxpt);
    float diameter = 0;
    diameter = max(fabs(maxpt.z - minpt.z), max(fabs(maxpt.x - minpt.x), fabs(maxpt.y - minpt.y)));

    ///train and match on GPU
    Model *model = new Model(model_downsampled.get(), scene_downsampled.get(), diameter * tau_d,
                             sceneStep,refPointDownsampleFactor);
    ///Deal voting result and Cluster
    SurfaceMatching match(model_downsampled, scene_downsampled, sceneStep, model,
                          clusterAngleThresh, clusterDisThresh);
    ///ICP to Fine match with fitness score and Visualization
    match.icpAndGetFitnessScore();
    return 0;
}



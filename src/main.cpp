#include <string>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/point_operators.h>
#include <pcl/common/io.h>
#include <pcl/search/organized.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/io/vtk_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <fstream>
#include <iostream>
#include "SurfaceMatching.h"
using namespace std;
using namespace pcl;


typedef pcl::PointNormal PointNT;   //using PointNT = pcl::PointNormal;
typedef pcl::PointCloud<PointNT> PointCloudNT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;

int main (int argc, char *argv[])
{
      ///The file to read from.
    string model_file_path = "../data/component.pcd";
    string outfile = "../data/d1.pcd";

      ///The file to output to.
    string scene_file_path = "../data/scene_noBackground.pcd";

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

    std::cerr << "PointCloud after filtering: " << model_downsampled->width * model_downsampled->height
              << " data points (" << pcl::getFieldsList(*model_downsampled) << ").";

    /*ColorHandlerT green(model_downsampled, 0, 255, 0);
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer());
    viewer->addPointCloud<PointNT>(model_downsampled, green, "result_cloud_icp");
    viewer->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
            4, "result_cloud_icp");

    while (!viewer->wasStopped()) {
        viewer->spinOnce();
    }*/

    pcl::io::loadPCDFile(scene_file_path, *scene);
    cout << "scene file loaded:" << scene_file_path << endl;
    cout << "scene size:" << scene->size() << endl;

    ///caculate background
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

    /*///segment the background
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene_without_plane(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::ExtractIndices<pcl::PointXYZ> eifilter (true);
    eifilter.setInputCloud (scene);
    eifilter.setIndices (inliers);
    eifilter.setNegative (true);
    eifilter.filter (*scene_without_plane);*/
    /*pcl::PCDWriter writer;
     writer.write<pcl::PointXYZ> (outfile.c_str (), *scene_without_plane, false);*/

    /// estimate normals of scene (too many ways, why chose this?)
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_scene(new pcl::search::KdTree<pcl::PointXYZ>());
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne_scene;
    ne_scene.setInputCloud(scene);
    ne_scene.setSearchMethod(tree_scene);   //Kd tree to find neighbouring points
    ne_scene.setRadiusSearch(10.0f);  //set the sphere radius,more small more specific
    ne_scene.compute(*scene_normals);
    pcl::concatenateFields(*scene, *scene_normals, *scene_with_normals);
    cout << "scene_with_normals size:" << scene_with_normals->size() << endl;

    //downsample scene
    const float scene_leaf_size = 1.5f;
    pcl::VoxelGrid<PointNT> sor_scene;
    sor_scene.setInputCloud(scene_with_normals);
    sor_scene.setLeafSize(scene_leaf_size, scene_leaf_size, scene_leaf_size);
    sor_scene.filter(*scene_downsampled);
    cout << "scene_downsampled size:" << scene_downsampled->size() << endl;

    /*pcl::PCDWriter writer;
    writer.write<PointNT> (outfile.c_str (), *scene_downsampled, false);*/

    // do the ppf matching work
    cout << "training..." << endl;
    SurfaceMatching match(model_downsampled);
    cout<<"training is over!"<<endl;
    match.setScene(scene_downsampled);
    cout << "matching..." << endl;
    match.Voting();
    cout<<"voting is over!"<<endl;

    /*pcl::PCDWriter writer;
    writer.write<pcl::Normal> (outfile.c_str (), *scene_normals, false);*/
}



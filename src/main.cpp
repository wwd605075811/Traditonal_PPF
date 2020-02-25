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
#include <pcl/registration/icp.h>
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
    string outfile = "../data/scnenwwwwwwwww.pcd";

      ///The file to output to.
    string scene_file_path = "../data/scene_downsample_noground.pcd";

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

    pcl::io::loadPCDFile(scene_file_path, *scene_downsampled);
    cout <<endl<< "scene file loaded:" << scene_file_path << endl;
    cout << "scene size:" << scene_downsampled->size() << endl;

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
    *//*pcl::PCDWriter writer;
     writer.write<pcl::PointXYZ> (outfile.c_str (), *scene_without_plane, false);*//*

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
    cout << "scene_downsampled size:" << scene_downsampled->size() << endl;*/

    /*pcl::PCDWriter writer;
    writer.write<PointNT> (outfile.c_str (), *scene_downsampled, false);*/

    // do the ppf matching work
    cout << "training..." << endl;
    SurfaceMatching match(model_downsampled);
    cout<<"training is over!"<<endl;
    match.setScene(scene_downsampled);
    cout << "matching..." << endl;
    match.CudaVotingWithHash();
    cout<<"voting is over!"<<endl;

    struct timeval timeEnd1, timeEnd2, timeEnd3, timeSystemStart;
    double systemRunTime;
    gettimeofday(&timeSystemStart, NULL);

    match.CreateTranformtion_HCluster(0.2, 5);  // 参数待定

    PointCloudNT::Ptr bestResult = match.getBestResult();

    TransformWithProbVector transforms = match.getTransforms();

    for (TransformWithProb &transform : transforms) {
        cout << "probability: " << transform.prob << endl;
        Eigen::Matrix4f transform_mat = transform.transform_mat;
        cout << "before icp:" << endl;
        cout << transform_mat << endl;
        PointCloudNT::Ptr result_cloud(new PointCloudNT());
        pcl::transformPointCloud(*model_downsampled, *result_cloud, transform_mat);

        PointCloudNT::Ptr input_cloud = result_cloud;

        pcl::IterativeClosestPoint<PointNT, PointNT> icp;
        icp.setInputSource(input_cloud);
        icp.setInputTarget(scene_downsampled);
        // Set the max correspondence distance to 5cm (e.g., correspondences with higher distances will be ignored)
        icp.setMaxCorrespondenceDistance(10);
        // Set the maximum number of iterations (criterion 1)
        // icp.setMaximumIterations (50);
        // Set the transformation epsilon (criterion 2)
        // icp.setTransformationEpsilon (1e-8);
        // Set the euclidean distance difference epsilon (criterion 3)
        // icp.setEuclideanFitnessEpsilon (1);
        // icp.setMaximumIterations(10000);
        PointCloudNT Final;
        icp.align(Final);
        PointCloudNT::Ptr result_cloud_icp = Final.makeShared();
        Eigen::Matrix4f transform_mat_icp = icp.getFinalTransformation();

        cout << "after icp:" << endl;
        cout << transform_mat_icp * transform_mat << endl;

        pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer());
        ColorHandlerT red(model_downsampled, 255, 0, 0);
        viewer->addPointCloud<PointNT>(model_downsampled, red, "model_downsampled");
        viewer->setPointCloudRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                5, "model_downsampled");
        viewer->addPointCloudNormals<PointNT, PointNT>(
                model_downsampled,
                model_downsampled,
                1, 1,
                "model_downsampled_normals");

        ColorHandlerT yellow(scene_downsampled, 255, 255, 0);
        viewer->addPointCloud<PointNT>(scene_downsampled, yellow, "scene_downsampled");
        viewer->setPointCloudRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                4, "scene_downsampled");
        viewer->addPointCloudNormals<PointNT, PointNT>(
                scene_downsampled,
                scene_downsampled,
                1, 1,
                "scene_downsampled_normals");

        // viewer->addPointCloud(result_cloud, red, "result_cloud");
        // viewer->setPointCloudRenderingProperties(
        //     pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
        //     4, "result_cloud");

        ColorHandlerT green(result_cloud_icp, 0, 255, 0);
        viewer->addPointCloud<PointNT>(result_cloud_icp, green, "result_cloud_icp");
        viewer->setPointCloudRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                4, "result_cloud_icp");

                while (!viewer->wasStopped()) {
            viewer->spinOnce();
                }
                viewer->resetStoppedFlag();
                viewer->close();
    }


}



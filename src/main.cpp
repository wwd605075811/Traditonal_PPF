#include <fstream>
#include <iostream>
#include <string>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/normal_3d_omp.h>
#include "../include/PPFMatch.h"
#include "../include/PointCloudPretreatment.h"
using namespace std;
using namespace pcl;

typedef pcl::PointNormal PointNT;   //using PointNT = pcl::PointNormal;
typedef pcl::PointCloud<PointNT> PointCloudNT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;

int main (int argc, char *argv[]) {
    string scene_file_path;
    string model_file_path;
    string outFile = "../model/newmodel.pcd";
    float clusterAngleThresh = 0.5;
    float clusterDisThresh = 8;

    if (argc == 5) {
        cout<< "CLA!" <<endl;
        model_file_path = argv[1];
        scene_file_path = argv[2];
        clusterAngleThresh = atof(argv[3]);
        clusterDisThresh = atof(argv[4]);
    } else {
        cout<< "read from file"<<endl;
        ///The file to read from.
        model_file_path = "../modelScene/component.pcd";
        //scene_file_path = "../dataTest/ruben/2_without_outliers.ply";
        scene_file_path = "../modelScene/scene/box.ply";
    }

    PointCloudPretreatment preprocess(model_file_path, 2.0, scene_file_path, 5.0, 5.0, 2.0);

    /// PPF parameter
    int modelStep = 1;
    int sceneStep = 5;
    float tau_d = 0.05;
    int resultNumber = 5;
    float icpMaxCorrespondenceDistance = 5;
    float scoreSearchRadius = 6;
    float scoreThresh = 3;

    PPFMatch ma(preprocess.modelDownsampled_, tau_d, modelStep);
    TransformWithProbVector matrixAndScoreList_ =
            ma.match(preprocess.sceneDownsampled_, sceneStep,
             clusterAngleThresh, clusterDisThresh,
             resultNumber, icpMaxCorrespondenceDistance,
             scoreSearchRadius, scoreThresh);

    return 0;
}



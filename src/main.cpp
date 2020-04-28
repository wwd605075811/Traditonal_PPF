#include <fstream>
#include <iostream>
#include <string>
#include <pcl/point_types.h>

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
    string outFile = "../modelScene/scene/ceBar.pcd";


    if (argc == 5) {
        cout<< "CLA!" <<endl;
        model_file_path = argv[1];
        scene_file_path = argv[2];
        clusterAngleThresh = atof(argv[3]);
        clusterDisThresh = atof(argv[4]);
    } else {
        cout<< "read from file"<<endl;
        ///The file to read from.
        model_file_path = "../data/component.pcd";
        scene_file_path = "../dataTest/qianyi/test1/cloudWithFilter4.ply";

        //model_file_path = "../sonor2018/1model/model1.pcd";
        //scene_file_path = "../sonor2018/1scene/Scene1.pcd";
    }

    PointCloudPretreatment preprocess(model_file_path, 2.0, scene_file_path, 5, 3.0, 2.0);
    /// PPF parameter
    float clusterAngleThresh = 0.5;
    float clusterDisThresh = 8;
    int modelStep = 1;
    int sceneStep = 5;
    float tau_d = 0.05;
    int resultNumber = 8;
    float icpMaxCorrespondenceDistance = 1.5;
    float scoreSearchRadius = 4;
    float scoreThresh = 2;

    PPFMatch ma(preprocess.modelDownsampled_, tau_d, modelStep);
    TransformWithProbVector matrixAndScoreList_ =
            ma.match(preprocess.sceneDownsampled_, sceneStep,
             clusterAngleThresh, clusterDisThresh,
             resultNumber, icpMaxCorrespondenceDistance,
             scoreSearchRadius, scoreThresh);

    return 0;
}
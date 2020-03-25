#ifndef SURFACE_MATCHING_H
#define SURFACE_MATCHING_H
#include <iostream>
#include<fstream>
#include<vector>
#include <pcl/registration/icp.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/StdVector>
#include "PPF.h"
#include "HCluster.h"
#include "Model.h"
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudNT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;
using namespace std;
struct TransformWithProb {
    Eigen::Matrix4f transform_mat;
    float prob;
};

struct ResultWithScore {
    PointCloudNT::Ptr resultCloud;
    PointCloudNT::Ptr realScene;
    PointCloudNT::Ptr alignedPointFromSceneToModel;
    float score;
    int id;
};
// to avoid misalignment problem, otherwise the program may crash,
// see https://eigen.tuxfamily.org/dox/group__TopicStlContainers.html for more details
typedef std::vector<TransformWithProb, Eigen::aligned_allocator<TransformWithProb>> TransformWithProbVector;

class SurfaceMatching {
public:
    ///Public Types
    //the model and scene point cloud
    pcl::PointCloud<pcl::PointNormal>::Ptr m_modelWithNormals_;
    pcl::PointCloud<pcl::PointNormal>::Ptr m_sceneWithNormals_;
    //0->plant point 1->表示前景物体 2->reference point
    vector<int> sceneLabelList_;
    //the step of picking the reference in scene point cloud
    int sceneStep_;

    ///Public Member Functions
    SurfaceMatching(pcl::PointCloud<pcl::PointNormal>::Ptr modelWithNormals,
            pcl::PointCloud<pcl::PointNormal>::Ptr sceneWithNormals,
            int step, Model *model,float clusterAngleThresh, float clusterDisThresh);
    ~SurfaceMatching();
    void pickReferencePoints();
    //put the voting result into vector for the cluster
    void dealVoteResult(Model *model);
    void setScene(pcl::PointCloud<pcl::PointNormal>::Ptr sceneWithNormals, int step);
    //Hierarchical clustering
    void createTranformtionHCluster(float angle_thresh, float dis_thresh);
    void icpAndGetFitnessScore();
    ///Get Function
    Eigen::Matrix4f getBestTransform();
    TransformWithProbVector getTransforms();
    pcl::PointCloud<pcl::PointNormal>::Ptr getBestResult();

protected:

    PPF ppfExtractor_;
    const int BEST_TRANS_ID = 0;
    TransformWithProbVector transformMatList_;
    vector<int> votingValueList_;
    vector< vector<float> > transformdataSet_;

};
#endif //SURFACE_MATCHING_h

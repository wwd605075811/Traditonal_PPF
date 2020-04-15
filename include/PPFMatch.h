#ifndef MATCH_H
#define MATCH_H
#include <iostream>
#include<fstream>
#include<vector>
#include <pcl/registration/icp.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/StdVector>
#include "../include/PPF.h"
#include "../include/HCluster.h"
#include "../include/PointCloudPretreatment.h"
#include "../include/Model.h"
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudNT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;
using namespace std;
struct TransformWithProb {
    Eigen::Matrix4f transform_mat;
    float prob;
};

struct QuaternionWithVote {
    int votes;
    vector<float> quaternion7;
};

struct ResultWithScore {
    PointCloudNT::Ptr resultCloud;
    PointCloudNT::Ptr realScene;
    PointCloudNT::Ptr alignedPointFromSceneToModel;
    Eigen::Matrix4f rotationMatrix;
    float score;
    int id;
    int prob;
};
// to avoid misalignment problem, otherwise the program may crash,
// see https://eigen.tuxfamily.org/dox/group__TopicStlContainers.html for more details
typedef std::vector<TransformWithProb, Eigen::aligned_allocator<TransformWithProb>> TransformWithProbVector;
typedef std::vector<ResultWithScore, Eigen::aligned_allocator<ResultWithScore>> ResultWithScoreVector;

class PPFMatch {
public:
    PPFMatch(pcl::PointCloud<pcl::PointNormal>::Ptr modelWithNormals, float tau_d, int modelStep);

    TransformWithProbVector match(pcl::PointCloud<pcl::PointNormal>::Ptr sceneWithNormals,
            int sceneStep, float clusterAngleThresh, float clusterDisThresh,
            int resultNumber, float icpMaxCorrespondenceDistance,
            float scoreSearchRadius, float scoreThresh);

    void setModel(pcl::PointCloud<pcl::PointNormal>::Ptr modelWithNormals, int modelStep, float tau_d);
    void setScene(pcl::PointCloud<pcl::PointNormal>::Ptr sceneWithNormals, int sceneStep,
                  float clusterAngleThresh, float clusterDisThresh);
    void setICP(int resultNumber, float icpMaxCorrespondenceDistance,
                float scoreSearchRadius, float scoreThresh);

        ~PPFMatch();
private:

    ///parameter
    //f1Dispersion_ = diameter * tau_d
    float f1Dispersion_;
    float tau_d_;
    int sceneStep_;
    int modelStep_;
    float clusterAngleThresh_;
    float clusterDisThresh_;
    int resultNumber_;
    float icpMaxCorrespondenceDistance_;
    float scoreSearchRadius_;
    float scoreThresh_;
    ///Types
    //the model and scene point cloud
    pcl::PointCloud<pcl::PointNormal>::Ptr m_modelWithNormals_;
    pcl::PointCloud<pcl::PointNormal>::Ptr m_sceneWithNormals_;

    //put the voting result into vector for the cluster
    void dealVoteResult(Model *model);
    ResultWithScoreVector icpAndGetFitnessScore();
    void withoutICP();
    void watchResultAfterPPF(TransformWithProbVector afterPPFTransformatList);
    TransformWithProbVector distanceClustering(TransformWithProbVector afterPPFTransformatList);
    TransformWithProbVector NMSClustering(vector<QuaternionWithVote> poses);
    bool posesWithinErrorDistance(Eigen::Matrix4f t1, Eigen::Matrix4f t2);
    bool posesWithinError(vector<float> t1, vector<float> t2);
    QuaternionWithVote averageCluster(vector<QuaternionWithVote> q);

    ///Hierarchical clustering
    PPF ppfExtractor_;
    void createTranformtionHCluster(float angle_thresh, float dis_thresh);
    Eigen::Matrix4f getBestTransform();
    TransformWithProbVector getTransforms();
    pcl::PointCloud<pcl::PointNormal>::Ptr getBestResult();
    const int BEST_TRANS_ID = 0;
    TransformWithProbVector transformMatList_;
    vector<int> votingValueList_;
    vector<vector<float> > transformdataSet_;

};
#endif //MATCH_h

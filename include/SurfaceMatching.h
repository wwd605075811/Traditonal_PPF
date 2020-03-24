#ifndef SURFACE_MATCHING_H
#define SURFACE_MATCHING_H
#include <iostream>
#include<fstream>
#include<vector>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/centroid.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
#include <pcl/console/time.h>
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/filters/voxel_grid.h>

#include "PPF.h"
#include "HCluster.h"
#include "../include/model.h"

#include <Eigen/StdVector>

using namespace std;

struct TransformWithProb
{
    Eigen::Matrix4f transform_mat;
    float prob;
};


// to avoid misalignment problem, otherwise the program may crash,
// see https://eigen.tuxfamily.org/dox/group__TopicStlContainers.html for more details
typedef std::vector<TransformWithProb, Eigen::aligned_allocator<TransformWithProb>> TransformWithProbVector;

class SurfaceMatching
{

public:
    pcl::PointCloud<pcl::PointNormal>::Ptr m_modelWithNormals;
    pcl::PointCloud<pcl::PointNormal>::Ptr m_sceneWithNormals;

    pcl::PointCloud<pcl::PointNormal>::Ptr sceneWithNormal;
    vector<int> sceneLabelList; //0-背景点，如平面，1-表示前景物体,2-表示参考点
    void  setScene(pcl::PointCloud<pcl::PointNormal>::Ptr sceneWithNormals);

    void newVoting(Model *model);
    void CreateTranformtion_HCluster(float angle_thresh, float dis_thresh); //层次聚类

    pcl::PointCloud<pcl::PointNormal>::Ptr getBestResult();
    Eigen::Matrix4f getBestTransform();
    TransformWithProbVector getTransforms();

    SurfaceMatching(pcl::PointCloud<pcl::PointNormal>::Ptr modelWithNormals);
    ~SurfaceMatching();
private:
    int hashTableSize;
    float tau_d;
    int  N_angle;
    PPF  ppfExtractor;

    void pickReferencePoints();

    const int BEST_TRANS_ID = 0;

    TransformWithProbVector  transformMatList;

    vector<int>  votingValueList;
    vector< vector<float> > transformdataSet;
};



#endif //SURFACE_MATCHING_h
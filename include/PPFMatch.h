#ifndef MATCH_H
#define MATCH_H
#include <iostream>
#include<fstream>
#include<vector>
#include <pcl/registration/icp.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/StdVector>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include "../include/PPF.h"
#include "../include/HCluster.h"
#include "../include/PointCloudPretreatment.h"
#include "../include/Model.h"
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudNT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;
using namespace std;

struct TransformWithProb {
    Eigen::Matrix4f transform_mat;  //旋转矩阵
    float prob;  
};
struct QuaternionWithVote {
    int votes;  //得票数
    vector<float> quaternion7; //四元数+平移坐标，共7个float
};
struct ResultWithScore {
    PointCloudNT::Ptr resultCloud;   //最终结果点云
    PointCloudNT::Ptr realScene;      //匹配到的场景点云
    PointCloudNT::Ptr alignedPointFromSceneToModel;     //real scene 中匹配上的点
    Eigen::Matrix4f rotationMatrix;  //旋转矩阵
    float score;  //得分
    int id;  //聚类总得票数排名
    int prob;
};
// to avoid misalignment problem, otherwise the program may crash,
// see https://eigen.tuxfamily.org/dox/group__TopicStlContainers.html for more details
typedef std::vector<TransformWithProb, Eigen::aligned_allocator<TransformWithProb>> TransformWithProbVector;
typedef std::vector<ResultWithScore, Eigen::aligned_allocator<ResultWithScore>> ResultWithScoreVector;

class PPFMatch {
public:

    /**
     * \简介 初始化PPF匹配
     * \参数[in] 
     *   1.算过点云的model
     *   2.model 取样步长，默认=1，也就是model点全部参与model Hash的构建
     *   3.model 模型的离散参数，默认0.05也就是把model分成20份(参照OPENcv库实现，可能理解不准确)
     */
    PPFMatch(pcl::PointCloud<pcl::PointNormal>::Ptr modelWithNormals, float tau_d, int modelStep);
    /**
     * \简介 初始化scene参数，PPF-cuda计算 ，设置ICP参数 ，处理PPF结果 ， PPF聚类
     * \参数[in]
     *   1.scene点云，算过法向量
     *   2.场景采样步长
     *   3.聚类角度阈值
     *   4.聚类距离阈值
     *   5.需要得到的聚类结果数(聚类投票的前n名)
     *   6.ICP阈值(阈值越小迭代越深入，PCL库还有很多ICP阈值，可以添加)
     *   7.对result pose，以scoreSearchRadius为半径搜索真实的scene零件(越小，纳入的噪声点越少)
     *   8.对于7.找到的real scene，判断其周围有result pose点的比率 (参考PCLgetScore 和pickit策略)
     *  \注意 7 8 参数用来计算评分，判断scene中的真实零件点有多少点匹配上，默认7参数稍微比8大
     *   8参数越小，评分越严格
     */
    TransformWithProbVector match(pcl::PointCloud<pcl::PointNormal>::Ptr sceneWithNormals,
            int sceneStep, float clusterAngleThresh, float clusterDisThresh,
            int resultNumber, float icpMaxCorrespondenceDistance,
            float scoreSearchRadius, float scoreThresh);
    /**
     * \简介 计算ppf所需的model参数
     * \参数[in] 
     *   1. model有法向量点云
     *   2. model采样步长
     *   3.tau_d
     */
    void setModel(pcl::PointCloud<pcl::PointNormal>::Ptr modelWithNormals, int modelStep, float tau_d);
    /**
     * \简介 保存scene点云参数 和 聚类参数
     * \参数[in]
     *   1.scene有法向量点云
     *   2.scene采样步长
     *   3.聚类角度阈值
     *   4.聚类距离阈值
     */
    void setScene(pcl::PointCloud<pcl::PointNormal>::Ptr sceneWithNormals, int sceneStep,
                  float clusterAngleThresh, float clusterDisThresh);
    /**
     * \简介 保存ICP参数
     * \参数[in]
     *   1.结果pose数
     *   2.icp阈值
     *   3.计算real scene阈值
     *   4.计算评分阈值
     */ 
    void setICP(int resultNumber, float icpMaxCorrespondenceDistance,
                float scoreSearchRadius, float scoreThresh);

    ~PPFMatch();
private:
    ///parameter
    float f1Dispersion_;         //f1Dispersion_ = diameter * tau_d
    float tau_d_;                     //
    int sceneStep_;                 //scene采样步长
    int modelStep_;                //model采样步长
    float clusterAngleThresh_;  //聚类角度阈值
    float clusterDisThresh_;      //聚类距离阈值
    int resultNumber_;              //结果pose数
    float icpMaxCorrespondenceDistance_;  //ICP阈值
    float scoreSearchRadius_;     //求real scene 阈值
    float scoreThresh_;                //评分阈值
    ///Types
    //the model and scene point cloud
    pcl::PointCloud<pcl::PointNormal>::Ptr m_modelWithNormals_;
    pcl::PointCloud<pcl::PointNormal>::Ptr m_sceneWithNormals_;

    /**
     * \简介 对PPF-cuda的返回结果做处理(转为四元数、旋转矩阵)
     * \参数[in] 
     *   1.PPF-cuda计算类
     *   todo:直接返回PPF-cuda 的三个保存结果的数组
     *   
     */
    void dealVoteResult(Model *model);
    /**
     * \简介 对前n名ICP精匹配再计算评分，然后按照评分排名，然后可视化
     * \参数[in]
     *   1.TransformWithProbVector 旋转矩阵vector
     * \参数[out]
     *   1.ResultWithScoreVector 结果vector，这个vector包含的东西多一点
     */
    ResultWithScoreVector icpAndGetFitnessScore(TransformWithProbVector transforms);
    /**
     * \简介 可视化聚类之后，ICP之前的结果
     * \参数[in]
     *   1.TransformWithProbVector 旋转矩阵vector
     */
    void watchWithoutICP(TransformWithProbVector transforms);
      /**
     * \简介 可视化PPF之后，聚类之前的结果
     * \参数[in]
     *   1.TransformWithProbVector 旋转矩阵vector
     */
    void watchResultAfterPPF(TransformWithProbVector afterPPFTransformatList);
      /**
     * \简介 不更新的层次聚类，类似PCL库的 PoseCluster
     * \参数[in]
     *   1.vector<QuaternionWithVote> 四元数和投票数
     * \参数[out]
     *   1.TransformWithProbVector 旋转矩阵vector和投票数
     */
    TransformWithProbVector clusteringNoUpdate(vector<QuaternionWithVote> poses);
     /**
     * \简介 聚类时，是否相似类别的判断
     * \参数[in]
     *   1.vector<float> t1 四元数和平移
     *   2.vector<float> t2 四元数和平移
     */
    bool posesWithinErrorQuaternion(vector<float> t1, vector<float> t2);
      /**
     * \简介 对clusterNoUpdate，最终把同一类别的pose做平均
     * \参数[in]
     *   1.vector<QuaternionWithVote>  同一个类下的所有pose
     * \参数[out]
     *   2.QuaternionWithVote 这些poses的平均位姿
     * Note:这里直接用算术平均数计算四元数的平均值原理上是不合理的，
     * 因为四元数对于旋转矩阵是2对1的映射(在dealVoteResult保证四元数都在同一方向)
     * 其次四元数的变化不是线性的，但是测试过四元数平均算法，结果并不好
     */
    QuaternionWithVote averageCluster(vector<QuaternionWithVote> q);
    TransformWithProbVector getTransformsWithNoUpdateCluster();
    TransformWithProbVector transformMatWithNoUpdateClusterList_;
    vector<QuaternionWithVote> quaternionWithVoteList_;

    //Hierarchical clustering
    PPF ppfExtractor_;
    /**
     * \简介 层次聚类
     * \参数[in] 
     *   1.角度阈值
     *   2.距离阈值
     */
    void createTranformtionHCluster(float angle_thresh, float dis_thresh);
    Eigen::Matrix4f getBestTransform();
    TransformWithProbVector getTransforms();
    TransformWithProbVector transformMatList_;
    vector<int> votingValueList_;
    vector<vector<float> > transformdataSet_;
    //After ICP
    TransformWithProbVector greatPose_;
};
#endif //MATCH_h

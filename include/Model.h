#ifndef __MODEL_H
#define __MODEL_H
#include <iostream>
#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "Scene.h"
//Algorithm library
#include <thrust/inner_product.h>
#include <thrust/binary_search.h>
struct VoteResult {
    int mPoint;     //对应的model点
    int mNumber;    //得到的票数
    int mAngle;     //旋转的角度
    int sPoint;     //对应的scene点
};

class Model {

public:
    Model();
    /**
     * \简介 载入model点到GPU中，计算PPFs，计算转角αm
     * \参数[in]
     *   1.model 点云
     *   2.scene点云
     *   3.model的离散参数(用来离散和判断点对距离是否过长)
     *   4.scene采样步长
     *   5.model采样步长
     *  NOTE：cuda代码主要参考github的cuda-ppf，在文档中详细说明
     */
    Model(pcl::PointCloud<pcl::PointNormal> *cloud_ptr, pcl::PointCloud<pcl::PointNormal> *sceneCloud_ptr,
          float d_dist, int sceneStep, unsigned int refPointDownsampleFactor = 1);
    ~Model();
    int numPoints();
    thrust::device_vector<float3> *getModelPoints();
    thrust::device_vector<float3> *getModelNormals();
    thrust::device_vector<int4> *getModelPPFs();
    thrust::device_vector<unsigned int> *getHashKeys();

    pcl::PointCloud<pcl::PointNormal> *cloud_ptr;

    //Result:each reference points in scene could find theirs mPoint, votes and angle.
    //todo: use the get function
    int *sPoint;
    int *mPoint;
    int *mNumber;
    int *mAngle;
    int *secondmPoint;
    int *secondmNumber;
    int *secondmAngle;
    //the number of reference point
    int refNum;

private:
    // Number of PPF in the mode. I.e., number of elements in each of
    // the following arrays;
    unsigned long n;

    // down sample step of scene.
    int sceneStep;

    // Vector of model points
    thrust::device_vector<float3> *modelPoints;

    // Vector of model normals
    thrust::device_vector<float3> *modelNormals;

    //Number of PPFs
    unsigned long ppfM;
    // Vector of model point pair features
    thrust::device_vector<int4> *modelPPFs;
    thrust::device_vector<float> *modelAngles;

    // For a scene, hashKeys stores the hashes of all point pair features.
    // For a model, hashKeys is an array of all UNIQUE hashKeys. A binary search
    //   should be used to find the index of desired hash key.
    thrust::device_vector<unsigned int> *hashKeys;

    float d_dist;
    /**
     * \简介 计算PPFs 和 转角
     * \参数[in]
     *   1.model 点坐标
     *   2.model 点法向量
     *   3.model点数量
     *   4.离散参数
     *   5.model采样步长
     */
    void initPPFs(thrust::host_vector<float3> *points, thrust::host_vector<float3> *normals, int n,
                  float d_dist, unsigned int refPointDownsampleFactor=1);

    //this is related to the hash table
    // Indices into data.
    // nonunique_hashkeys[i] == hash(data[hashkeyToDataMap[i]])
    thrust::device_vector<std::size_t> hashkeyToDataMap;

    // *unique* hashUniqueKeys.
    thrust::device_vector<unsigned int> hashUniqueKeys;
    // number of occurances of each hashkey
    thrust::device_vector<std::size_t> counts;
    // Indices in hashkeyToDataMap where blocks of identical hashkeys begin.
    thrust::device_vector<std::size_t> firstHashkeyIndex;

    /**
     * \简介 直方图算法，用来把排序后的Hash表变成，unique hash表
     * \参数[in]
     *   1.原始hash表
     *   2.unique hash表
     *   3.每个hash值的数量vector
     */
    template <typename Vector1, typename Vector2, typename Vector3>
    void histogram(const Vector1& input,  // assumed to be already sorted
                   Vector2& histogram_values,
                   Vector3& histogram_counts);
    /**
     * \简介 构建HashTable和相关索引表
     * \参数[in]
     *   1.每个model点对的原始ppfs
     * note:主要就是计算4个vector,在CUDA-ppf论文中详细说明
     */
    void initHash(thrust::device_vector<unsigned int>& data);
    /**
     * \简介 逐点对投票，然后整和结果，筛选结果
     * \参数[in]
     *   1.scene类
     *   note:这个函数是核心，比较复杂，分成两部分，投票和统计。
     *   投票部分详细写在PPF-cuda文档里
     *   统计:找出每个累加数组的前两名，取(2*参考点数 的前50%)作为投票结果
     */
    void voteAndResult(Scene *scene);
    /**
     * \简介 取出投票得到的pose中得票数前50%
     *  todo：50%是否合理？  取累加数组的最大值和次大值是否合理？
     */
    void selectResult();
    thrust::device_vector<unsigned int> getIndices(thrust::device_vector<unsigned int>&data_hashkeys);
};
#endif /* __MODEL_H */

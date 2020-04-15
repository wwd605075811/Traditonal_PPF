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
    int mPoint;
    int mNumber;
    int mAngle;
    int sPoint;
};

class Model {

public:
    Model();
    // every point in model will to be calculated the ppfs, so the refPointDownsampleFactor is 1.
    //It just like the down sample step.
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

    template <typename Vector1, typename Vector2, typename Vector3>
    void histogram(const Vector1& input,  // assumed to be already sorted
                   Vector2& histogram_values,
                   Vector3& histogram_counts);

    void initHash(thrust::device_vector<unsigned int>& data);

    void voteAndResult(Scene *scene);
    void selectResult();
    thrust::device_vector<unsigned int> getIndices(thrust::device_vector<unsigned int>&data_hashkeys);
};
#endif /* __MODEL_H */

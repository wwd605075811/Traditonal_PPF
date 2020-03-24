#ifndef __SCENE_H
#define __SCENE_H
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
//Algorithm library
#include <thrust/inner_product.h>

class Scene {

public:

    Scene();
    /*Model(pcl::PointCloud<pcl::PointNormal> *cloud_ptr, float d_dist,
          unsigned int refPointDownsampleFactor=1);*/
    Scene(pcl::PointCloud<pcl::PointNormal> *cloud_ptr, float d_dist,
          unsigned int refPointDownsampleFactor=1);

    ~Scene();
    int numPoints();
    int getSceneStep();
    thrust::device_vector<float3> *getScenePoints();
    thrust::device_vector<float3> *getSceneNormals();
    thrust::device_vector<int4> *getScenePPFs();
    thrust::device_vector<float> *getSceneAngles();
    thrust::device_vector<unsigned int> *getSceneHashKeys();

    pcl::PointCloud<pcl::PointNormal> *cloud_ptr;

protected:

    // Number of PPF in the mode. I.e., number of elements in each of
    // the following arrays;
    unsigned long n;

    int sceneStep;

    // Vector of model points
    thrust::device_vector<float3> *scenePoints;

    // Vector of model normals
    thrust::device_vector<float3> *sceneNormals;

    //Number of PPFs
    unsigned long ppfS;
    // Vector of model point pair features
    thrust::device_vector<int4> *scenePPFs;
    thrust::device_vector<float> *sceneAngles;

    // For a scene, hashKeys stores the hashes of all point pair features.
    // For a model, hashKeys is an array of all UNIQUE hashKeys. A binary search
    //   should be used to find the index of desired hash key.
    thrust::device_vector<unsigned int> *scenehashKeys;

    float d_dist;

    void initPPFs(thrust::host_vector<float3> *points, thrust::host_vector<float3> *normals, int n,
                  float d_dist, unsigned int refPointDownsampleFactor);

};


#endif /* __SCENE_H */

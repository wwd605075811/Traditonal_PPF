#include "../include/Scene.h"
#include "../include/Kernel.h"
Scene::Scene(){}

Scene::Scene(pcl::PointCloud<pcl::PointNormal> *cloud_ptr, float d_dist,
             unsigned int refPointDownsampleFactor) {
    /*
    *计算Scene PPF 和 Hash
    */
    this->cloud_ptr =cloud_ptr;
    this->d_dist = d_dist;
    this->sceneStep = refPointDownsampleFactor;
    cout<<"GPU::scene point size:"<<this->cloud_ptr->size()<<endl;
    cout<<"GPU::scene d_dist:"<<this->d_dist <<endl;
    cout<<"GPU::scene reference step:"<<this->sceneStep<<endl;
    thrust::host_vector<float3> *points =
            new thrust::host_vector<float3>(cloud_ptr->size());
    thrust::host_vector<float3> *normals =
            new thrust::host_vector<float3>(cloud_ptr->size());

    for(int i = 0; i < cloud_ptr->size(); i++){
        (*points)[i].x = (*cloud_ptr)[i].x;
        (*points)[i].y = (*cloud_ptr)[i].y;
        (*points)[i].z = (*cloud_ptr)[i].z;
        (*normals)[i].x = (*cloud_ptr)[i].normal_x;
        (*normals)[i].y = (*cloud_ptr)[i].normal_y;
        (*normals)[i].z = (*cloud_ptr)[i].normal_z;
    }

    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    this->initPPFs(points, normals, cloud_ptr->size(), d_dist,
                   refPointDownsampleFactor);
    cout<< "GPU::scenePPF size: " <<scenePPFs->size()<<endl;

    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    this->scenehashKeys = new thrust::device_vector<unsigned int>(this->scenePPFs->size());

    int blocks = std::min(((int)(this->scenePPFs->size()) + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_NBLOCKS);

    ppf_hash_kernel<<<blocks,BLOCK_SIZE>>>
            (RAW_PTR(this->scenePPFs),
            RAW_PTR(this->scenehashKeys),
            this->scenePPFs->size());
}

Scene::~Scene() {
    delete this->scenePoints;
    delete this->sceneNormals;
    delete this->scenePPFs;
    delete this->sceneAngles;
    delete this->scenehashKeys;
}

void Scene::initPPFs(thrust::host_vector<float3> *points, thrust::host_vector<float3> *normals, int n,
                     float d_dist, unsigned int refPointDownsampleFactor){
    this->n = n;
    this->ppfS = n * n;
    // check if these are used later or can be discarded after this function
    this->scenePoints = new thrust::device_vector<float3>(*points);
    this->sceneNormals = new thrust::device_vector<float3>(*normals);
    this->scenePPFs = new thrust::device_vector<int4>(ppfS);
    this->sceneAngles = new thrust::device_vector<float>(ppfS);

    // This will crash if n = 0;
    int blocks = std::min(((int)(this->n + BLOCK_SIZE) - 1) / BLOCK_SIZE, MAX_NBLOCKS);
    // ppfKernel computes ppfs and descritizes them, but does *not* hash them
    // hashing is done by ppf_hash_kernel, called only for model, not scene (model.cu:46)
    ppfKernel<<<blocks,BLOCK_SIZE>>>
                       (RAW_PTR(this->scenePoints),
                        RAW_PTR(this->sceneNormals),
                        RAW_PTR(this->scenePPFs),
                        n, refPointDownsampleFactor, this->d_dist);
    ppfAngle<<<blocks,BLOCK_SIZE>>>
                       (RAW_PTR(this->scenePoints),
                        RAW_PTR(this->sceneNormals),
                        RAW_PTR(this->sceneAngles),
                        n, refPointDownsampleFactor, this->d_dist);
}

int Scene::numPoints(){
    return this->n;
}

int Scene::getSceneStep(){
    return this->sceneStep;
}

thrust::device_vector<float3> *Scene::getScenePoints(){
    return this->scenePoints;
}

thrust::device_vector<float3> *Scene::getSceneNormals(){
    return this->sceneNormals;
}

thrust::device_vector<int4>* Scene::getScenePPFs(){
    return this->scenePPFs;
}

thrust::device_vector<float>* Scene::getSceneAngles(){
    return this->sceneAngles;
}

thrust::device_vector<unsigned int>* Scene::getSceneHashKeys(){
    return this->scenehashKeys;
}

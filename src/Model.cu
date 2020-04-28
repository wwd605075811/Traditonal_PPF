#include "../include/Model.h"
#include "../include/Kernel.h"
using namespace std;
bool resultVoteCompare(VoteResult t1, VoteResult t2){
    return t1.mNumber > t2.mNumber;
}

Model::Model() {}

Model::Model(pcl::PointCloud<pcl::PointNormal> *cloud_ptr, pcl::PointCloud<pcl::PointNormal> *sceneCloud_ptr,
        float d_dist, int sceneStep, unsigned int refPointDownsampleFactor){

    cout<<"GPU::model point size:" <<cloud_ptr->size()<<endl;
    this->cloud_ptr =cloud_ptr;
    this->d_dist = d_dist;
    this->sceneStep = sceneStep;

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

    this->initPPFs(points, normals, cloud_ptr->size(), d_dist,
                   refPointDownsampleFactor);
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    this->hashKeys = new thrust::device_vector<unsigned int>(this->modelPPFs->size());

    //note: the blocks is 1024, so that their are many grids in this kernel
    int blocks = std::min(((int)(this->modelPPFs->size()) + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_NBLOCKS);
    ppf_hash_kernel<<<blocks,BLOCK_SIZE>>>(RAW_PTR(this->modelPPFs),
                                        RAW_PTR(this->hashKeys),
                                        this->modelPPFs->size());

    this->initHash(*hashKeys);
    Scene *scene = new Scene(sceneCloud_ptr, d_dist, this->sceneStep);
    this->voteAndResult(scene);
}

Model::~Model() {
    delete this->modelPoints;
    delete this->modelNormals;
    delete this->modelPPFs;
    // delete this->hashKeys;
}

void Model::initPPFs(thrust::host_vector<float3> *points, thrust::host_vector<float3> *normals, int n,
                     float d_dist, unsigned int refPointDownsampleFactor){
    this->n = n;
    this->ppfM = n*n;
    // check if these are used later or can be discarded after this function
    this->modelPoints = new thrust::device_vector<float3>(*points);
    this->modelNormals = new thrust::device_vector<float3>(*normals);
    this->modelPPFs = new thrust::device_vector<int4>(ppfM);
    this->modelAngles = new thrust::device_vector<float>(ppfM);

    // This will crash if n = 0;
    int blocks = std::min(((int)(this->n + BLOCK_SIZE) - 1) / BLOCK_SIZE, MAX_NBLOCKS);
    // ppfKernel computes ppfs and descritizes them, but does *not* hash them
    // hashing is done by ppf_hash_kernel, called only for model, not for scene
    ppfKernel<<<blocks,BLOCK_SIZE>>>(RAW_PTR(this->modelPoints),
                                    RAW_PTR(this->modelNormals),
                                    RAW_PTR(this->modelPPFs),
                    n, refPointDownsampleFactor, this->d_dist);

    //calculate alpha
    ppfAngle<<<blocks,BLOCK_SIZE>>>
                       (RAW_PTR(this->modelPoints),
                               RAW_PTR(this->modelNormals),
                               RAW_PTR(this->modelAngles),
                               n, refPointDownsampleFactor, this->d_dist);
}

template <typename Vector1, typename Vector2, typename Vector3>
void Model::histogram(const Vector1& input,  // assumed to be already sorted
               Vector2& histogram_values,
               Vector3& histogram_counts){
    typedef typename Vector1::value_type ValueType; // input value type
    typedef typename Vector3::value_type IndexType; // histogram index type

    //Is this device_vector needed delete???
    thrust::device_vector<ValueType> data(input);
    IndexType num_bins = thrust::inner_product(data.begin(), data.end() - 1,
                                               data.begin() + 1,
                                               IndexType(1),
                                               thrust::plus<IndexType>(),
                                               thrust::not_equal_to<ValueType>());
    histogram_values.resize(num_bins);
    histogram_counts.resize(num_bins);

    thrust::reduce_by_key(data.begin(), data.end(),
                          thrust::constant_iterator<IndexType>(1),
                          histogram_values.begin(),
                          histogram_counts.begin());
}

void Model::initHash(thrust::device_vector<unsigned int>& data) {
    this->hashkeyToDataMap = thrust::device_vector<std::size_t>(this->ppfM);
    // set the elements of H to 0, 1, 2, 3, ... , n
    thrust::sequence(hashkeyToDataMap.begin(), hashkeyToDataMap.end());

    // Sort nonunique_hashkeys and hashkeyToDataMap.
    thrust::sort_by_key(data.begin(),
                        data.end(),
                        hashkeyToDataMap.begin());

    // Create array of unique hashkeys and their associated counts.
    this->hashUniqueKeys = thrust::device_vector<unsigned int>();
    this->counts = thrust::device_vector<unsigned int>();
    histogram(data, this->hashUniqueKeys, this->counts);

    // Find the indices in hashkeyToDataMap of the beginning of each block of identical hashkeys.
    // the firstHash is Starting Indices
    this->firstHashkeyIndex = thrust::device_vector<std::size_t>(this->hashUniqueKeys.size());
    thrust::exclusive_scan(this->counts.begin(),
                           this->counts.end(),
                           this->firstHashkeyIndex.begin());
}

void Model::voteAndResult(Scene *scene){
    // refNum means: the number of reference points in scene
    int refNum = (scene->numPoints()/scene->getSceneStep() + 1);
    this->refNum =refNum;
    thrust::device_vector<unsigned int> *sceneIndices =
            new thrust::device_vector<unsigned int>(scene->getSceneHashKeys()->size());

    //对于unique hash table，对sceneHashKey建立索引.
    //lower_bound函数功能是：输入数组A，B，C
    //对于B数组，在A中查找，找到第一个比自己大的值，返回其数组下标索引给C
    thrust::lower_bound(this->hashUniqueKeys.begin(),
                        this->hashUniqueKeys.end(),
                        scene->getSceneHashKeys()->begin(),
                        scene->getSceneHashKeys()->end(),
                        sceneIndices->begin());

    thrust::device_vector<int> *voteAccumSpace=
            new thrust::device_vector<int>(refNum * this->n * N_ANGLE);

    cout<<"GPU::voteAccumSpace size: "<<voteAccumSpace->size()<<endl;

    // start cuda timer
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    int blocks = std::min(((int)(scene->getSceneHashKeys()->size()) + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_NBLOCKS);
    ppf_vote_kernel<<<blocks,BLOCK_SIZE>>>
                             (RAW_PTR(scene->getSceneHashKeys()), RAW_PTR(sceneIndices),
                              thrust::raw_pointer_cast(this->hashUniqueKeys.data()),
                              thrust::raw_pointer_cast(this->counts.data()),
                              thrust::raw_pointer_cast(this->firstHashkeyIndex.data()),
                              thrust::raw_pointer_cast(this->hashkeyToDataMap.data()),
                              RAW_PTR(this->modelPoints), RAW_PTR(this->modelNormals),
                              this->n, RAW_PTR(scene->getScenePoints()),
                              RAW_PTR(scene->getSceneNormals()), scene->numPoints(),
                              RAW_PTR(voteAccumSpace),
                              RAW_PTR(this->modelPPFs),
                              RAW_PTR(scene->getScenePPFs()),
                              RAW_PTR(this->modelAngles),
                              RAW_PTR(scene->getSceneAngles()),
                              scene->getSceneHashKeys()->size(),
                              this->d_dist);

    // end cuda timer
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    cout<<"GPU::the time cost of voting is:"<<elapsedTime<<"ms"<<endl;

    //to select the max and second max value
    thrust::device_vector<int> *votePoint=
            new thrust::device_vector<int>(refNum);
    thrust::device_vector<int> *voteNumber=
            new thrust::device_vector<int>(refNum);
    thrust::device_vector<int> *voteAngle=
            new thrust::device_vector<int>(refNum);
    thrust::device_vector<int> *secondPoint=
            new thrust::device_vector<int>(refNum);
    thrust::device_vector<int> *secondNumber=
            new thrust::device_vector<int>(refNum);
    thrust::device_vector<int> *secondAngle=
            new thrust::device_vector<int>(refNum);

    blocks = std::min(((int)refNum + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_NBLOCKS);
    addVote<<<blocks,BLOCK_SIZE>>>(RAW_PTR(voteAccumSpace), RAW_PTR(votePoint),
            RAW_PTR(voteNumber), RAW_PTR(voteAngle), this->n, scene->numPoints(),
            RAW_PTR(secondPoint), RAW_PTR(secondNumber), RAW_PTR(secondAngle));

    this->sPoint =  (int *) malloc(sizeof(int) * refNum);
    this->mPoint =  (int *) malloc(sizeof(int) * refNum);
    this->mAngle =  (int *) malloc(sizeof(int) * refNum);
    this->mNumber =  (int *) malloc(sizeof(int) * refNum);
    this->secondmPoint =  (int *) malloc(sizeof(int) * refNum);
    this->secondmAngle =  (int *) malloc(sizeof(int) * refNum);
    this->secondmNumber =  (int *) malloc(sizeof(int) * refNum);
    for (int i = 0; i < refNum; ++i) {
        this->sPoint[i] = i * this->sceneStep;
        this->mPoint[i] = (*votePoint)[i];
        this->mNumber[i] = (*voteNumber)[i];
        this->mAngle[i] = (*voteAngle)[i];
        this->secondmPoint[i] = (*secondPoint)[i];
        this->secondmNumber[i] = (*secondNumber)[i];
        this->secondmAngle[i] = (*secondAngle)[i];
    }
    this->selectResult();
}

void Model::selectResult() {
    //this function is to select 50% of (max + second value)
    //todo to add a Threshold to select result
    std::vector<VoteResult> resultList;
    for (int i = 0; i < this->refNum; ++i) {
        VoteResult tempFirst;
        tempFirst.mPoint = this->mPoint[i];
        tempFirst.mNumber = this->mNumber[i];
        tempFirst.mAngle = this->mAngle[i];
        tempFirst.sPoint = i;
        resultList.push_back(tempFirst);
    }
    for (int j = 0; j < this->refNum; ++j) {
        VoteResult tempSecond;
        tempSecond.mPoint = this->secondmPoint[j];
        tempSecond.mNumber = this->secondmNumber[j];
        tempSecond.mAngle = this->secondmAngle[j];
        tempSecond.sPoint = j;
        resultList.push_back(tempSecond);
    }

    std::sort(resultList.begin(),resultList.end(),resultVoteCompare);

    for (int i = 0; i < this->refNum; ++i) {
        this->mPoint[i] = resultList[i].mPoint;
        this->mNumber[i] = resultList[i].mNumber;
        this->mAngle[i] = resultList[i].mAngle;
        this->sPoint[i] = resultList[i].sPoint * this->sceneStep;
    }
}

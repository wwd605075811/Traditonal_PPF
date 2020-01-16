//
// Created by wwd on 2019/12/23.
//
#include <sys/time.h>
#include "stdafx.h"
#include "SurfaceMatching.h"


SurfaceMatching::SurfaceMatching(pcl::PointCloud<pcl::PointNormal>::Ptr modelWithNormals) {
    this->accumSpace = NULL;

    hashTableSize = 20000;
    tau_d = 0.05;
    N_angle = 30;

    this->m_modelWithNormals = modelWithNormals;
    this->init();
}

void SurfaceMatching::init(){
    // int loadedTrainModel = this->LoadTrainModel("train_model.txt");
    int loadedTrainModel = 0;
    if(0 == loadedTrainModel)
    {
        this->Train();
    }
    this->InitAccumSpace();
}

SurfaceMatching::~SurfaceMatching() {

    cout<<"析构函数～:****over!"<<endl;
    if(accumSpace!= NULL) {

        for (int i = 0; i < m_modelWithNormals->size(); i++) {
            if (accumSpace[i] != NULL) {
                delete[](accumSpace[i]);
                accumSpace[i] = NULL;
            }

        }
        delete[] accumSpace;
        accumSpace = NULL;
    }
    cout<<"~over over over"<<endl;
}

void  SurfaceMatching::InitAccumSpace(){
    //2 wei array, used like java
    if (m_modelWithNormals == NULL ){
        std::cerr <<"m_modelWithNormals is NULL in InitAccumSpace"<< endl;
        return;
    }

    if (accumSpace != NULL){
        std::cerr <<"accumSpace is not NULL"<< endl;
        return;
    }
    accumSpace = new int*[m_modelWithNormals->size()];
    for (int i = 0; i < m_modelWithNormals->size(); i++) {
        accumSpace[i] = new int[ppfExtractor.nAngle];
    }

    for (int i = 0; i < m_modelWithNormals->size(); i++) {
        for (int j = 0; j < ppfExtractor.nAngle; j++) {
            accumSpace[i][j] = 0;
        }
    }
}


void SurfaceMatching::Train() {
    if (m_modelWithNormals == NULL) {
        std::cerr << "m_modelWithNormals is NULL in Train" << endl;
        return;
    }
    pcl::PointNormal minpt, maxpt;
    pcl::getMinMax3D(*m_modelWithNormals, minpt, maxpt);
    float diameter = 0;
    diameter = max(fabs(maxpt.z - minpt.z), max(fabs(maxpt.x - minpt.x), fabs(maxpt.y - minpt.y)));

    ppfExtractor.Set(tau_d * diameter, N_angle, int(1 / tau_d + 5), hashTableSize);

    ppfModel.clear();
    for (int i = 0; i < hashTableSize; i++) {
        vector<PPInfo> hashItemList;
        hashItemList.clear();
        ppfModel.push_back(hashItemList);
    }
    int totalTrainCount = m_modelWithNormals->size() * (m_modelWithNormals->size() - 1);

    ///gpu-cuda
    CudaTrain();
}

void SurfaceMatching::CudaTrain(){
    //m_modelWithNormals->size =3655, so the PointNum is bigger than it
    int PointNum=4096;
    cuda_PPFInfo *g_F1=(cuda_PPFInfo *)malloc(sizeof(cuda_PPFInfo)*PointNum);
    cuda_PPFInfo *g_F1_copy=(cuda_PPFInfo *)malloc(sizeof(cuda_PPFInfo)*PointNum);
    for (int i = 0; i < PointNum; i++){
        if(i>=m_modelWithNormals->size()){
            g_F1[i].x=0;
            g_F1[i].y=0;
            g_F1[i].z=0;
            g_F1[i].nomal_x=0;
            g_F1[i].nomal_y=0;
            g_F1[i].nomal_z=0;
            g_F1_copy[i].x=0;
            g_F1_copy[i].y=0;
            g_F1_copy[i].z=0;
            g_F1_copy[i].nomal_x=0;
            g_F1_copy[i].nomal_y=0;
            g_F1_copy[i].nomal_z=0;
        }
        else{
            g_F1[i].x=m_modelWithNormals->points[i].x;
            g_F1[i].y=m_modelWithNormals->points[i].y;
            g_F1[i].z=m_modelWithNormals->points[i].z;
            g_F1[i].nomal_x=m_modelWithNormals->points[i].normal_x;
            g_F1[i].nomal_y=m_modelWithNormals->points[i].normal_y;
            g_F1[i].nomal_z=m_modelWithNormals->points[i].normal_z;
            g_F1_copy[i].x=m_modelWithNormals->points[i].x;
            g_F1_copy[i].y=m_modelWithNormals->points[i].y;
            g_F1_copy[i].z=m_modelWithNormals->points[i].z;
            g_F1_copy[i].nomal_x=m_modelWithNormals->points[i].normal_x;
            g_F1_copy[i].nomal_y=m_modelWithNormals->points[i].normal_y;
            g_F1_copy[i].nomal_z=m_modelWithNormals->points[i].normal_z;
        }
    }
    cuda_PPFotherInfo h_other;
    h_other.g_P1=ppfExtractor.P1;
    h_other.g_P2=ppfExtractor.P2;
    h_other.g_P3=ppfExtractor.P3;
    h_other.g_P4=ppfExtractor.P4;
    h_other.Min_angle=ppfExtractor.Min_angle;
    h_other.Max_angle=ppfExtractor.Max_angle;
    h_other.nAngle=ppfExtractor.nAngle;
    h_other.d_distance=ppfExtractor.d_distance;
    h_other.max_hashIndex=ppfExtractor.max_hashIndex;
    h_other.hashTableSize=hashTableSize;

    float *h_fromGPU_alpha=(float *)malloc (sizeof(float)*PointNum*PointNum);
    int *h_fromGPU_hash=(int *)malloc (sizeof(int)*PointNum*PointNum);

    int *h_fromGPU_F1=(int *)malloc (sizeof(int)*PointNum*PointNum);
    int *h_fromGPU_F2=(int *)malloc (sizeof(int)*PointNum*PointNum);
    int *h_fromGPU_F3=(int *)malloc (sizeof(int)*PointNum*PointNum);
    int *h_fromGPU_F4=(int *)malloc (sizeof(int)*PointNum*PointNum);

    ModelCuda(g_F1,g_F1_copy,h_other,h_fromGPU_hash,h_fromGPU_alpha,h_fromGPU_F1,h_fromGPU_F2,h_fromGPU_F3,h_fromGPU_F4);

    cout<<"GPU done!"<<endl;

    for (int i = 0; i < m_modelWithNormals->size(); ++i) {
        for (int j = 0; j < m_modelWithNormals->size(); ++j) {
            if(j==i){
                continue;
            }
            if(h_fromGPU_hash[i*PointNum+j]>=0){
                PPInfo tempInfo;
                tempInfo.F1=h_fromGPU_F1[i*PointNum+j];
                tempInfo.F2=h_fromGPU_F2[i*PointNum+j];
                tempInfo.F3=h_fromGPU_F3[i*PointNum+j];
                tempInfo.F4=h_fromGPU_F4[i*PointNum+j];
                tempInfo.pt1_id = i;
                tempInfo.pt2_id = j;
                tempInfo.alpha = h_fromGPU_alpha[i*PointNum+j];
                ppfModel[h_fromGPU_hash[i*PointNum+j]].push_back(tempInfo);
            }
        }
    }
}

void SurfaceMatching::setScene(pcl::PointCloud<pcl::PointNormal>::Ptr sceneWithNormals) {
    m_sceneWithNormals = sceneWithNormals;
    this->pickReferencePoints();
}

void  SurfaceMatching::pickReferencePoints() {
    sceneLabelList.clear();
    for (int i = 0; i < m_sceneWithNormals->size(); i++) {
        sceneLabelList.push_back(1); //初始都是前景点
    }

    //每隔step个作为参考点
    int step = 5;
    for (int i = 0; i < m_sceneWithNormals->size(); i++) {
        if (sceneLabelList[i] == 1 && i%step==0){
            sceneLabelList[i] = 2;
        }
    }
/*    int  count=0;
    for (int i = 0; i < m_sceneWithNormals->size(); i++) {
        if (sceneLabelList[i] == 2) {
            count ++;
            cout<<" "<<i<<" ";
        }
    }
    cout<<endl<<count<<" "<<endl;*/
}

void SurfaceMatching::CudaVoting(){
    //m_sceneWithNormals->size =10091, and the reference point step is 5. 10091/5=2018,2018+1=2019, so the PointNum is bigger than 2019
    int PointNum=10240;
    int PointReference=2019;
    cuda_PPFInfo *g_F1=(cuda_PPFInfo *)malloc(sizeof(cuda_PPFInfo)*PointNum);
    cuda_PPFInfo *g_F1_Reference=(cuda_PPFInfo *)malloc(sizeof(cuda_PPFInfo)*PointReference);
    for (int i = 0; i < PointNum; i++){
        if(i>=m_sceneWithNormals->size()){
            g_F1[i].x=0;
            g_F1[i].y=0;
            g_F1[i].z=0;
            g_F1[i].nomal_x=0;
            g_F1[i].nomal_y=0;
            g_F1[i].nomal_z=0;
        }
        else{
            g_F1[i].x=m_sceneWithNormals->points[i].x;
            g_F1[i].y=m_sceneWithNormals->points[i].y;
            g_F1[i].z=m_sceneWithNormals->points[i].z;
            g_F1[i].nomal_x=m_sceneWithNormals->points[i].normal_x;
            g_F1[i].nomal_y=m_sceneWithNormals->points[i].normal_y;
            g_F1[i].nomal_z=m_sceneWithNormals->points[i].normal_z;
        }
    }
    int count=0;
    for (int i = 0; i < m_sceneWithNormals->size(); i++) {
        if (sceneLabelList[i] == 2) {
            g_F1_Reference[count].x=m_sceneWithNormals->points[i].x;
            g_F1_Reference[count].y=m_sceneWithNormals->points[i].y;
            g_F1_Reference[count].z=m_sceneWithNormals->points[i].z;
            g_F1_Reference[count].nomal_x=m_sceneWithNormals->points[i].normal_x;
            g_F1_Reference[count].nomal_y=m_sceneWithNormals->points[i].normal_y;
            g_F1_Reference[count].nomal_z=m_sceneWithNormals->points[i].normal_z;
            count ++;
        }
    }

    cuda_PPFotherInfo h_other;
    h_other.g_P1=ppfExtractor.P1;
    h_other.g_P2=ppfExtractor.P2;
    h_other.g_P3=ppfExtractor.P3;
    h_other.g_P4=ppfExtractor.P4;
    h_other.Min_angle=ppfExtractor.Min_angle;
    h_other.Max_angle=ppfExtractor.Max_angle;
    h_other.nAngle=ppfExtractor.nAngle;
    h_other.d_distance=ppfExtractor.d_distance;
    h_other.max_hashIndex=ppfExtractor.max_hashIndex;

    float *h_fromGPU_alpha=(float *)malloc (sizeof(float)*PointNum*PointReference);
    int *h_fromGPU_hash=(int *)malloc (sizeof(int)*PointNum*PointReference);

    int *h_fromGPU_F1=(int *)malloc (sizeof(int)*PointNum*PointReference);
    int *h_fromGPU_F2=(int *)malloc (sizeof(int)*PointNum*PointReference);
    int *h_fromGPU_F3=(int *)malloc (sizeof(int)*PointNum*PointReference);
    int *h_fromGPU_F4=(int *)malloc (sizeof(int)*PointNum*PointReference);

    SceneCuda(g_F1,g_F1_Reference,h_other,h_fromGPU_hash,h_fromGPU_alpha,h_fromGPU_F1,h_fromGPU_F2,h_fromGPU_F3,h_fromGPU_F4);

    cout<<"GPU scene done!"<<endl;

    //printf file to check the value
    ofstream  outFile("scene_ppf_cuda.txt");
    for (int i = 0; i < m_sceneWithNormals->size(); i++)
    {
        if (sceneLabelList[i] == 2){

            outFile << i << "****************************"<<endl;
            for (int j = 0; j< m_sceneWithNormals->size(); j++)
            {
                if (sceneLabelList[j] > 0 && (i != j)){
                    outFile << j<<"--"<<h_fromGPU_hash[j*PointReference+i]<<" "<<h_fromGPU_F1[j*PointReference+i]<<" "<<h_fromGPU_F2[j*PointReference+i]<<" "<<h_fromGPU_F3[j*PointReference+i]<<" "<<h_fromGPU_F4[j*PointReference+i]<<" "<<h_fromGPU_alpha[j*PointReference+i]<<" ";
                }
            }
            outFile << endl;
        }

    }
    outFile.close();

}

void SurfaceMatching::CudaVotingWithHash() {
    ///creat new hash table
    int hashNum=m_modelWithNormals->size() * (m_modelWithNormals->size()-1);
    PPInfo* modelHashValue=(PPInfo*)malloc(sizeof(PPInfo)*hashNum);
    PPInfo* modelHashKey[hashTableSize+1];
    int hashCount=0;
    for (int i = 0; i < ppfModel.size(); i++) {
        if (ppfModel[i].size() == 0) {
            modelHashKey[i] = &modelHashValue[hashCount];
        } else {
            for (int j = 0; j < ppfModel[i].size(); j++) {
                modelHashValue[hashCount] = ppfModel[i][j];
                if (j == 0) {
                    modelHashKey[i] = &modelHashValue[hashCount];
                }
                hashCount++;
            }
        }
    }
    //like the vector.end();
    PPInfo *ptrEnd=&modelHashValue[hashCount];
    modelHashKey[hashTableSize]=ptrEnd;

    cout<<modelHashKey[20000]-modelHashKey[0]<<" test key number!"<<modelHashKey[19999]-modelHashKey[0]<<endl;
    int* modelHashKeyIndex=(int*)malloc(sizeof(int)*hashTableSize);
    for (int k = 0; k < hashTableSize; ++k) {
        modelHashKeyIndex[k]=modelHashKey[k]-modelHashKey[0];
    }

    //m_sceneWithNormals->size =10091, and the reference point step is 5. 10091/5=2018,2018+1=2019, so the PointNum is bigger than 2019
    int PointReference = 2048;
    int PointNum = m_sceneWithNormals->size();
    int count = 0;
    cuda_PPFInfo *g_F1 = (cuda_PPFInfo *) malloc(sizeof(cuda_PPFInfo) * PointReference);
    cuda_PPFInfo *g_F2 = (cuda_PPFInfo *) malloc(sizeof(cuda_PPFInfo) * PointNum);
    for (int i = 0; i < m_sceneWithNormals->size(); i++) {
        if (sceneLabelList[i] == 2) {
            g_F1[count].x = m_sceneWithNormals->points[i].x;
            g_F1[count].y = m_sceneWithNormals->points[i].y;
            g_F1[count].z = m_sceneWithNormals->points[i].z;
            g_F1[count].nomal_x = m_sceneWithNormals->points[i].normal_x;
            g_F1[count].nomal_y = m_sceneWithNormals->points[i].normal_y;
            g_F1[count].nomal_z = m_sceneWithNormals->points[i].normal_z;
            count++;
        }
    }
    for (int i = 0; i < m_sceneWithNormals->size(); i++) {
        g_F2[i].x = m_sceneWithNormals->points[i].x;
        g_F2[i].y = m_sceneWithNormals->points[i].y;
        g_F2[i].z = m_sceneWithNormals->points[i].z;
        g_F2[i].nomal_x = m_sceneWithNormals->points[i].normal_x;
        g_F2[i].nomal_y = m_sceneWithNormals->points[i].normal_y;
        g_F2[i].nomal_z = m_sceneWithNormals->points[i].normal_z;
    }
    cuda_PPFotherInfo h_other;
    h_other.g_P1 = ppfExtractor.P1;
    h_other.g_P2 = ppfExtractor.P2;
    h_other.g_P3 = ppfExtractor.P3;
    h_other.g_P4 = ppfExtractor.P4;
    h_other.Min_angle = ppfExtractor.Min_angle;
    h_other.Max_angle = ppfExtractor.Max_angle;
    h_other.nAngle = ppfExtractor.nAngle;
    h_other.d_distance = ppfExtractor.d_distance;
    h_other.max_hashIndex = ppfExtractor.max_hashIndex;
    h_other.hashTableSize=hashTableSize;
    h_other.hashNum=hashNum;

    float *h_fromGPU_alpha = (float *) malloc(sizeof(float) * PointNum * PointReference);
    int *h_fromGPU_hash = (int *) malloc(sizeof(int) * PointNum * PointReference);
    int *h_fromGPU_F1 = (int *) malloc(sizeof(int) * PointNum * PointReference);
    int *h_fromGPU_F2 = (int *) malloc(sizeof(int) * PointNum * PointReference);
    int *h_fromGPU_F3 = (int *) malloc(sizeof(int) * PointNum * PointReference);
    int *h_fromGPU_F4 = (int *) malloc(sizeof(int) * PointNum * PointReference);

    Reference2NumCuda(g_F1, g_F2, h_other, h_fromGPU_hash, h_fromGPU_alpha, h_fromGPU_F1, h_fromGPU_F2, h_fromGPU_F3,
                  h_fromGPU_F4,modelHashValue,modelHashKeyIndex);

    cout << "GPU Voting done!" << endl;

    for (int i = 0; i < 2019; i++) {

        for (int ii = 0; ii < m_modelWithNormals->size(); ii++)
            for (int jj = 0; jj < ppfExtractor.nAngle; jj++)
            {
                accumSpace[ii][jj] = 0;
            }

        for (int j = 0; j < m_sceneWithNormals->size(); j++) {
            if (sceneLabelList[j] > 0 && (i * 5) != j) {
                if (h_fromGPU_F1[i * PointNum + j] > ppfExtractor.max_nDistance);   //not a correct value
                else {
                    vector<PPInfo> ppInfoList = ppfModel[h_fromGPU_hash[i * PointNum + j]];
                    for (int listID = 0; listID < ppInfoList.size(); listID++) {
                        int cur_F1 = ppInfoList[listID].F1;
                        int cur_F2 = ppInfoList[listID].F2;
                        int cur_F3 = ppInfoList[listID].F3;
                        int cur_F4 = ppInfoList[listID].F4;
                        float cur_alpha = ppInfoList[listID].alpha;
                        int pt_id = ppInfoList[listID].pt1_id;

                        int dis_thresh = 0;
                        int angle_thresh = 0;

                        if (isFeatureSimilar(dis_thresh, angle_thresh, h_fromGPU_F1[i * PointNum + j], cur_F1,
                                             h_fromGPU_F2[i * PointNum + j], cur_F2, h_fromGPU_F3[i * PointNum + j],
                                             cur_F3, h_fromGPU_F4[i * PointNum + j], cur_F4)) {
                            float alpha_m2s = cur_alpha - h_fromGPU_alpha[i * PointNum + j];

                            if (alpha_m2s < -M_PI)
                                alpha_m2s += M_PI * 2;
                            if (alpha_m2s > M_PI)
                                alpha_m2s -= M_PI * 2;

                            int angleID = (alpha_m2s - ppfExtractor.Min_angle) /
                                          (ppfExtractor.Max_angle - ppfExtractor.Min_angle) * ppfExtractor.nAngle;

                            accumSpace[pt_id][angleID] += 1;
                        }//if similar
                    }//for each same hash value
                }//for the right hash value
            }
        }//to all other points

        int rowLen = m_modelWithNormals->size();
        int colLen = ppfExtractor.nAngle;
        int maxAccum = -1;
        int idy_max, idx_max;

        for (int idy = 0; idy < rowLen; idy++)
            for (int idx = 0; idx < colLen; idx++)
            {
                int votingValue = accumSpace[idy][idx];
                if (votingValue > maxAccum)
                {
                    maxAccum = votingValue;
                    idy_max = idy;
                    idx_max = idx;
                }
            }
           //cout << "GPUMaxAccum:  i" <<i<<"  "<< maxAccum << endl;
        votingValueList.push_back(maxAccum);
    }
    cout<<"cuda is over!"<<endl;

    //printf file to check the value
    /*ofstream outFile("scene_ppf_cuda.txt");
    for (int i = 0; i < 2019; i++) {
        outFile << i * 5 << "****************************" << endl;
        for (int j = 0; j < m_sceneWithNormals->size(); j++) {
            if ((i * 5) != j) {
                if (h_fromGPU_F1[i * PointNum + j] > ppfExtractor.max_nDistance)
                            ;   //not a correct value
                else {
                    outFile << j << "--" << h_fromGPU_hash[i * PointNum + j] <<" "<< h_fromGPU_F1[i * PointNum + j]
                            <<" "<< h_fromGPU_F2[i * PointNum + j] <<" "<< h_fromGPU_F3[i * PointNum + j] <<" "<< h_fromGPU_F4[i * PointNum + j] <<" ";
                }
            }
        }
        outFile << endl;
    }
    outFile.close();
    cout<<"cuda_wenjian over !"<<endl;*/
}


void SurfaceMatching::Voting()
{
    CudaVotingWithHash();
    int transformation_ID = 0;

    for (int i = 0; i < m_sceneWithNormals->size(); i++)
    {
        if (sceneLabelList[i] == 2)
        {
            for (int ii = 0; ii < m_modelWithNormals->size(); ii++)
                for (int jj = 0; jj < ppfExtractor.nAngle; jj++)
                {
                    accumSpace[ii][jj] = 0;
                }

            for (int j = 0; j < m_sceneWithNormals->size(); j++)
            {
                if (sceneLabelList[j] > 0 && (i != j))
                {
                    pcl::PointNormal  pn1 = m_sceneWithNormals->points[i];
                    pcl::PointNormal  pn2 = m_sceneWithNormals->points[j];
                    int hashIndex = ppfExtractor.CreateFeatureHashIndex(pn1, pn2);
                    if (hashIndex >= 0)
                    {
                        int F1 = ppfExtractor.F1;
                        int F2 = ppfExtractor.F2;
                        int F3 = ppfExtractor.F3;
                        int F4 = ppfExtractor.F4;

                        float alpha = ppfExtractor.CreateAngle2TouchXZPostivePlane(pn1, pn2);

                        vector<PPInfo>  ppInfoList = ppfModel[hashIndex];
                        for (int listID = 0; listID < ppInfoList.size(); listID++)
                        {
                            int cur_F1 = ppInfoList[listID].F1;
                            int cur_F2 = ppInfoList[listID].F2;
                            int cur_F3 = ppInfoList[listID].F3;
                            int cur_F4 = ppInfoList[listID].F4;
                            float cur_alpha = ppInfoList[listID].alpha;
                            int  pt_id = ppInfoList[listID].pt1_id;

                            int dis_thresh = 0;
                            int angle_thresh = 0;

                            if (isFeatureSimilar(dis_thresh, angle_thresh, F1, cur_F1, F2, cur_F2, F3, cur_F3, F4, cur_F4))
                            {
                                float alpha_m2s = cur_alpha - alpha;

                                if (alpha_m2s < -M_PI)
                                    alpha_m2s += M_PI * 2;
                                if (alpha_m2s > M_PI)
                                    alpha_m2s -= M_PI * 2;

                                int angleID = (alpha_m2s - ppfExtractor.Min_angle) / (ppfExtractor.Max_angle - ppfExtractor.Min_angle)*ppfExtractor.nAngle;

                                accumSpace[pt_id][angleID] += 1;
                            }
                        }
                    }
                }
            }
            int rowLen = m_modelWithNormals->size();
            int colLen = ppfExtractor.nAngle;
            int maxAccum = -1;
            int idy_max, idx_max;

            for (int idy = 0; idy < rowLen; idy++)
                for (int idx = 0; idx < colLen; idx++)
                {
                    int votingValue = accumSpace[idy][idx];
                    if (votingValue > maxAccum)
                    {
                        maxAccum = votingValue;
                        idy_max = idy;
                        idx_max = idx;
                    }

                }
            if(i<50)
                cout << "CPUMaxAccum:  " << maxAccum << endl;
            votingValueList.push_back(maxAccum);

            pcl::PointNormal  ps = m_sceneWithNormals->points[i];         //reference point
            pcl::PointNormal  pm = m_modelWithNormals->points[idy_max];   //model point
            float rot_angle = idx_max*ppfExtractor.d_angle + ppfExtractor.Min_angle;
            Matrix4f Trans_Mat = ppfExtractor.CreateTransformationFromModelToScene(pm, ps, rot_angle);
            //	cout << "Trans_mat: " << transformation_ID << "     VotingValue: " << maxAccum << endl;

            Eigen::Vector4f quaternion = ppfExtractor.RotationMatrixToQuaternion(Trans_Mat.block(0, 0, 3, 3));
            //	cout << "quaternion:  " << quaternion(0) << " " << quaternion(1) << " " << quaternion(2) << quaternion(3) << endl;
            vector<float>  curTransData;
            curTransData.push_back(quaternion(0));
            curTransData.push_back(quaternion(1));
            curTransData.push_back(quaternion(2));
            curTransData.push_back(quaternion(3));
            curTransData.push_back(Trans_Mat(0, 3));
            curTransData.push_back(Trans_Mat(1, 3));
            curTransData.push_back(Trans_Mat(2, 3));

            transformdataSet.push_back(curTransData);

            transformation_ID++;

        }///sceneLabelList[i] == 2
    }
}



bool SurfaceMatching::isFeatureSimilar(int dis_thresh, int angle_thresh, int F1, int cur_F1, int F2, int cur_F2, int F3, int cur_F3, int F4, int cur_F4)
{
    if (fabs(F1 - cur_F1) > dis_thresh || fabs(F2 - cur_F2) > angle_thresh || fabs(F3 - cur_F3) > angle_thresh || fabs(F4 - cur_F4) > angle_thresh)
        return false;

    return  true;

}
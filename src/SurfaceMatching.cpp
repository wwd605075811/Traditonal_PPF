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

void SurfaceMatching::init() {
    // int loadedTrainModel = this->LoadTrainModel("train_model.txt");
    int loadedTrainModel = 0;
    if (loadedTrainModel == 0) {
        this->Train();
    }
    this->InitAccumSpace();
}

SurfaceMatching::~SurfaceMatching() {
    if (accumSpace != NULL) {
        for (int i = 0; i < m_modelWithNormals->size(); i++) {
            if (accumSpace[i] != NULL) {
                delete[](accumSpace[i]);
                accumSpace[i] = NULL;
            }
        }
        delete[] accumSpace;
        accumSpace = NULL;
    }
    cout << "~over" << endl;
}

void SurfaceMatching::InitAccumSpace() {
    if (m_modelWithNormals == NULL) {
        std::cerr << "m_modelWithNormals is NULL in InitAccumSpace" << endl;
        return;
    }
    if (accumSpace != NULL) {
        std::cerr << "accumSpace is not NULL" << endl;
        return;
    }
    accumSpace = new int *[m_modelWithNormals->size()];
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
    //TODO: to remember Note 1 in README.md
    pcl::PointNormal minpt, maxpt;
    pcl::getMinMax3D(*m_modelWithNormals, minpt, maxpt);
    float diameter = 0;
    diameter = max(fabs(maxpt.z - minpt.z), max(fabs(maxpt.x - minpt.x), fabs(maxpt.y - minpt.y)));
    ppfExtractor.Set(tau_d * diameter, N_angle, int(1 / tau_d + 5), hashTableSize);
    //use vector<vector > to build hash table
    ppfModel.clear();
    for (int i = 0; i < hashTableSize; i++) {
        vector <PPInfo> hashItemList;
        hashItemList.clear();
        ppfModel.push_back(hashItemList);
    }
    CudaTrain();
}

void SurfaceMatching::CudaTrain() {

    int pointNum = 4096;      //TODO: to remember Note 2 in README.md
    CudaPPFInfo *g_F1 = (CudaPPFInfo *) malloc(sizeof(CudaPPFInfo) * pointNum);
    CudaPPFInfo *g_F1_copy = (CudaPPFInfo *) malloc(sizeof(CudaPPFInfo) * pointNum);
    for (int i = 0; i < pointNum; i++) {
        if (i >= m_modelWithNormals->size()) {
            g_F1[i].x = 0;
            g_F1[i].y = 0;
            g_F1[i].z = 0;
            g_F1[i].nomal_x = 0;
            g_F1[i].nomal_y = 0;
            g_F1[i].nomal_z = 0;
            g_F1_copy[i].x = 0;
            g_F1_copy[i].y = 0;
            g_F1_copy[i].z = 0;
            g_F1_copy[i].nomal_x = 0;
            g_F1_copy[i].nomal_y = 0;
            g_F1_copy[i].nomal_z = 0;
        } else {
            g_F1[i].x = m_modelWithNormals->points[i].x;
            g_F1[i].y = m_modelWithNormals->points[i].y;
            g_F1[i].z = m_modelWithNormals->points[i].z;
            g_F1[i].nomal_x = m_modelWithNormals->points[i].normal_x;
            g_F1[i].nomal_y = m_modelWithNormals->points[i].normal_y;
            g_F1[i].nomal_z = m_modelWithNormals->points[i].normal_z;
            g_F1_copy[i].x = m_modelWithNormals->points[i].x;
            g_F1_copy[i].y = m_modelWithNormals->points[i].y;
            g_F1_copy[i].z = m_modelWithNormals->points[i].z;
            g_F1_copy[i].nomal_x = m_modelWithNormals->points[i].normal_x;
            g_F1_copy[i].nomal_y = m_modelWithNormals->points[i].normal_y;
            g_F1_copy[i].nomal_z = m_modelWithNormals->points[i].normal_z;
        }
    }
    CudaOtherInfo h_other;
    h_other.g_P1 = ppfExtractor.P1;
    h_other.g_P2 = ppfExtractor.P2;
    h_other.g_P3 = ppfExtractor.P3;
    h_other.g_P4 = ppfExtractor.P4;
    h_other.Min_angle = ppfExtractor.Min_angle;
    h_other.Max_angle = ppfExtractor.Max_angle;
    h_other.nAngle = ppfExtractor.nAngle;
    h_other.d_distance = ppfExtractor.d_distance;
    h_other.max_hashIndex = ppfExtractor.max_hashIndex;
    h_other.hashTableSize = hashTableSize;

    float *h_alpha = (float *) malloc(sizeof(float) * pointNum * pointNum);
    int *h_hash = (int *) malloc(sizeof(int) * pointNum * pointNum);
    int *h_F1 = (int *) malloc(sizeof(int) * pointNum * pointNum);
    int *h_F2 = (int *) malloc(sizeof(int) * pointNum * pointNum);
    int *h_F3 = (int *) malloc(sizeof(int) * pointNum * pointNum);
    int *h_F4 = (int *) malloc(sizeof(int) * pointNum * pointNum);

    ModelCuda(g_F1, g_F1_copy, h_other, h_hash, h_alpha, h_F1, h_F2, h_F3, h_F4);
    cout << "GPU done!" << endl;

    //write the answer to the HashTable
    for (int i = 0; i < m_modelWithNormals->size(); ++i) {
        for (int j = 0; j < m_modelWithNormals->size(); ++j) {
            if (j == i) {
                continue;
            }
            if (h_hash[i * pointNum + j] >= 0) {
                PPInfo tempInfo;
                tempInfo.F1 = h_F1[i * pointNum + j];
                tempInfo.F2 = h_F2[i * pointNum + j];
                tempInfo.F3 = h_F3[i * pointNum + j];
                tempInfo.F4 = h_F4[i * pointNum + j];
                tempInfo.pt1_id = i;
                tempInfo.pt2_id = j;
                tempInfo.alpha = h_alpha[i * pointNum + j];
                ppfModel[h_hash[i * pointNum + j]].push_back(tempInfo);
            }
        }
    }
}

void SurfaceMatching::setScene(pcl::PointCloud<pcl::PointNormal>::Ptr sceneWithNormals) {
    m_sceneWithNormals = sceneWithNormals;
    this->pickReferencePoints();
}

void SurfaceMatching::pickReferencePoints() {
    sceneLabelList.clear(); //0:background 1:points 2:reference points
    for (int i = 0; i < m_sceneWithNormals->size(); i++) {
        sceneLabelList.push_back(1);
    }

    int step = 5;
    for (int i = 0; i < m_sceneWithNormals->size(); i++) {
        if (sceneLabelList[i] == 1 && i % step == 0) {
            sceneLabelList[i] = 2;
        }
    }
}

void SurfaceMatching::CudaVotingWithHash() {
    ///creat new hash table
    int hashPPFNum = m_modelWithNormals->size() * (m_modelWithNormals->size() - 1);
    PPInfo *modelHashValue = (PPInfo *) malloc(sizeof(PPInfo) * hashPPFNum);
    PPInfo *modelHashKey[hashTableSize + 1];
    int hashCount = 0;
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
    PPInfo *ptrEnd = &modelHashValue[hashCount];
    modelHashKey[hashTableSize] = ptrEnd;
    //calculate the real index in valueArray, because can not pass pointer to the kernel
    int *modelHashKeyIndex = (int *) malloc(sizeof(int) * hashTableSize);
    for (int k = 0; k < hashTableSize; ++k) {
        modelHashKeyIndex[k] = modelHashKey[k] - modelHashKey[0];
    }

    //m_sceneWithNormals->size =10091, and the reference point step is 5. 10091/5=2018,2018+1=2019, so the PointNum is bigger than 2019
    int pointReference = 2048;                  //TODO: to remember Note 2 in README.md
    int pointNum = m_sceneWithNormals->size();
    int count = 0;
    CudaPPFInfo *g_F1 = (CudaPPFInfo *) malloc(sizeof(CudaPPFInfo) * pointReference);
    CudaPPFInfo *g_F2 = (CudaPPFInfo *) malloc(sizeof(CudaPPFInfo) * pointNum);
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
    CudaOtherInfo h_other;
    h_other.g_P1 = ppfExtractor.P1;
    h_other.g_P2 = ppfExtractor.P2;
    h_other.g_P3 = ppfExtractor.P3;
    h_other.g_P4 = ppfExtractor.P4;
    h_other.Min_angle = ppfExtractor.Min_angle;
    h_other.Max_angle = ppfExtractor.Max_angle;
    h_other.nAngle = ppfExtractor.nAngle;
    h_other.d_distance = ppfExtractor.d_distance;
    h_other.max_hashIndex = ppfExtractor.max_hashIndex;
    h_other.hashTableSize = hashTableSize;
    h_other.hashNum = hashPPFNum;
    h_other.modelPointsNum = m_modelWithNormals->size();

    float *h_alpha = (float *) malloc(sizeof(float) * pointNum * pointReference);
    int *h_hash = (int *) malloc(sizeof(int) * pointNum * pointReference);
    int *h_F1 = (int *) malloc(sizeof(int) * pointNum * pointReference);
    int *h_F2 = (int *) malloc(sizeof(int) * pointNum * pointReference);
    int *h_F3 = (int *) malloc(sizeof(int) * pointNum * pointReference);
    int *h_F4 = (int *) malloc(sizeof(int) * pointNum * pointReference);
    int *h_accumSpace = (int *) malloc(sizeof(int) * m_modelWithNormals->size() * pointReference);

    Reference2NumCuda(g_F1, g_F2, h_other, h_hash, h_alpha, h_F1, h_F2, h_F3,
                      h_F4, modelHashValue, modelHashKeyIndex, h_accumSpace);

    cout << "GPU Voting done!" << endl;
    //printf file to check the value
    /*ofstream outFile("scene_ppf_cuda.txt");
    for (int i = 0; i < 2019; i++) {
        outFile << i * 5 << "****************************" << endl;
        for (int j = 0; j < m_sceneWithNormals->size(); j++) {
            if ((i * 5) != j) {
                if (h_fromGPU_F1[i * PointNum + j] > ppfExtractor.max_nDistance);   //not a correct value
                else {
                    outFile << j << "--" << h_fromGPU_hash[i * PointNum + j] << " " << h_fromGPU_F1[i * PointNum + j]
                            << " " << h_fromGPU_F2[i * PointNum + j] << " " << h_fromGPU_F3[i * PointNum + j]
                            << " " << h_fromGPU_F4[i * PointNum + j] << " ";
                }
            }
        }
        outFile << endl;
    }
    outFile.close();
    cout << "cuda_file has written !" << endl;*/
}

void SurfaceMatching::Voting() {
    CudaVotingWithHash();
    cout << "cuda is over!" << endl;

    int transformation_ID = 0;
    for (int i = 0; i < m_sceneWithNormals->size(); i++) {
        if (sceneLabelList[i] == 2) {
            for (int ii = 0; ii < m_modelWithNormals->size(); ii++)
                for (int jj = 0; jj < ppfExtractor.nAngle; jj++) {
                    accumSpace[ii][jj] = 0;
                }
            for (int j = 0; j < m_sceneWithNormals->size(); j++) {
                if (sceneLabelList[j] > 0 && (i != j)) {

                    pcl::PointNormal pn1 = m_sceneWithNormals->points[i];
                    pcl::PointNormal pn2 = m_sceneWithNormals->points[j];
                    int hashIndex = ppfExtractor.CreateFeatureHashIndex(pn1, pn2);
                    if (hashIndex >= 0) {
                        int F1 = ppfExtractor.F1;
                        int F2 = ppfExtractor.F2;
                        int F3 = ppfExtractor.F3;
                        int F4 = ppfExtractor.F4;
                        float alpha = ppfExtractor.CreateAngle2TouchXZPostivePlane(pn1, pn2);

                        vector <PPInfo> ppInfoList = ppfModel[hashIndex];
                        for (int listID = 0; listID < ppInfoList.size(); listID++) {
                            int cur_F1 = ppInfoList[listID].F1;
                            int cur_F2 = ppInfoList[listID].F2;
                            int cur_F3 = ppInfoList[listID].F3;
                            int cur_F4 = ppInfoList[listID].F4;
                            float cur_alpha = ppInfoList[listID].alpha;
                            int pt_id = ppInfoList[listID].pt1_id;

                            int dis_thresh = 0;
                            int angle_thresh = 0;

                            if (isFeatureSimilar(dis_thresh, angle_thresh, F1, cur_F1, F2, cur_F2, F3, cur_F3, F4,
                                                 cur_F4)) {
                                float alpha_m2s = cur_alpha - alpha;

                                if (alpha_m2s < -M_PI)
                                    alpha_m2s += M_PI * 2;
                                if (alpha_m2s > M_PI)
                                    alpha_m2s -= M_PI * 2;

                                int angleID = (alpha_m2s - ppfExtractor.Min_angle) /
                                              (ppfExtractor.Max_angle - ppfExtractor.Min_angle) * ppfExtractor.nAngle;
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

            for (int idy = 0; idy < rowLen; idy++) {
                for (int idx = 0; idx < colLen; idx++) {
                    int votingValue = accumSpace[idy][idx];
                    if (votingValue > maxAccum) {
                        maxAccum = votingValue;
                        idy_max = idy;
                        idx_max = idx;
                    }
                }
            }
            // cout << "i:" << i <<" idy_max:"<<idy_max<< endl;
            // cout << "   MaxAccum:  " << maxAccum << endl;
            votingValueList.push_back(maxAccum);

            pcl::PointNormal ps = m_sceneWithNormals->points[i];
            pcl::PointNormal pm = m_modelWithNormals->points[idy_max];
            float rot_angle = idx_max * ppfExtractor.d_angle + ppfExtractor.Min_angle;
            Matrix4f Trans_Mat = ppfExtractor.CreateTransformationFromModelToScene(pm, ps, rot_angle);
            //	cout << "Trans_mat: " << transformation_ID << "     VotingValue: " << maxAccum << endl;

            Eigen::Vector4f quaternion = ppfExtractor.RotationMatrixToQuaternion(Trans_Mat.block(0, 0, 3, 3));
            //	cout << "quaternion:  " << quaternion(0) << " " << quaternion(1) << " " << quaternion(2) << quaternion(3) << endl;
            vector<float> curTransData;
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

bool SurfaceMatching::isFeatureSimilar(int dis_thresh, int angle_thresh, int F1, int cur_F1, int F2,
                                        int cur_F2, int F3, int cur_F3, int F4, int cur_F4)
{
    if (fabs(F1 - cur_F1) > dis_thresh || fabs(F2 - cur_F2) > angle_thresh || fabs(F3 - cur_F3) > angle_thresh || fabs(F4 - cur_F4) > angle_thresh)
        return false;

    return  true;
}
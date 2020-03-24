#include <sys/time.h>
#include "../include/SurfaceMatching.h"
bool transformCompare(TransformWithProb trans1, TransformWithProb trans2){
    return trans1.prob> trans2.prob;
}

SurfaceMatching::SurfaceMatching(pcl::PointCloud<pcl::PointNormal>::Ptr modelWithNormals) {
    this->m_modelWithNormals = modelWithNormals;
}

SurfaceMatching::~SurfaceMatching() {
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

void SurfaceMatching::newVoting(Model *model) {
    cout << "GPU Voting done!" << endl;
    //add votes result into transformdataSet
    int transformation_ID = 0;
    for (int i = 0; i < (m_sceneWithNormals->size()); i++) {
        if (sceneLabelList[i] == 2) {
            pcl::PointNormal ps = m_sceneWithNormals->points[i];
            pcl::PointNormal pm = m_modelWithNormals->points[(model->mPoint)[i/5]];
            float rot_angle = (model->mAngle)[i/5] * ppfExtractor.d_angle + ppfExtractor.Min_angle;

            Matrix4f Trans_Mat = ppfExtractor.CreateTransformationFromModelToScene(pm, ps, rot_angle);
            Eigen::Vector4f quaternion = ppfExtractor.RotationMatrixToQuaternion(Trans_Mat.block(0, 0, 3, 3));

            ///this nan is from the scene.file, which is because of computing the normals
            if(isnan(quaternion(0)) || isnan(quaternion(1)) || isnan(quaternion(2)) || isnan(quaternion(3))
                || isnan(Trans_Mat(0, 3)) || isnan(Trans_Mat(1, 3)) || isnan(Trans_Mat(2, 3)))
                continue;

            votingValueList.push_back((model->mNumber)[i/5]);  //step=5
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
        }
    }
}

void SurfaceMatching::CreateTranformtion_HCluster(float angle_thresh, float dis_thresh)
{
    HCluster  cluster;
    cluster.SetThresholds(angle_thresh, dis_thresh);
    cluster.CreateDataSet(transformdataSet, votingValueList);//从内存加载
    cluster.CreateCluster();

    vector< vector<float> >  centroidList;
    vector< int >  clusterLabelList;
    cluster.GetCentroidsAndClusterLabels(centroidList, clusterLabelList);

    int K = centroidList.size();
    vector< int > totalVotingListForKCluster(K,0);
    for (int i = 0; i < clusterLabelList.size(); i++)
    {
        int clusterNo = clusterLabelList[i];
        totalVotingListForKCluster[clusterNo] += votingValueList[i];
    }

    ofstream fout;
    fout.open("transformationList.txt");
    if (!fout)
    {
        cout << "file transformationList.txt open failed" << endl;
        exit(0);
    }

    float sumVoting = 0;
    for (int i = 0; i < K; i++)
    {
        sumVoting += totalVotingListForKCluster[i];
    }

    int maxVotingNo = -1;
    int maxVoting = -1;
    for (int i = 0; i < K; i++)
    {
        if (maxVoting < totalVotingListForKCluster[i])
        {
            maxVoting = totalVotingListForKCluster[i];
            maxVotingNo = i;
        }

        vector<float>  centrod = centroidList[i];
        fout << totalVotingListForKCluster[i] << "  " << centrod[0] << " " << centrod[1] << "  " << centrod[2] << " "
        << centrod[3] << " " << centrod[4] << " " << centrod[5] << " " << centrod[6] << " " << endl;

        Vector4f  quaterniondVec;
        quaterniondVec[0] = centrod[0];
        quaterniondVec[1] = centrod[1];
        quaterniondVec[2] = centrod[2];
        quaterniondVec[3] = centrod[3];

        Vector3f  translationVec;
        translationVec[0] = centrod[4];
        translationVec[1] = centrod[5];
        translationVec[2] = centrod[6];

        Matrix3f  rot_mat = ppfExtractor.QuaternionToRotationMatrix(quaterniondVec);
        Eigen::Matrix4f transform_mat = ppfExtractor.RotationAndTranslation2Transformation(rot_mat, translationVec);

        float prob = totalVotingListForKCluster[i] / sumVoting;
        TransformWithProb trans;
        trans.prob = prob;
        trans.transform_mat = transform_mat;
        transformMatList.push_back(trans);
    }
    std::sort(transformMatList.begin(), transformMatList.end(), transformCompare);

    cout << "Best:  " << transformMatList[BEST_TRANS_ID].transform_mat << endl;
    cout << "Probability:    " << transformMatList[BEST_TRANS_ID].prob << endl;
    cout << endl;
}

TransformWithProbVector SurfaceMatching::getTransforms(){
    return  transformMatList;
}

pcl::PointCloud<pcl::PointNormal>::Ptr SurfaceMatching::getBestResult()
{
    pcl::PointCloud<pcl::PointNormal>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointNormal>());
    pcl::transformPointCloud (*m_modelWithNormals, *transformed_cloud, transformMatList[BEST_TRANS_ID].transform_mat);
    return transformed_cloud;
}

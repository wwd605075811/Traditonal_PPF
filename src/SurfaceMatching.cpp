#include <sys/time.h>
#include "../include/SurfaceMatching.h"
bool transformScoreCompare(TransformWithProb trans1, TransformWithProb trans2){
    return trans1.prob > trans2.prob;
}

bool resultScoreCompare(ResultWithScore t1, ResultWithScore t2){
    return t1.score > t2.score;
}

SurfaceMatching::SurfaceMatching(pcl::PointCloud<pcl::PointNormal>::Ptr modelWithNormals,
        pcl::PointCloud<pcl::PointNormal>::Ptr sceneWithNormals, int step, Model *model,
        float clusterAngleThresh, float clusterDisThresh) {
    this->m_modelWithNormals_ = modelWithNormals;
    this->setScene(sceneWithNormals, step);
    this->dealVoteResult(model);
    this->createTranformtionHCluster(clusterAngleThresh, clusterDisThresh);
}

SurfaceMatching::~SurfaceMatching() {
}

void SurfaceMatching::setScene(pcl::PointCloud<pcl::PointNormal>::Ptr sceneWithNormals, int step) {
    m_sceneWithNormals_ = sceneWithNormals;
    this->sceneStep_ = step;
    this->pickReferencePoints();
}

void SurfaceMatching::pickReferencePoints() {
    sceneLabelList_.clear(); //0:background 1:points 2:reference points
    for (int i = 0; i < m_sceneWithNormals_->size(); i++) {
        sceneLabelList_.push_back(1);
    }
    for (int i = 0; i < m_sceneWithNormals_->size(); i++) {
        if (sceneLabelList_[i] == 1 && i % this->sceneStep_ == 0) {
            sceneLabelList_[i] = 2;
        }
    }
}

void SurfaceMatching::dealVoteResult(Model *model) {
    cout << "GPU Voting done!" << endl;
    //add votes result into transformdataSet_
    int transformation_ID = 0;
    for (int i = 0; i < (m_sceneWithNormals_->size()); i++) {
        if (sceneLabelList_[i] == 2) {
            pcl::PointNormal ps = m_sceneWithNormals_->points[i];
            pcl::PointNormal pm = m_modelWithNormals_->points[(model->mPoint)[i/5]];
            float rot_angle = (model->mAngle)[i/5] * ppfExtractor_.d_angle
                               + ppfExtractor_.Min_angle;

            Matrix4f Trans_Mat = ppfExtractor_.CreateTransformationFromModelToScene
                    (pm, ps, rot_angle);
            Eigen::Vector4f quaternion = ppfExtractor_.RotationMatrixToQuaternion
                    (Trans_Mat.block(0, 0, 3, 3));

            ///this nan is from the scene.file, which is because of computing the normals
            if(isnan(quaternion(0)) || isnan(quaternion(1)) || isnan(quaternion(2)) || isnan(quaternion(3))
                || isnan(Trans_Mat(0, 3)) || isnan(Trans_Mat(1, 3)) || isnan(Trans_Mat(2, 3))){
                continue;
            }

            votingValueList_.push_back((model->mNumber)[i/5]);  //step=5
            vector<float> curTransData;
            curTransData.push_back(quaternion(0));
            curTransData.push_back(quaternion(1));
            curTransData.push_back(quaternion(2));
            curTransData.push_back(quaternion(3));
            curTransData.push_back(Trans_Mat(0, 3));
            curTransData.push_back(Trans_Mat(1, 3));
            curTransData.push_back(Trans_Mat(2, 3));

            transformdataSet_.push_back(curTransData);
            transformation_ID++;
        }
    }
}

void SurfaceMatching::createTranformtionHCluster(float angle_thresh, float dis_thresh) {

    HCluster  cluster;
    cluster.SetThresholds(angle_thresh, dis_thresh);
    cluster.CreateDataSet(transformdataSet_, votingValueList_);//从内存加载
    cluster.CreateCluster();

    vector< vector<float> >  centroidList;
    vector< int >  clusterLabelList;
    cluster.GetCentroidsAndClusterLabels(centroidList, clusterLabelList);

    int K = centroidList.size();
    vector< int > totalVotingListForKCluster(K,0);
    for (int i = 0; i < clusterLabelList.size(); i++) {
        int clusterNo = clusterLabelList[i];
        totalVotingListForKCluster[clusterNo] += votingValueList_[i];
    }

    float sumVoting = 0;
    for (int i = 0; i < K; i++) {
        sumVoting += totalVotingListForKCluster[i];
    }

    int maxVotingNo = -1;
    int maxVoting = -1;
    for (int i = 0; i < K; i++) {
        if (maxVoting < totalVotingListForKCluster[i]) {
            maxVoting = totalVotingListForKCluster[i];
            maxVotingNo = i;
        }
        vector<float>  centrod = centroidList[i];

        Vector4f  quaterniondVec;
        quaterniondVec[0] = centrod[0];
        quaterniondVec[1] = centrod[1];
        quaterniondVec[2] = centrod[2];
        quaterniondVec[3] = centrod[3];

        Vector3f  translationVec;
        translationVec[0] = centrod[4];
        translationVec[1] = centrod[5];
        translationVec[2] = centrod[6];

        Matrix3f  rot_mat = ppfExtractor_.QuaternionToRotationMatrix(quaterniondVec);
        Eigen::Matrix4f transform_mat = ppfExtractor_.RotationAndTranslation2Transformation(rot_mat, translationVec);

        float prob = totalVotingListForKCluster[i] / sumVoting;
        TransformWithProb trans;
        trans.prob = prob;
        trans.transform_mat = transform_mat;
        transformMatList_.push_back(trans);
    }

    std::sort(transformMatList_.begin(), transformMatList_.end(), transformScoreCompare);

    /*cout << "Best:  " << transformMatList_[BEST_TRANS_ID].transform_mat << endl;
    cout << "Probability:    " << transformMatList_[BEST_TRANS_ID].prob << endl;
    cout << endl;*/
}

TransformWithProbVector SurfaceMatching::getTransforms(){
    return  transformMatList_;
}

pcl::PointCloud<pcl::PointNormal>::Ptr SurfaceMatching::getBestResult(){
    pcl::PointCloud<pcl::PointNormal>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointNormal>());
    pcl::transformPointCloud (*m_modelWithNormals_, *transformed_cloud,
            transformMatList_[BEST_TRANS_ID].transform_mat);
    return transformed_cloud;
}

void SurfaceMatching::icpAndGetFitnessScore() {
    PointCloudNT::Ptr bestResult = this->getBestResult();
    TransformWithProbVector transforms = this->getTransforms();
    vector<ResultWithScore> R;

    int numberOfGoodPose = 0;
    for (TransformWithProb &transform : transforms) {
        ///use ICP to align the scene
        Eigen::Matrix4f transform_mat = transform.transform_mat;
        if(numberOfGoodPose == 0){
            cout << "before icp:"<< endl << transform_mat << endl;
        }
        PointCloudNT::Ptr result_cloud(new PointCloudNT());
        ///just change the pos, didn't change the normal
        pcl::transformPointCloud(*(this->m_modelWithNormals_), *result_cloud, transform_mat);
        PointCloudNT::Ptr input_cloud = result_cloud;
        pcl::IterativeClosestPoint<PointNT, PointNT> icp;
        icp.setInputSource(input_cloud);
        icp.setInputTarget(this->m_sceneWithNormals_);
        icp.setMaxCorrespondenceDistance(20);
        PointCloudNT Final;
        icp.align(Final);
        PointCloudNT::Ptr result_cloud_icp = Final.makeShared();
        float maxDistance=icp.getMaxCorrespondenceDistance ();
        Eigen::Matrix4f transform_mat_icp = icp.getFinalTransformation();
        Eigen::Matrix4f afterICP;
        afterICP = transform_mat_icp * transform_mat;
        if(numberOfGoodPose == 0){
            cout << "max distance :"<< maxDistance <<endl;
            cout<< "after icp:" <<endl<<afterICP<<endl;
        }
        ///use the end pose to segmentation scene to get real scene
        float radius = 10.0;
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;
        std::vector<int> IdxRadius;
        std::vector<float> RadiusDistance;
        PointCloudNT::Ptr realScene(new PointCloudNT());
        for (int i = 0; i < result_cloud_icp->points.size(); ++i) {
            icp.getSearchMethodTarget()->radiusSearch(result_cloud_icp->points[i],
                                                      radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
            if(!pointIdxRadiusSearch.empty()){
                for (int j = 0; j < pointIdxRadiusSearch.size (); ++j){
                    IdxRadius.push_back(pointIdxRadiusSearch[j]);
                }
            }
        }
        if(numberOfGoodPose == 0){
            cout<< "scene size :" <<IdxRadius.size() << endl;
        }
        sort(IdxRadius.begin(), IdxRadius.end());
        IdxRadius.erase(unique(IdxRadius.begin(), IdxRadius.end()), IdxRadius.end());
        for (int j = 0; j < IdxRadius.size(); ++j) {
            realScene->push_back(this->m_sceneWithNormals_->points[IdxRadius[j]]);
        }

        if(numberOfGoodPose == 0){
            cout<< "realScene size:" << realScene->points.size() <<endl;
        }

        ///To check each point in realScene : which point could match the end pose, and give the score
        pcl::KdTreeFLANN<pcl::PointNormal> kdtree;
        kdtree.setInputCloud (result_cloud_icp);
        float max_range = 3.5;
        std::vector<int> nn_indices (1);
        std::vector<float> nn_dists (1);
        std::vector<int> goodIndices ;
        std::vector<float> goodDists ;
        std::vector<int> succeed;
        int goodPoint = 0;
        int badPoint = 0;
        for (int i = 0; i < realScene->points.size(); ++i) {
            kdtree.nearestKSearch(realScene->points[i], 1,
                                  nn_indices, nn_dists);

            if (nn_dists[0] <= max_range){
                goodPoint ++;
                goodIndices.push_back(nn_indices[0]);
                succeed.push_back(i);
            } else {
                badPoint ++;
            }
        }
        if(numberOfGoodPose == 0){
            cout<< "goodPoint number:"<<goodPoint << "badPoint number:" <<badPoint <<endl;
        }
        PointCloudNT::Ptr alignedPointFromSceneToModel(new PointCloudNT());
        for (int j = 0; j < succeed.size(); ++j) {
            alignedPointFromSceneToModel->push_back(realScene->points[succeed[j]]);
        }

        ///Save the first 10 results in the vector
        ResultWithScore r;
        r.score = float(goodPoint) / float(realScene->points.size()) *100;
        r.resultCloud = result_cloud_icp;
        r.realScene = realScene;
        r.alignedPointFromSceneToModel = alignedPointFromSceneToModel;
        r.id = numberOfGoodPose;
        numberOfGoodPose ++;
        R.push_back(r);
        if(!(numberOfGoodPose < 6)){
            break;
        }
    }

    ///sort and Visualization
    std::sort(R.begin(),R.end(),resultScoreCompare);
    for( int i=0; i < 6; i++){
        cout<< " id:" <<R[i].id <<" score:"<< R[i].score <<endl;
    }
    for (int i = 0; i < 6; ++i) {
        pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer());
        ColorHandlerT red(this->m_modelWithNormals_, 255, 0, 0);
        viewer->addPointCloud<PointNT>(this->m_modelWithNormals_, red, "model_downsampled");
        viewer->setPointCloudRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                5, "model_downsampled");
        viewer->addPointCloudNormals<PointNT, PointNT>(
                this->m_modelWithNormals_,
                this->m_modelWithNormals_,
                1, 1,
                "model_downsampled_normals");

        ColorHandlerT yellow(this->m_sceneWithNormals_, 255, 255, 0);
        viewer->addPointCloud<PointNT>(this->m_sceneWithNormals_, yellow, "scene_downsampled");
        viewer->setPointCloudRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                4, "scene_downsampled");
        viewer->addPointCloudNormals<PointNT, PointNT>(
                this->m_sceneWithNormals_,
                this->m_sceneWithNormals_,
                1, 1,
                "scene_downsampled_normals");

        ColorHandlerT green(R[i].resultCloud, 0, 255, 0);
        viewer->addPointCloud<PointNT>(R[i].resultCloud, green, "result_cloud_icp");
        viewer->setPointCloudRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                4, "result_cloud_icp");

        ColorHandlerT blue(R[i].alignedPointFromSceneToModel, 0, 0, 255);
        viewer->addPointCloud<PointNT>(R[i].alignedPointFromSceneToModel, blue, "alignedPointFromSceneToModel");
        viewer->setPointCloudRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                4, "alignedPointFromSceneToModel");

        while (!viewer->wasStopped()) {
            viewer->spinOnce();
        }
        viewer->resetStoppedFlag();
        viewer->close();
    }
}

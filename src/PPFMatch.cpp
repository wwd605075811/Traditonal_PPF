#include <sys/time.h>
#include "../include/PPFMatch.h"
bool transformScoreCompare(TransformWithProb trans1, TransformWithProb trans2){
    return trans1.prob > trans2.prob;
}

bool quaternionScoreCompare(QuaternionWithVote trans1, QuaternionWithVote trans2){
    return trans1.votes > trans2.votes;
}


bool resultScoreCompare(ResultWithScore t1, ResultWithScore t2){
    return t1.score > t2.score;
}

float diffTransmat(Eigen::Matrix4f t1, Eigen::Matrix4f t2) {
    Vector3f pos1;
    pos1[0] = t1(0, 3);
    pos1[1] = t1(1, 3);
    pos1[2] = t1(2, 3);
    Vector3f pos2;
    pos2[0] = t2(0, 3);
    pos2[1] = t2(1, 3);
    pos2[2] = t2(2, 3);

    float distance;
    distance = (pos2 -pos1).norm();
    return distance;
}

PPFMatch::PPFMatch(pcl::PointCloud<pcl::PointNormal>::Ptr modelWithNormals,
             float tau_d, int modelStep) {
    this->setModel(modelWithNormals, modelStep, tau_d);
}

TransformWithProbVector PPFMatch::match(pcl::PointCloud<pcl::PointNormal>::Ptr sceneWithNormals, int sceneStep,
                     float clusterAngleThresh, float clusterDisThresh,
                     int resultNumber, float icpMaxCorrespondenceDistance,
                     float scoreSearchRadius, float scoreThresh) {

    this->setScene(sceneWithNormals, sceneStep, clusterAngleThresh, clusterDisThresh);
    //PPF计算初始化，f1Dispersion对于F1值(ppf 四个值)的离散化(我的理解)，在OPENCV中也有类似输入值(tau_d)
    Model *model = new Model(this->m_modelWithNormals_.get(), this->m_sceneWithNormals_.get(),
                             this->f1Dispersion_, this->sceneStep_, this->modelStep_);
    this->dealVoteResult(model);
    this->createTranformtionHCluster(this->clusterAngleThresh_, this->clusterDisThresh_);
    this->setICP(resultNumber, icpMaxCorrespondenceDistance, scoreSearchRadius, scoreThresh);
    //this function is to check the result before ICP
    this->withoutICP();

    ResultWithScoreVector allResultInfo = this->icpAndGetFitnessScore();

    TransformWithProbVector matrixAndScoreList_;
    TransformWithProb matrixAndScore;
    for(int i = 0; i < this->resultNumber_ ; i++) {
        matrixAndScore.transform_mat = allResultInfo[i].rotationMatrix;
        matrixAndScore.prob = allResultInfo[i].score;
        matrixAndScoreList_.push_back(matrixAndScore);
    }
    return matrixAndScoreList_;
}

void PPFMatch::setModel(pcl::PointCloud<pcl::PointNormal>::Ptr modelWithNormals, int modelStep, float tau_d) {
    this->m_modelWithNormals_ = modelWithNormals;
    this->modelStep_ = modelStep;
    this->tau_d_ = tau_d;
    pcl::PointNormal minpt, maxpt;
    pcl::getMinMax3D(*(this->m_modelWithNormals_), minpt, maxpt);
    float diameter = 0;
    diameter = max(fabs(maxpt.z - minpt.z), max(fabs(maxpt.x - minpt.x), fabs(maxpt.y - minpt.y)));
    cout << "diameter:" << diameter << " tau_d:" <<tau_d <<endl;
    this->f1Dispersion_ = diameter * this->tau_d_;
}

void PPFMatch::setScene(pcl::PointCloud<pcl::PointNormal>::Ptr sceneWithNormals, int sceneStep,
                        float clusterAngleThresh, float clusterDisThresh) {
    this->m_sceneWithNormals_ = sceneWithNormals;
    this->sceneStep_ = sceneStep;
    this->clusterAngleThresh_ = clusterAngleThresh;
    this->clusterDisThresh_ = clusterDisThresh;
}

void PPFMatch::setICP(int resultNumber, float icpMaxCorrespondenceDistance,
                      float scoreSearchRadius, float scoreThresh) {
    this->resultNumber_ = resultNumber;
    this->icpMaxCorrespondenceDistance_ = icpMaxCorrespondenceDistance;
    this->scoreSearchRadius_ = scoreSearchRadius;
    this->scoreThresh_ = scoreThresh;
}

void PPFMatch::dealVoteResult(Model *model) {
    /*
    *把PPF结束后返回的结果(n个可能位姿+票数)进行处理。把旋转角度还原为旋转矩阵和四元数，
    *用来聚类和ICP，注意在计算旋转矩阵时使用的Goldman公式，因为旋转轴为坐标轴，所以成立
    *(此处旋转矩阵计算方式和CUDA计算方式不同，但结果一样)
    */
    cout << "GPU Voting done!" << endl;
    TransformWithProbVector afterPPFTransformatList; 
    TransformWithProb afterPPFTransformat;

    vector<QuaternionWithVote> quaternionWithVoteList;
    QuaternionWithVote quaternionWithVote;
    //add votes result into transformdataSet_

    for (int i = 0; i < model->refNum; i++) { //对每一个pose，其实不应该refNum，因为选取了refNum个pose
        pcl::PointNormal ps = m_sceneWithNormals_->points[model->sPoint[i]];
        pcl::PointNormal pm = m_modelWithNormals_->points[model->mPoint[i]];
        float rot_angle = (model->mAngle)[i] * ppfExtractor_.d_angle
                          + ppfExtractor_.Min_angle;

        Matrix4f Trans_Mat = ppfExtractor_.CreateTransformationFromModelToScene
                (pm, ps, rot_angle);
        //这个vector是收集所有pose的旋转矩阵，为了看一下PPF结束后不聚类的匹配情况
        afterPPFTransformat.transform_mat = Trans_Mat;
        afterPPFTransformat.prob = float((model->mNumber)[i]);
        afterPPFTransformatList.push_back(afterPPFTransformat);

        Eigen::Vector4f quaternion = ppfExtractor_.RotationMatrixToQuaternion
                (Trans_Mat.block(0, 0, 3, 3));
        //this nan is from the scene.file, which is because of computing the normals，
        //在根源上应该把算过法向量后的scene中的nan点都去掉
        if (isnan(quaternion(0)) || isnan(quaternion(1)) || isnan(quaternion(2)) || isnan(quaternion(3))
            || isnan(Trans_Mat(0, 3)) || isnan(Trans_Mat(1, 3)) || isnan(Trans_Mat(2, 3))) {
            continue;
        }
        votingValueList_.push_back((model->mNumber)[i]);
        vector<float> curTransData;
        curTransData.push_back(quaternion(0));
        curTransData.push_back(quaternion(1));
        curTransData.push_back(quaternion(2));
        curTransData.push_back(quaternion(3));
        curTransData.push_back(Trans_Mat(0, 3));
        curTransData.push_back(Trans_Mat(1, 3));
        curTransData.push_back(Trans_Mat(2, 3));
        transformdataSet_.push_back(curTransData);

        quaternionWithVote.votes = float((model->mNumber)[i]);
        quaternionWithVote.quaternion7 = curTransData;
        quaternionWithVoteList.push_back(quaternionWithVote);
    }

    cout << "watch and PCLcluster in distance before PPF " <<endl;
    //this->watchResultAfterPPF(this->NMSClustering(quaternionWithVoteList));
}

void PPFMatch::watchResultAfterPPF(TransformWithProbVector afterPPFTransformatList) {

    std::sort(afterPPFTransformatList.begin(), afterPPFTransformatList.end(), transformScoreCompare);
    cout << "the size of PPF List is:" <<afterPPFTransformatList.size() <<endl;
    for (int i = 0; i<10; i++) {
        cout << "prob:" <<afterPPFTransformatList[i].prob <<endl;
        Eigen::Matrix4f transform_mat = afterPPFTransformatList[i].transform_mat;

        PointCloudNT::Ptr result_cloud(new PointCloudNT());
        pcl::transformPointCloud(*(this->m_modelWithNormals_), *result_cloud, transform_mat);
        PointCloudNT::Ptr resultCloud = result_cloud;

        pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer());
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

        ColorHandlerT green(resultCloud, 0, 255, 0);
        viewer->addPointCloud<PointNT>(resultCloud, green, "result_cloud_icp");
        viewer->setPointCloudRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                4, "result_cloud_icp");
        viewer->addPointCloudNormals<PointNT, PointNT>(
                resultCloud,
                resultCloud,
                1, 1,
                "afterPPF[i].resultCloud");

        while (!viewer->wasStopped()) {
            viewer->spinOnce();
        }
        viewer->resetStoppedFlag();
        viewer->close();
    }
    cout <<"watch the NMSCluster over" <<endl;
}

void PPFMatch::createTranformtionHCluster(float angle_thresh, float dis_thresh) {

    HCluster cluster;
    cluster.SetThresholds(angle_thresh, dis_thresh);
    cluster.CreateDataSet(transformdataSet_, votingValueList_);//从内存加载
    cluster.CreateCluster();

    vector<vector<float>>  centroidList;
    vector<int>  clusterLabelList;
    cluster.GetCentroidsAndClusterLabels(centroidList, clusterLabelList);

    int K = centroidList.size();
    vector<int> totalVotingListForKCluster(K,0);
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

        float prob = totalVotingListForKCluster[i];
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

TransformWithProbVector PPFMatch::getTransforms(){
    return  transformMatList_;
}

pcl::PointCloud<pcl::PointNormal>::Ptr PPFMatch::getBestResult(){
    pcl::PointCloud<pcl::PointNormal>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointNormal>());
    pcl::transformPointCloud (*m_modelWithNormals_, *transformed_cloud,
            transformMatList_[BEST_TRANS_ID].transform_mat);
    return transformed_cloud;
}

ResultWithScoreVector PPFMatch::icpAndGetFitnessScore() {
    PointCloudNT::Ptr bestResult = this->getBestResult();
    TransformWithProbVector transforms = this->getTransforms();
    ResultWithScoreVector R;

    int numberOfGoodPose = 0;
    for (TransformWithProb &transform : transforms) {
        ///use ICP to align the scene
        Eigen::Matrix4f transform_mat = transform.transform_mat;
        int prob = transform.prob;
        if(numberOfGoodPose == 0){
            cout << "before icp(id:"<<numberOfGoodPose<<"):" << endl << transform_mat << endl;
        }
        PointCloudNT::Ptr result_cloud(new PointCloudNT());
        //just change the pos, didn't change the normal
        pcl::transformPointCloud(*(this->m_modelWithNormals_), *result_cloud, transform_mat);
        PointCloudNT::Ptr input_cloud = result_cloud;

        pcl::IterativeClosestPoint<PointNT, PointNT> icp;
        icp.setInputSource(input_cloud);
        icp.setInputTarget(this->m_sceneWithNormals_);
        icp.setMaxCorrespondenceDistance(this->icpMaxCorrespondenceDistance_);
        PointCloudNT Final;
        icp.align(Final);
        PointCloudNT::Ptr result_cloud_icp = Final.makeShared();
        float maxDistance=icp.getMaxCorrespondenceDistance ();
        Eigen::Matrix4f transform_mat_icp = icp.getFinalTransformation();
        Eigen::Matrix4f afterICPTransform;
        afterICPTransform = transform_mat_icp * transform_mat;
        if(numberOfGoodPose == 0){
            cout << "max distance :"<< maxDistance <<endl;
            cout<< "afterICPTransform(id:"<<numberOfGoodPose<<"):" <<endl<<afterICPTransform<<endl;
        }

        /*
        * 先拿ICP的结果当做输入，找scene周围的点来确定real scene(分母)
        * 在拿real scene当做输入，找model周围的点来确定有多少点匹配上了(分子)
        * 用以上两个结果就可以计算出评分，注意两个阈值的设置，第一个阈值要设置的比第二个大一点
        * 第一个阈值不能设置的过大，不然就会把场景中需要匹配的零件之外的点也算进来，分母变大，分数降低。
        */
       //这里直接用icp对象，因为他们是继承关系，也就是InputCloud(scene)，也可以用下面的KDtreeFlann
        float radius = this->scoreSearchRadius_;
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

        sort(IdxRadius.begin(), IdxRadius.end());
        IdxRadius.erase(unique(IdxRadius.begin(), IdxRadius.end()), IdxRadius.end());
        for (int j = 0; j < IdxRadius.size(); ++j) {
            realScene->push_back(this->m_sceneWithNormals_->points[IdxRadius[j]]);
        }

        if(numberOfGoodPose == 0){
            cout<< "realScene size(id:"<<numberOfGoodPose<< "):" << realScene->points.size() <<endl;
        }

        ///To check each point in realScene : which point could match the end pose, and give the score
        pcl::KdTreeFLANN<pcl::PointNormal> kdtree;
        kdtree.setInputCloud (result_cloud_icp);
        float max_range = this->scoreThresh_;
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
            cout<< "id:0 goodPoint number:"<<goodPoint << "badPoint number:" <<badPoint <<endl;
        }
        PointCloudNT::Ptr alignedPointFromSceneToModel(new PointCloudNT());
        for (int j = 0; j < succeed.size(); ++j) {
            alignedPointFromSceneToModel->push_back(realScene->points[succeed[j]]);
        }

        ///Save some results in the vector
        ResultWithScore r;
        r.score = float(goodPoint) / float(realScene->points.size()) *100;
        r.prob = prob;
        r.resultCloud = result_cloud_icp;
        r.realScene = realScene;
        r.alignedPointFromSceneToModel = alignedPointFromSceneToModel;
        r.rotationMatrix = afterICPTransform;
        r.id = numberOfGoodPose;
        numberOfGoodPose ++;
        R.push_back(r);
        if(!(numberOfGoodPose < this->resultNumber_)){
            break;
        }
    }

    //sort and Visualization
    std::sort(R.begin(),R.end(),resultScoreCompare);

    for( int i=0; i < this->resultNumber_; i++){
        cout<< "id:" <<R[i].id <<" score:"<< R[i].score <<" prob:"<<R[i].prob<<endl;
        //cout << R[i].rotationMatrix << endl;
    }

    for (int i = 0; i < this->resultNumber_; ++i) {
        cout << "view id:" <<R[i].id <<endl;
        pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer());
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
        viewer->addPointCloudNormals<PointNT, PointNT>(
                R[i].resultCloud,
                R[i].resultCloud,
                1, 1,
                "R[i].resultCloud");

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
    return R;
}

void PPFMatch::withoutICP() {
    PointCloudNT::Ptr bestResult = this->getBestResult();
    TransformWithProbVector transforms = this->getTransforms();
    ResultWithScoreVector R;

    int numberOfGoodPose = 0;
    for (TransformWithProb &transform : transforms) {
        ///use ICP to align the scene
        Eigen::Matrix4f transform_mat = transform.transform_mat;
        int prob = transform.prob;

        if (numberOfGoodPose == 0) {
            cout << "before icp(id:"<<numberOfGoodPose<<"):" << endl << transform_mat << endl;
        }
        PointCloudNT::Ptr result_cloud(new PointCloudNT());
        ///just change the pose, didn't change the normal
        pcl::transformPointCloud(*(this->m_modelWithNormals_), *result_cloud, transform_mat);
        PointCloudNT::Ptr input_cloud = result_cloud;
        ///Save some results in the vector
        ResultWithScore r;
        r.prob = transform.prob;
        r.resultCloud = input_cloud;
        r.rotationMatrix = transform.transform_mat;
        r.id = numberOfGoodPose;
        numberOfGoodPose ++;
        R.push_back(r);
        if(!(numberOfGoodPose < this->resultNumber_)){
            break;
        }
    }

    for( int i=0; i < this->resultNumber_; i++){
        cout<< "i:" << i << " score:" << R[i].prob <<endl;
    }

    for (int i = 0; i < this->resultNumber_; i++) {

        pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer());

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
        viewer->addPointCloudNormals<PointNT, PointNT>(
                R[i].resultCloud,
                R[i].resultCloud,
                1, 1,
                "R[i].resultCloud");

        /*ColorHandlerT blue(R[i].alignedPointFromSceneToModel, 0, 0, 255);
        viewer->addPointCloud<PointNT>(R[i].alignedPointFromSceneToModel, blue, "alignedPointFromSceneToModel");
        viewer->setPointCloudRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                4, "alignedPointFromSceneToModel");*/

        while (!viewer->wasStopped()) {
            viewer->spinOnce();
        }
        viewer->resetStoppedFlag();
        viewer->close();
    }

}

PPFMatch::~PPFMatch() {

}

TransformWithProbVector PPFMatch::distanceClustering(TransformWithProbVector poses) {

    std::sort(poses.begin(), poses.end(), transformScoreCompare);
    std::vector<TransformWithProbVector> clustersDis; //just cluster the distance
    //pair.first = pose number;  pair.second = sum votes;
    std::vector<std::pair<size_t, unsigned int> > clusterVotes;
    for (size_t poses_i = 0; poses_i < poses.size(); ++ poses_i) {
        bool foundCluster = false;
        for (size_t clusters_i = 0; clusters_i < clustersDis.size(); ++ clusters_i) {
            if (posesWithinErrorDistance (poses[poses_i].transform_mat, clustersDis[clusters_i].front ().transform_mat)) {
                foundCluster = true;
                clustersDis[clusters_i].push_back (poses[poses_i]);
                clusterVotes[clusters_i].second += (int)poses[poses_i].prob;
                break;
            }
        }

        if (foundCluster == false) {
            // Create a new cluster with the current pose
            TransformWithProbVector newClusterDis;
            newClusterDis.push_back (poses[poses_i]);
            clustersDis.push_back (newClusterDis);
            clusterVotes.push_back (std::pair<size_t, unsigned int> (clustersDis.size () - 1, (int)poses[poses_i].prob));
        }
    }
    cout <<"cluster size :" << clustersDis.size()<<endl;

    TransformWithProbVector clu;
    for (int i = 0; i < clustersDis.size(); ++i) {
        TransformWithProb singleObject;
        cout << clustersDis[i].size() <<endl;
        singleObject.transform_mat = clustersDis[i].front ().transform_mat;
        singleObject.prob = clusterVotes[i].second;
        clu.push_back(singleObject);
    }
    return clu;
}

TransformWithProbVector PPFMatch::NMSClustering(vector<QuaternionWithVote> poses) {

    std::sort(poses.begin(), poses.end(), quaternionScoreCompare);
    std::vector<vector<QuaternionWithVote> > clustersDis;
    std::vector<std::pair<size_t, unsigned int> > clusterVotes;
    for (size_t poses_i = 0; poses_i < poses.size(); ++ poses_i) {
        bool foundCluster = false;
        for (size_t clusters_i = 0; clusters_i < clustersDis.size(); ++ clusters_i) {
            if (posesWithinError (poses[poses_i].quaternion7, clustersDis[clusters_i].front ().quaternion7)) {
                foundCluster = true;
                clustersDis[clusters_i].push_back (poses[poses_i]);
                clusterVotes[clusters_i].second += poses[poses_i].votes;
                break;
            }
        }

        if (foundCluster == false) {
            // Create a new cluster with the current pose
            vector<QuaternionWithVote> newClusterDis;
            newClusterDis.push_back (poses[poses_i]);
            clustersDis.push_back (newClusterDis);
            clusterVotes.push_back (std::pair<size_t, unsigned int> (clustersDis.size () - 1, poses[poses_i].votes));
        }
    }
    cout <<"cluster size :" << clustersDis.size() << endl;
    vector<QuaternionWithVote> result;
    for (int cluster_i = 0; cluster_i < clustersDis.size(); ++cluster_i) {
        vector<QuaternionWithVote> q = clustersDis[cluster_i];
        result.push_back(averageCluster(q));
    }

    TransformWithProbVector clu;
    for (int i = 0; i < result.size(); ++i) {
        TransformWithProb singleObject;
        Vector4f  quaterniondVec;
        quaterniondVec[0] = result[i].quaternion7[0];
        quaterniondVec[1] = result[i].quaternion7[1];
        quaterniondVec[2] = result[i].quaternion7[2];
        quaterniondVec[3] = result[i].quaternion7[3];
        Vector3f  translationVec;
        translationVec[0] = result[i].quaternion7[4];
        translationVec[1] = result[i].quaternion7[5];
        translationVec[2] = result[i].quaternion7[6];
        Matrix3f  rot_mat = ppfExtractor_.QuaternionToRotationMatrix(quaterniondVec);
        Eigen::Matrix4f transform_mat = ppfExtractor_.RotationAndTranslation2Transformation(rot_mat, translationVec);

        singleObject.transform_mat = transform_mat;
        singleObject.prob = (float)clusterVotes[i].second;
        clu.push_back(singleObject);
    }
    return clu;
}

bool PPFMatch::posesWithinErrorDistance(Eigen::Matrix4f t1, Eigen::Matrix4f t2) {
    float position_diff = (diffTransmat(t1,t2));
    if (position_diff < 8) //5
        return true;
    return false;
}

bool PPFMatch::posesWithinError(vector<float> t1, vector<float> t2) {
    /*PPF t;
    Eigen::Vector4f quaternion1 = t.RotationMatrixToQuaternion
            (t1.block(0, 0, 3, 3));
    Eigen::Vector4f quaternion2 = t.RotationMatrixToQuaternion
            (t2.block(0, 0, 3, 3));

    Vector4f angle1(quaternion1[0], quaternion1[1], quaternion1[2], quaternion1[3]);
    Vector4f angle2(quaternion2[0], quaternion2[1], quaternion2[2], quaternion2[3]);

    if ((angle1 - angle2).norm() > 0.8)
        return false;

    Matrix3f r1 = t1.block(0, 0, 3, 3);
    Matrix3f r2 = t2.block(0, 0, 3, 3);

    Eigen::AngleAxisf rotation_diff_mat((r1.inverse().lazyProduct(r2).eval()));

    float rotation_diff_angle = fabsf (rotation_diff_mat.angle ());

    float position_diff = (diffTransmat(t1,t2));

    if (position_diff > 5)
        return false;

    return true;*/

    Vector4f angle1(t1[0], t1[1], t1[2], t1[3]);
    Vector4f angle2(t2[0], t2[1], t2[2], t2[3]);
    if ((angle1 - angle2).norm() > 0.5)
        return false;
    Vector3f trans1(t1[4], t1[5], t1[6]);
    Vector3f trans2(t2[4], t2[5], t2[6]);
    if ((trans1 - trans2).norm() > 8)
        return false;

    return true;
}

QuaternionWithVote PPFMatch::averageCluster(vector<QuaternionWithVote> q) {
    QuaternionWithVote aveCluster;
    int size = q.size();
    if(size == 0){
        cerr<<"this cluster is null!"<<endl;
        return aveCluster;
    }
    aveCluster.votes = -1;
    for (int j = 0; j < 7; ++j) {
        aveCluster.quaternion7.push_back(0);
    }
    for (int i = 0; i < size; ++i) {
        aveCluster.quaternion7[0] += q[i].quaternion7[0];
        aveCluster.quaternion7[1] += q[i].quaternion7[1];
        aveCluster.quaternion7[2] += q[i].quaternion7[2];
        aveCluster.quaternion7[3] += q[i].quaternion7[3];
        aveCluster.quaternion7[4] += q[i].quaternion7[4];
        aveCluster.quaternion7[5] += q[i].quaternion7[5];
        aveCluster.quaternion7[6] += q[i].quaternion7[6];
    }
    aveCluster.quaternion7[0] /= size;
    aveCluster.quaternion7[1] /= size;
    aveCluster.quaternion7[2] /= size;
    aveCluster.quaternion7[3] /= size;
    aveCluster.quaternion7[4] /= size;
    aveCluster.quaternion7[5] /= size;
    aveCluster.quaternion7[6] /= size;
    return aveCluster;
}



#include "../include/PointCloudPretreatment.h"

bool fileExist(const string &name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}

bool isContainsStr(string str,string contains_str) {
    string::size_type idx = str.find(contains_str);
    if (idx!=string::npos) {
        return true;
    }
    else {
        return false;
    }
}

PointCloudPretreatment::PointCloudPretreatment() {

}

PointCloudPretreatment::PointCloudPretreatment(const string &modelFilePath, float modelLeafSize, const string &sceneFilePath,
                               float backgroundThreshold, float normalThreshold, float sceneLeafSize) {
    this->modelLeafSize_ = modelLeafSize;
    this->modelFilePath_ = modelFilePath;
    this->sceneFilePath = sceneFilePath;
    this->backgroundThreshold = backgroundThreshold;
    this->normalThreshold = normalThreshold;
    this->sceneLeafSize = sceneLeafSize;

    this->setModel();
    this->setScene();
}

void PointCloudPretreatment::setModel() {

    float modelLeafSize = this->modelLeafSize_;
    string modelFile = "../model/modelSample_";
    ostringstream oss;
    oss << modelLeafSize;
    string str(oss.str());
    string Path = "f.pcd";
    string modelFilePath = modelFile + str + Path;

    if( 0 ) { /*fileExist(modelFilePath)*/
        cout << " defaultModel is exist" <<endl;
        PointCloudNT::Ptr modelDownsampled_(new PointCloudNT());
        pcl::io::loadPCDFile(modelFilePath, *modelDownsampled_);
        this->modelDownsampled_ = modelDownsampled_;
    } else {
        cout << "defaultModel is not exist!!!" <<endl;
        this->initModel();
    }
}

void PointCloudPretreatment::initModel() {
    PointCloudNT::Ptr model_with_normals(new PointCloudNT());
    PointCloudNT::Ptr modelDownsampled_(new PointCloudNT());
    std::string model_file_path = this->modelFilePath_;
    pcl::io::loadPCDFile(model_file_path, *model_with_normals);
    cout << "model file loaded: " << model_file_path << endl;
    cout << "model_with_normals size:" << model_with_normals->size() << endl;

    // downsample model
    const float modelLeafSize = this->modelLeafSize_;
    pcl::VoxelGrid<PointNT> sor;
    sor.setInputCloud(model_with_normals);
    sor.setLeafSize(modelLeafSize, modelLeafSize, modelLeafSize);
    sor.filter(*modelDownsampled_);
    this->modelDownsampled_ = modelDownsampled_;
    cout << "model_downsampled_size:" << this->modelDownsampled_->size() << endl;

    string outFile = "../modelScene/model/modelSample_";
    ostringstream oss;
    oss << modelLeafSize;
    string str(oss.str());
    string Path = "f.pcd";
    string outFilePath = outFile + str + Path;
    pcl::PCDWriter writer;
    writer.write<PointNT> (outFilePath.c_str (), *modelDownsampled_, false);
}

void PointCloudPretreatment::setScene() {
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene(new pcl::PointCloud<pcl::PointXYZ>());
    if (isContainsStr(this->sceneFilePath,".pcd")) {
        cout << "the scene is .pcd"<<endl;
        pcl::io::loadPCDFile(this->sceneFilePath, *scene);
        this->scene = scene;
    } else if(isContainsStr(this->sceneFilePath,".ply")) {
        cout << "the scene is .ply" <<endl;
        pcl::io::loadPLYFile(this->sceneFilePath, *scene);
        this->scene = scene;
    } else {
        cerr << "file is wrong!" <<endl;
    }

    cout << "scene file loaded:" << this->sceneFilePath << endl;
    cout << "scene size:" << this->scene->size() << endl;

    this->segmentBackground(this->scene);
    this->estimateNormal();
    this->downSample();
}

void PointCloudPretreatment::segmentBackground(pcl::PointCloud<pcl::PointXYZ>::Ptr scene) {
    ///caculate background
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    // Optional
    seg.setOptimizeCoefficients (true);
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (this->backgroundThreshold);
    seg.setInputCloud (scene);
    seg.segment (*inliers, *coefficients);
    cout << "Background plane size: " << inliers->indices.size() << endl;
    cout << "Size of Foreground points:  " << scene->size() - inliers->indices.size() << endl;

    ///segment the background
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene_without_plane(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::ExtractIndices<pcl::PointXYZ> eifilter (true);
    eifilter.setInputCloud (scene);
    eifilter.setIndices (inliers);
    eifilter.setNegative (true);
    eifilter.filter (*scene_without_plane);
    this->scene_without_plane = scene_without_plane;

    pcl::visualization::PCLVisualizer::Ptr viewer1(new pcl::visualization::PCLVisualizer());
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red(scene_without_plane, 255, 0, 0);
    viewer1->addPointCloud<pcl::PointXYZ>(scene_without_plane, red, "scene_without_plane");
    viewer1->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
            2, "scene_without_plane");
    while (!viewer1->wasStopped()) {
        viewer1->spinOnce();
    }
    viewer1->resetStoppedFlag();
    viewer1->close();

}

void PointCloudPretreatment::estimateNormal() {

    PointCloudNT::Ptr sceneWithNormals(new PointCloudNT());
    pcl::PointCloud<pcl::Normal>::Ptr scene_normals(new pcl::PointCloud<pcl::Normal>());
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_scene(new pcl::search::KdTree<pcl::PointXYZ>());
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne_scene;

    ne_scene.setInputCloud(this->scene_without_plane);
    ne_scene.setSearchMethod(tree_scene);
    ne_scene.setRadiusSearch(this->normalThreshold);
    ne_scene.compute(*scene_normals);

    pcl::concatenateFields(*(this->scene_without_plane), *scene_normals, *sceneWithNormals);
    cout << "scene_with_normals size:" << sceneWithNormals->size() << endl;
    this->sceneWithNormals = sceneWithNormals;
    /*string outfile = "../modelScene/scene/sceneNormal0.5.ply";
    pcl::PLYWriter writer;
    writer.write<PointNT > (outfile.c_str (), *(this->sceneWithNormals), false);*/
}

void PointCloudPretreatment::downSample() {
    PointCloudNT::Ptr sceneDownsampled_(new PointCloudNT());
    const float scene_leaf_size = this->sceneLeafSize;
    pcl::VoxelGrid<PointNT> sor_scene;
    sor_scene.setInputCloud(this->sceneWithNormals);
    sor_scene.setLeafSize(scene_leaf_size, scene_leaf_size, scene_leaf_size);
    sor_scene.filter(*sceneDownsampled_);
    cout << "scene_downsampled size:" << sceneDownsampled_->size() << endl;
    this->sceneDownsampled_ = sceneDownsampled_;
    /*pcl::PCDWriter writer;
    writer.write<PointNT > (outfile.c_str (), *(this->sceneDownsampled_), false);*/
}

PointCloudPretreatment::~PointCloudPretreatment(){

}


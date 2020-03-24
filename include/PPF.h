#ifndef PPF_PPF_H
#define PPF_PPF_H
#pragma once
#include <iostream>
#include<fstream>
#include<vector>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/centroid.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
#include <pcl/console/time.h>
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/console/parse.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/io/ply_io.h>
#include <math.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
using namespace std;
using namespace Eigen;

class PPF
{
public:
    PPF();
    void Set(float d_distance, int nAngle, int max_nDistance, int max_hashIndex);

    int CreateFeatureHashIndex(pcl::PointNormal  pn1, pcl::PointNormal  pn2);

    Eigen::Matrix4f   RotateAboutAnyVector(Vector3f  vec, float angle);
    Eigen::Matrix4f   CreateTransformation2AlignNormWithX(pcl::PointNormal  pn);
    Eigen::Matrix4f   CreateTransformationFromModelToScene(pcl::PointNormal  pm, pcl::PointNormal  ps, float alpha);

    bool isRotationMatrix(Eigen::Matrix3f R);

    Vector4f  RotationMatrixToQuaternion(Matrix3f  rotationMat);
    Matrix3f  QuaternionToRotationMatrix(Vector4f  quaterniondVec);
    Matrix4f   RotationAndTranslation2Transformation(Matrix3f  rotationMat, Vector3f  translationVec);

    Vector3f  RotationMat2EulerAngles(Matrix3f  rotationMat);
    Matrix3f  EulerAnglesToRotationMatrix(Vector3f  EulerAngles);
    Matrix4f   EulerAnglesAndTranslation2Transformation( Vector3f  EulerAngles, Vector3f  translationVec);

    float  CreateAngle2TouchXZPostivePlane(pcl::PointNormal  pn, pcl::PointNormal  pn2);

    float d_distance;
    int max_nDistance;  ///pointcloud max distance
    int max_hashIndex;

    float Min_angle;
    float Max_angle;
    int nAngle;
    float d_angle;

    float alpha; //the rotated angle with respect to x-axis in order to touch  the  (x plus z+) half-plane after the norm is aligned with the x-axis

    int F1;
    int F2;
    int F3;
    int F4;
    int hashIndex;

    int P1;
    int P2;
    int P3;
    int P4;

    ~PPF();
};
#endif //PPF_PPF_H

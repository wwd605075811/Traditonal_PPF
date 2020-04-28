#ifndef PPF_PPF_H
#define PPF_PPF_H
#pragma once
#include <iostream>
#include<fstream>
#include<vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/surface/mls.h>
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
    /**
     * \简介 计算ppfs和hash值
     * @param m_r
     * @param m_i
     * @return hash值
     */
    int CreateFeatureHashIndex(pcl::PointNormal  pn1, pcl::PointNormal pn2);
    /**
     * \简介 goldman公式，绕旋转轴旋转
     * @param vec 旋转轴
     * @param angle 转转角度
     * @return 旋转矩阵
     */
    Eigen::Matrix4f RotateAboutAnyVector(Vector3f vec, float angle);
    Eigen::Matrix4f CreateTransformation2AlignNormWithX(pcl::PointNormal pn);
    Eigen::Matrix4f CreateTransformationFromModelToScene(pcl::PointNormal pm, pcl::PointNormal ps, float alpha);
    /**
     * \简介 判断是否是旋转矩阵
     * @param 旋转矩阵
     * @return bool
     *  \note 因为是float,计算有误差，1e-6的阈值可能有点小
     */
    bool isRotationMatrix(Eigen::Matrix3f R);
    /**
     * \简介 旋转矩阵转化为四元数
     * @param 旋转矩阵
     * @return 四元数
     */
    Vector4f RotationMatrixToQuaternion(Matrix3f rotationMat);
    /**
     * \简介 四元数转为旋转矩阵
     * @param 四元数
     * @return 旋转矩阵
     */
    Matrix3f QuaternionToRotationMatrix(Vector4f quaterniondVec);
    Matrix4f RotationAndTranslation2Transformation(Matrix3f rotationMat, Vector3f translationVec);
    /**
     * \简介 四元数转欧拉角
     * @param 四元数
     * @return 欧拉角
     */
    Vector3f QuaternionToEulerAngles(Eigen::Vector4f R);
    /**
     * \简介 旋转矩阵转欧拉角
     * @param 旋转矩阵
     * @return 欧拉角
     */
    Vector3f RotationMat2EulerAngles(Matrix3f rotationMat);
    /**
     * \简介 欧拉角转旋转矩阵
     * @param 欧拉角
     * @return 旋转矩阵
     */
    Matrix3f EulerAnglesToRotationMatrix(Vector3f EulerAngles);
    Matrix4f EulerAnglesAndTranslation2Transformation( Vector3f EulerAngles, Vector3f translationVec);
    float CreateAngle2TouchXZPostivePlane(pcl::PointNormal pn, pcl::PointNormal pn2);

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

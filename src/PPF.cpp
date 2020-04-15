#include "../include/PPF.h"

PPF::PPF()
{
    this->d_distance = 0.01;//1cm

    Min_angle = -M_PI;
    Max_angle = M_PI;
    nAngle = 30;
    d_angle = (Max_angle - Min_angle) / nAngle;
    this->max_nDistance = 25;
    this->max_hashIndex = 10000;

    F1 = 0;
    F2 = 0;
    F3 = 0;
    F4 = 0;
    hashIndex = 0;

    P1 = nAngle * nAngle * this->max_nDistance;
    P2 = nAngle * nAngle;
    P3 = nAngle;
    P4 = 1;

}
void PPF::Set(float d_distance, int nAngle,  int max_nDistance, int max_hashIndex)
{
    this->d_distance = d_distance;

    Min_angle = -M_PI;
    Max_angle = M_PI;
    d_angle = (Max_angle-Min_angle)/ nAngle;

    this->max_nDistance = max_nDistance;
    this->max_hashIndex = max_hashIndex;

    F1 = 0;
    F2 = 0;
    F3 = 0;
    F4 = 0;

    P1 = nAngle * nAngle * this->max_nDistance;
    P2 = nAngle * nAngle;
    P3 = nAngle;
    P4 = 1;

    hashIndex = 0;

}

PPF::~PPF() {
}

int PPF::CreateFeatureHashIndex(pcl::PointNormal  pn1, pcl::PointNormal  pn2)
{
    Vector3f  pt1(pn1.x, pn1.y, pn1.z);
    Vector3f  normal1(pn1.normal_x, pn1.normal_y, pn1.normal_z);

    Vector3f  pt2(pn2.x, pn2.y, pn2.z);
    Vector3f  normal2(pn2.normal_x, pn2.normal_y, pn2.normal_z);

    Vector3f d= (pt2 - pt1);
    float dis = d.norm();
    F1 = int(dis / d_distance);

    if (F1 > this->max_nDistance)
        return -1;

    d.normalize();
    float angel_d_n1 = atan2((d.cross(normal1)).norm(), d.dot(normal1));
    F2 =int( (angel_d_n1 - Min_angle) / (Max_angle - Min_angle)*nAngle);

    float angel_d_n2 = atan2((d.cross(normal2)).norm(), d.dot(normal2));
    F3 = int((angel_d_n2 - Min_angle) / (Max_angle - Min_angle)*nAngle);

    float angel_n1_n2 = atan2((normal1.cross(normal2)).norm(), normal1.dot(normal2));
    F4= int((angel_n1_n2 - Min_angle) / (Max_angle - Min_angle)*nAngle);

    hashIndex = (F1*P1 + F2*P2 + F3*P3+ F4*P4) % max_hashIndex;//求余法构造Hash索引

    return hashIndex;
}

Eigen::Matrix4f PPF::RotateAboutAnyVector(Vector3f  vec,  float angle)
{
    // normalize axis vector  
    vec.normalize();
    float x =vec[0];
    float y = vec[1];
    float z =vec[2];
    // compute rotation matrix  
    float c = cosf(angle);
    float s = sinf(angle);
    float v = 1 - c;

    Matrix4f m;  // Goldman公式
    m(0, 0) = x*x*v + c;      m(0, 1) = x*y*v - z*s;     m(0, 2) = x*z*v + y*s;
    m(1, 0) = x*y*v + z*s;  m(1, 1) = y*y*v + c;        m(1, 2) = y*z*v - x*s;
    m(2, 0) = x*z*v - y*s;   m(2, 1) = y*z*v + x*s;    m(2, 2) = z*z*v + c;

    m(0, 3) = m(1, 3) = m(2, 3) = 0;
    m(3, 0) = m(3, 1) = m(3, 2) = 0; m(3, 3) = 1;
    return m;

}

Eigen::Matrix4f  PPF::CreateTransformation2AlignNormWithX(pcl::PointNormal pn)
{
    Matrix4f translation_mat;
    translation_mat.setIdentity();
    translation_mat(0, 3) = -pn.x;
    translation_mat(1, 3) = -pn.y;
    translation_mat(2, 3) = -pn.z;

    float thetaY;
    Matrix4f rot_y;
    thetaY = atan2(pn.normal_z, pn.normal_x);
    rot_y(0,0) = cosf(thetaY); rot_y(0,1) = 0;
    rot_y(0,2) = sinf(thetaY); rot_y(0,3) = 0;
    rot_y(1,1) = 1; rot_y(1,0) = 0; rot_y(1,2) = 0; rot_y(1,3) = 0;
    rot_y(2,0) = -1*rot_y(0,2); rot_y(2,1) = 0;
    rot_y(2,2) = rot_y(0,0); rot_y(2,3) = 0;
    rot_y(3,3) = 1; rot_y(3,0) = 0; rot_y(3,1) = 0; rot_y(3,2) = 0;

    Vector4f n_tmp(pn.normal_x, pn.normal_y, pn.normal_z, 1);
    n_tmp = rot_y * n_tmp;

    float thetaZ;
    Matrix4f rot_z;
    thetaZ = -1*atan2(n_tmp[1], n_tmp[0]);
    rot_z(0,0) = cosf(thetaZ); rot_z(0,2) = 0;
    rot_z(1,0) = sinf(thetaZ); rot_z(1,2) = 0;
    rot_z(0,1) = -1*rot_z(1,0); rot_z(0,3) = 0;
    rot_z(1,1) = rot_z(0,0); rot_z(1,3) = 0;
    rot_z(2,2) = 1; rot_z(2,0) = 0; rot_z(2,1) = 0; rot_z(2,3) = 0;
    rot_z(3,3) = 1; rot_z(3,0) = 0; rot_z(3,1) = 0; rot_z(3,2) = 0;

    Matrix4f T_tmp;
    T_tmp = rot_z * rot_y;

    Matrix4f T_m_g;
    T_m_g = T_tmp * translation_mat;

    return  T_m_g;
}

float  PPF::CreateAngle2TouchXZPostivePlane(pcl::PointNormal  pn,  pcl::PointNormal  pn2)
{
    Matrix4f  transform_mat = CreateTransformation2AlignNormWithX(pn);
    Vector4f  pt(pn2.x, pn2.y, pn2.z,1.0);
    Vector4f  trans_pt = transform_mat * pt;

    float  x = trans_pt[0];
    float  y = trans_pt[1];
    float  z = trans_pt[2];


    float rot_angle = atan2(y, z);
    alpha = rot_angle;

    return  rot_angle;


}

Eigen::Matrix4f   PPF::CreateTransformationFromModelToScene(pcl::PointNormal  pm, pcl::PointNormal  ps, float alpha)
{
    Matrix4f  Tmg = CreateTransformation2AlignNormWithX(pm);
    Matrix4f  Tsg = CreateTransformation2AlignNormWithX(ps);
    Matrix4f  Rxa = RotateAboutAnyVector(Vector3f(1, 0, 0), alpha);
    Matrix4f  trans_mat = Tsg.inverse()*Rxa*Tmg;
    return  trans_mat;
}

bool PPF::isRotationMatrix(Eigen::Matrix3f   R)
{
    Eigen::Matrix3f Rt;
    Rt = R.transpose();
    Eigen::Matrix3f  shouldBeIdentity = Rt * R;
    Eigen::Matrix3f  I = Matrix3f::Identity();

    Eigen::Matrix3f difMat = I - Rt;
    return difMat.norm()< 1e-6;

}

Eigen::Vector3f PPF::RotationMat2EulerAngles(Eigen::Matrix3f R)
{
    assert(isRotationMatrix(R));
    float sy = sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0));
    bool singular = sy < 1e-6; // If
    float x, y, z;
    if (!singular)
    {
        x = atan2(R(2, 1), R(2, 2));
        y = atan2(-R(2, 0), sy);
        z = atan2(R(1, 0), R(0, 0));
    }
    else {
        x = atan2(-R(1, 2), R(1, 1));
        y = atan2(-R(2, 0), sy);
        z = 0;
    }
    return  Eigen::Vector3f(x, y, z);

}

Vector4f   PPF::RotationMatrixToQuaternion(Matrix3f  rotationMat)
{
    //cout << rotationMat << endl << endl;
    //RotationMatrix to EulerAngles
    Eigen::Vector3f ea1 = rotationMat.eulerAngles(2, 1, 0);

    //EulerAngles to RotationMatrix
    Eigen::Matrix3f  R;
    R = Eigen::AngleAxisf(ea1[0], Eigen::Vector3f::UnitZ())
        * Eigen::AngleAxisf(ea1[1], Eigen::Vector3f::UnitY())
        * Eigen::AngleAxisf(ea1[2], Eigen::Vector3f::UnitX());

    //cout << R << endl << endl;

    //RotationMatrix to Quaterniond
    Eigen::Quaternionf q;
    q = R;


    Vector4f quaterniondVec;
    quaterniondVec[0] = q.x();
    quaterniondVec[1] = q.y();
    quaterniondVec[2] = q.z();
    quaterniondVec[3] = q.w();

    return  quaterniondVec;

}

Matrix4f   PPF::RotationAndTranslation2Transformation(Matrix3f  rotationMat, Vector3f  translationVec)
{
    Matrix3f  rot_mat = rotationMat;

    Matrix4f    trans_mat;
    trans_mat.setIdentity();
    for (int i = 0; i<3; i++)
        for (int j = 0; j < 3; j++)
        {
            trans_mat(i, j) = rot_mat(i, j);
        }
    trans_mat(0, 3) = translationVec[0];
    trans_mat(1, 3) = translationVec[1];
    trans_mat(2, 3) = translationVec[2];

    return  trans_mat;


}

Matrix3f  PPF::QuaternionToRotationMatrix(Vector4f  quaterniondVec)
{
    Quaternionf q;
    q.x() = quaterniondVec[0];
    q.y() = quaterniondVec[1];
    q.z() = quaterniondVec[2];
    q.w() = quaterniondVec[3];

    return q.toRotationMatrix();

}

Matrix4f   PPF::EulerAnglesAndTranslation2Transformation(Vector3f  EulerAngles, Vector3f  translationVec)
{
    Matrix3f  rot_mat = EulerAnglesToRotationMatrix(EulerAngles);

    Matrix4f    trans_mat;
    trans_mat.setIdentity();
    for( int i=0;i<3;i++)
        for (int j = 0; j < 3; j++)
        {
            trans_mat(i, j) = rot_mat(i, j);
        }
    trans_mat(0, 3) = translationVec[0];
    trans_mat(1, 3) = translationVec[1];
    trans_mat(2, 3) = translationVec[2];

    return  trans_mat;

}

Matrix3f  PPF::EulerAnglesToRotationMatrix(Vector3f  EulerAngles)
{
    Vector3f  theta = EulerAngles;
    // 计算旋转矩阵的X分量

    Matrix3f R_x;
    R_x<<
       1, 0, 0,
            0, cos(theta[0]), -sin(theta[0]),
            0, sin(theta[0]), cos(theta[0]);

    // 计算旋转矩阵的Y分量
    Matrix3f R_y;
    R_y<<
       cos(theta[1]), 0, sin(theta[1]),
            0, 1, 0,
            -sin(theta[1]), 0, cos(theta[1]);

    // 计算旋转矩阵的Z分量
    Matrix3f  R_z;
    R_z<<
       cos(theta[2]), -sin(theta[2]), 0,
            sin(theta[2]), cos(theta[2]), 0,
            0, 0, 1;

    // 合并
    Matrix3f R = R_z * R_y * R_x;

    return R;

}
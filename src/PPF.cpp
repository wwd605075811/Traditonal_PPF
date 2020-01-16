//
// Created by wwd on 2019/12/24.
//
#include "PPF.h"

void PPF::GPU_init(){

    cout<<"we are call a function from .cpp to .cu!"<<endl;
}


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

PPF::~PPF(){}

void PPF::Set(float d_distance, int nAngle,  int max_nDistance, int max_hashIndex){

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

int PPF::CreateFeatureHashIndex(pcl::PointNormal  pn1, pcl::PointNormal  pn2)
{
    Vector3f  pt1(pn1.x, pn1.y, pn1.z);
    Vector3f  normal1(pn1.normal_x, pn1.normal_y, pn1.normal_z);

    Vector3f  pt2(pn2.x, pn2.y, pn2.z);
    Vector3f  normal2(pn2.normal_x, pn2.normal_y, pn2.normal_z);

    Vector3f d= (pt2 - pt1);

    float dis = d.norm();  //平方根

    F1 = int(dis / d_distance);

    if (F1 > this->max_nDistance)
        return -1;

    d.normalize();

    float angel_d_n1 = atan2((d.cross(normal1)).norm(), d.dot(normal1));

    F2 =int( (angel_d_n1 - Min_angle) / (Max_angle - Min_angle)*nAngle);



    ///elem1= cha ji de mo  elem2=dianji
    float angel_d_n2 = atan2((d.cross(normal2)).norm(), d.dot(normal2));
    F3 = int((angel_d_n2 - Min_angle) / (Max_angle - Min_angle)*nAngle);

    float angel_n1_n2 = atan2((normal1.cross(normal2)).norm(), normal1.dot(normal2));
    F4= int((angel_n1_n2 - Min_angle) / (Max_angle - Min_angle)*nAngle);

    hashIndex = (F1*P1 + F2*P2 + F3*P3+ F4*P4) % max_hashIndex;//求余法构造Hash索引

    return hashIndex;
}


Eigen::Matrix4f   PPF::RotateAboutAnyVector(Vector3f  vec,  float angle)
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

    Matrix4f m;  //下为 Goldman公式
    m(0, 0) = x*x*v + c;      m(0, 1) = x*y*v - z*s;     m(0, 2) = x*z*v + y*s;
    m(1, 0) = x*y*v + z*s;  m(1, 1) = y*y*v + c;        m(1, 2) = y*z*v - x*s;
    m(2, 0) = x*z*v - y*s;   m(2, 1) = y*z*v + x*s;    m(2, 2) = z*z*v + c;

    m(0, 3) = m(1, 3) = m(2, 3) = 0;
    m(3, 0) = m(3, 1) = m(3, 2) = 0; m(3, 3) = 1;
    return m;

}

Eigen::Matrix4f  PPF::CreateTransformation2AlignNormWithX(pcl::PointNormal  pn)
{
    Vector3f  pt(pn.x, pn.y, pn.z);
    Vector3f  normal(pn.normal_x, pn.normal_y, pn.normal_z);

    Vector3f  rotAxisVec;
    Vector3f  x_axis(1, 0, 0);
    float rot_angle = 0;
    float cosAngle = normal.dot(x_axis);
    if (fabs(cosAngle - 1) < 0.000001) //与x轴重合
    {
        rotAxisVec = x_axis;
        rot_angle = 0;

    }
    else
    if (fabs(cosAngle + 1) < 0.000001) //与-x轴重合
    {
        rotAxisVec = Vector3f(0,0,1);
        rot_angle = M_PI;

    }
    else
    {
        rotAxisVec = normal.cross(x_axis);
        Vector3f  newYVec = rotAxisVec.cross(x_axis);

        float  x_part = normal.dot(x_axis);
        float  y_part = normal.dot(newYVec);

        float angle_from_x_to_normal = atan2(y_part, x_part);


        rot_angle = -angle_from_x_to_normal;  //将法向转到x轴的方向

    }

    Matrix4f rot_mat = RotateAboutAnyVector(rotAxisVec, rot_angle);

    Matrix4f   translation_mat;
    translation_mat.setIdentity();
    translation_mat(0, 3) = -pt[0];
    translation_mat(1, 3) = -pt[1];
    translation_mat(2, 3) = -pt[2];

    Matrix4f     transform_mat = rot_mat*translation_mat;

    return  transform_mat;

}

float  PPF::CreateAngle2TouchXZPostivePlane(pcl::PointNormal  pn,  pcl::PointNormal  pn2)
{
    Matrix4f  transform_mat = CreateTransformation2AlignNormWithX(pn);
  /*  if(wwdflag==1){
        cout<<wwdflag<<"::"<<endl;
        for(int i=0;i<4;i++){
            for (int j = 0; j < 4; ++j) {
                printf(" %f ",transform_mat(i,j));
            }
            printf("\n");
        }
    }


    if(wwdflag==3666){
        cout<<wwdflag<<"::"<<endl;
        for(int i=0;i<4;i++){
            for (int j = 0; j < 4; ++j) {
                printf(" %f ",transform_mat(i,j));
            }
            printf("\n");
        }
    }

    if(wwdflag==7321){
        cout<<wwdflag<<"::"<<endl;
        for(int i=0;i<4;i++){
            for (int j = 0; j < 4; ++j) {
                printf(" %f ",transform_mat(i,j));
            }
            printf("\n");
        }
    }

    if(wwdflag==10976){
        cout<<wwdflag<<"::"<<endl;
        for(int i=0;i<4;i++){
            for (int j = 0; j < 4; ++j) {
                printf(" %f ",transform_mat(i,j));
            }
            printf("\n");
        }
    }*/

    Vector4f  pt(pn2.x, pn2.y, pn2.z,1.0);
    Vector4f  trans_pt = transform_mat*pt;

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
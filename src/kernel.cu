#include "kernel.cuh"

# define M_PI        3.14159265358979323846    /* pi */
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__ ))

__device__ bool
isFeatureSimilar(int dis_thresh, int angle_thresh, int F1, int cur_F1, int F2,
                 int cur_F2, int F3, int cur_F3, int F4, int cur_F4) {
    if (fabsf(F1 - cur_F1) > dis_thresh || fabsf(F2 - cur_F2) > angle_thresh || fabsf(F3 - cur_F3) > angle_thresh ||
        fabsf(F4 - cur_F4) > angle_thresh)
        return false;

    return true;
}

__device__ float
createRotAngle(CudaPPFInfo p1, CudaPPFInfo p2) {
    MatrixXd m(2, 2);
    float rot_angle = 0;
    float rotAxisVec_x = 0, rotAxisVec_y = 0, rotAxisVec_z = 0;
    float newYVec_x = 0, newYVec_y = 0, newYVec_z = 0;
    //p1_normal dot X axis
    float cosAngle = p1.nomal_x * 1;
    if (fabsf(cosAngle - 1) < 0.000001) {
        rotAxisVec_x = 1;
    } else if (fabsf(cosAngle + 1) < 0.000001) {
        rotAxisVec_z = 1;
        rot_angle = M_PI;
    } else {
        //normal.corss(x_axis)
        rotAxisVec_x = 0;
        rotAxisVec_y = p1.nomal_z;
        rotAxisVec_z = -p1.nomal_y;

        newYVec_x = 0;
        newYVec_y = rotAxisVec_z;
        newYVec_z = -rotAxisVec_y;
        //x_part: normal.dot(x_axis)  y_part:normal.dot(newYVec)
        float x_part, y_part;
        x_part = p1.nomal_x;
        y_part = p1.nomal_y * newYVec_y + p1.nomal_z * newYVec_z;
        float angle_from_x_to_normal = atan2f(y_part, x_part);

        rot_angle = -angle_from_x_to_normal;
    }
    //rotAxisVec.normalise
    float vec_length = sqrtf(powf(rotAxisVec_x, 2) + powf(rotAxisVec_y, 2) +
                             powf(rotAxisVec_z, 2));
    rotAxisVec_x /= vec_length;
    rotAxisVec_y /= vec_length;
    rotAxisVec_z /= vec_length;
    // compute rotation matrix
    float x = rotAxisVec_x;
    float y = rotAxisVec_y;
    float z = rotAxisVec_z;
    float c = cosf(rot_angle);
    float s = sinf(rot_angle);
    float v = 1 - c;

    float rot_mat[4][4];
    rot_mat[0][0] = x * x * v + c;
    rot_mat[0][1] = x * y * v - z * s;
    rot_mat[0][2] = x * z * v + y * s;
    rot_mat[1][0] = x * y * v + z * s;
    rot_mat[1][1] = y * y * v + c;
    rot_mat[1][2] = y * z * v - x * s;
    rot_mat[2][0] = x * z * v - y * s;
    rot_mat[2][1] = y * z * v + x * s;
    rot_mat[2][2] = z * z * v + c;
    rot_mat[0][3] = rot_mat[1][3] = rot_mat[2][3] = 0;
    rot_mat[3][0] = rot_mat[3][1] = rot_mat[3][2] = 0;
    rot_mat[3][3] = 1;

    float translation_mat[4][4];
    //translation_mat.setIdentity();
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (i == j) {
                translation_mat[i][j] = 1;
            } else {
                translation_mat[i][j] = 0;
            }
        }
    }
    translation_mat[0][3] = -p1.x;
    translation_mat[1][3] = -p1.y;
    translation_mat[2][3] = -p1.z;

    float transform_mat[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            transform_mat[i][j] = 0;
            for (int k = 0; k < 4; k++) {
                transform_mat[i][j] += rot_mat[i][k] * translation_mat[k][j];
            }
        }
    }
    float trans_p2_x = p2.x;
    float trans_p2_y = p2.y;
    float trans_p2_z = p2.z;
    float trans_p2_u = 1.0;
    float rot_mat_y = 0;
    float rot_mat_z = 0;
    rot_mat_y = transform_mat[1][0] * trans_p2_x + transform_mat[1][1] *
                                                   trans_p2_y + transform_mat[1][2] * trans_p2_z +
                transform_mat[1][3] * trans_p2_u;
    rot_mat_z = transform_mat[2][0] * trans_p2_x + transform_mat[2][1] *
                                                   trans_p2_y + transform_mat[2][2] * trans_p2_z +
                transform_mat[2][3] * trans_p2_u;

    return atan2f(rot_mat_y, rot_mat_z);
}

__device__ float *
matrixInverse(float *rotMatRxa) {

}

__global__ void
oneDimensionTest(CudaPPFInfo *d_Pc, CudaPPFInfo *d_Pc_copy, float *result_F1, float *result_F2, float *result_F3,
                 float *result_F4, int *result_hash, int nx, CudaOtherInfo g_other) {
    int tid = threadIdx.x;
    printf("tid:%d\n", tid);
    float d_norm, dot, d_length, d_x = 1, d_y = 1, d_z = 1;
    float X2_x = 1, X2_y = 1, X2_z = 1;  //middle cha ji value
    float X3_x = 1, X3_y = 1, X3_z = 1;
    float X4_x = 1, X4_y = 1, X4_z = 1;

    //printf("blockIdx.x:%d * blockDim.x:%d + threadIdx.x:%d = tid:%d\n", blockIdx.x, blockDim.y, threadIdx.x, tid);
    if (tid < nx) {
        for (int j = 0; j < nx; j++) {
            //calculate F1(F1只是PPF.cpp 中的dis)，P1为F1的权值
            result_F1[tid * nx + j] = sqrtf(
                    powf((d_Pc[tid].x - d_Pc_copy[j].x), 2) + powf((d_Pc[tid].y - d_Pc_copy[j].y), 2) +
                    powf((d_Pc[tid].z - d_Pc_copy[j].z), 2));

            result_hash[tid * nx + j] = int(result_F1[tid * nx + j] / g_other.d_distance) * g_other.g_P1;
            //计算中间向量d,由tid 指向 j
            d_x = d_Pc_copy[j].x - d_Pc[tid].x;
            d_y = d_Pc_copy[j].y - d_Pc[tid].y;
            d_z = d_Pc_copy[j].z - d_Pc[tid].z;

            //calculate F2
            //d.normlize
            d_length = sqrtf(powf(d_x, 2) + powf(d_y, 2) + powf(d_z, 2));
            d_x /= d_length;
            d_y /= d_length;
            d_z /= d_length;
            //X and X.norm (X指向量叉积,后期改进再调__device__)
            X2_x = d_y * d_Pc[tid].nomal_z - d_z * d_Pc[tid].nomal_y;
            X2_y = d_z * d_Pc[tid].nomal_x - d_x * d_Pc[tid].nomal_z;
            X2_z = d_x * d_Pc[tid].nomal_y - d_y * d_Pc[tid].nomal_x;
            d_norm = sqrtf(powf(X2_x, 2) + powf(X2_y, 2) + powf(X2_z, 2));

            //dot 点积
            dot = d_x * d_Pc[tid].nomal_x + d_y * d_Pc[tid].nomal_y + d_z * d_Pc[tid].nomal_z;

            result_F2[tid * nx + j] = atan2f(d_norm, dot);
            result_hash[tid * nx + j] +=
                    int((result_F2[tid * nx + j] - g_other.Min_angle) / (g_other.Max_angle - g_other.Min_angle) *
                        g_other.nAngle) * (g_other.g_P2);

            //calculate F3
            X3_x = d_y * d_Pc_copy[j].nomal_z - d_z * d_Pc_copy[j].nomal_y;
            X3_y = d_z * d_Pc_copy[j].nomal_x - d_x * d_Pc_copy[j].nomal_z;
            X3_z = d_x * d_Pc_copy[j].nomal_y - d_y * d_Pc_copy[j].nomal_x;
            d_norm = sqrtf(powf(X3_x, 2) + powf(X3_y, 2) + powf(X3_z, 2));

            dot = d_Pc_copy[j].nomal_x * d_x + d_Pc_copy[j].nomal_y * d_y + d_Pc_copy[j].nomal_z * d_z;

            result_F3[tid * nx + j] = atan2f(d_norm, dot);
            result_hash[tid * nx + j] +=
                    int((result_F3[tid * nx + j] - g_other.Min_angle) / (g_other.Max_angle - g_other.Min_angle) *
                        g_other.nAngle) * (g_other.g_P3);

            //calculate F4
            X4_x = d_Pc[tid].nomal_y * d_Pc_copy[j].nomal_z - d_Pc[tid].nomal_z * d_Pc_copy[j].nomal_y;
            X4_y = d_Pc[tid].nomal_z * d_Pc_copy[j].nomal_x - d_Pc[tid].nomal_x * d_Pc_copy[j].nomal_z;
            X4_z = d_Pc[tid].nomal_x * d_Pc_copy[j].nomal_y - d_Pc[tid].nomal_y * d_Pc_copy[j].nomal_x;
            d_norm = sqrtf(powf(X4_x, 2) + powf(X4_y, 2) + powf(X4_z, 2));

            dot = d_Pc[tid].nomal_x * d_Pc_copy[j].nomal_x + d_Pc[tid].nomal_y * d_Pc_copy[j].nomal_y +
                  d_Pc[tid].nomal_z * d_Pc_copy[j].nomal_z;

            result_F4[tid * nx + j] = atan2f(d_norm, dot);
            result_hash[tid * nx + j] +=
                    int((result_F4[tid * nx + j] - g_other.Min_angle) / (g_other.Max_angle - g_other.Min_angle) *
                        g_other.nAngle) * (g_other.g_P4);

            //calculate hash
            result_hash[tid * nx + j] %= g_other.max_hashIndex;

            if (tid == 0)
                printf("tid: %d dis:%f  hash:%d\n", tid, result_F1[tid * nx + j], result_hash[tid * nx + j]);
        }
    }
}

__global__ void
modelPpf(CudaPPFInfo *d_Pc, CudaPPFInfo *d_Pc_copy, int *result_F1, int *result_F2, int *result_F3,
         int *result_F4, float *result_alpha, int *result_hash, int nx, int ny, CudaOtherInfo g_other) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float f1 = 0, f2 = 0, f3 = 0, f4 = 0;
    float d_norm, dot, d_length, d_x = 1, d_y = 1, d_z = 1;
    float X2_x = 1, X2_y = 1, X2_z = 1;
    float X3_x = 1, X3_y = 1, X3_z = 1;
    float X4_x = 1, X4_y = 1, X4_z = 1;

    if (tid < nx * ny) {
        for (int j = 0; j < nx * ny; j++) {
            //calculate F1, f1 is the dis
            f1 = sqrtf(powf((d_Pc[tid].x - d_Pc_copy[j].x), 2) + powf((d_Pc[tid].y - d_Pc_copy[j].y), 2) +
                       powf((d_Pc[tid].z - d_Pc_copy[j].z), 2));
            result_F1[tid * nx * ny + j] = int(f1 / g_other.d_distance);
            result_hash[tid * nx * ny + j] = result_F1[tid * nx * ny + j] * g_other.g_P1;

            //the d
            d_x = d_Pc_copy[j].x - d_Pc[tid].x;
            d_y = d_Pc_copy[j].y - d_Pc[tid].y;
            d_z = d_Pc_copy[j].z - d_Pc[tid].z;

            //calculate F2,and f1 is the angel_d_n1
            //d.normlize
            d_length = sqrtf(powf(d_x, 2) + powf(d_y, 2) + powf(d_z, 2));
            d_x /= d_length;
            d_y /= d_length;
            d_z /= d_length;
            //X and X.norm (X指向量叉积,后期改进再调__device__)
            X2_x = d_y * d_Pc[tid].nomal_z - d_z * d_Pc[tid].nomal_y;
            X2_y = d_z * d_Pc[tid].nomal_x - d_x * d_Pc[tid].nomal_z;
            X2_z = d_x * d_Pc[tid].nomal_y - d_y * d_Pc[tid].nomal_x;
            d_norm = sqrtf(powf(X2_x, 2) + powf(X2_y, 2) + powf(X2_z, 2));

            //dot 点积
            dot = d_x * d_Pc[tid].nomal_x + d_y * d_Pc[tid].nomal_y + d_z * d_Pc[tid].nomal_z;

            f2 = atan2f(d_norm, dot);
            result_F2[tid * nx * ny + j] = int((f2 - g_other.Min_angle) / (g_other.Max_angle - g_other.Min_angle) *
                                               g_other.nAngle);
            result_hash[tid * nx * ny + j] += result_F2[tid * nx * ny + j] * (g_other.g_P2);

            //calculate F3,f3 is the angel_d_n2
            X3_x = d_y * d_Pc_copy[j].nomal_z - d_z * d_Pc_copy[j].nomal_y;
            X3_y = d_z * d_Pc_copy[j].nomal_x - d_x * d_Pc_copy[j].nomal_z;
            X3_z = d_x * d_Pc_copy[j].nomal_y - d_y * d_Pc_copy[j].nomal_x;
            d_norm = sqrtf(powf(X3_x, 2) + powf(X3_y, 2) + powf(X3_z, 2));

            dot = d_Pc_copy[j].nomal_x * d_x + d_Pc_copy[j].nomal_y * d_y + d_Pc_copy[j].nomal_z * d_z;

            f3 = atan2f(d_norm, dot);
            result_F3[tid * nx * ny + j] = int((f3 - g_other.Min_angle) / (g_other.Max_angle - g_other.Min_angle) *
                                               g_other.nAngle);
            result_hash[tid * nx * ny + j] += result_F3[tid * nx * ny + j] * (g_other.g_P3);

            //calculate F4,f4 is the angel_n1_n2
            X4_x = d_Pc[tid].nomal_y * d_Pc_copy[j].nomal_z - d_Pc[tid].nomal_z * d_Pc_copy[j].nomal_y;
            X4_y = d_Pc[tid].nomal_z * d_Pc_copy[j].nomal_x - d_Pc[tid].nomal_x * d_Pc_copy[j].nomal_z;
            X4_z = d_Pc[tid].nomal_x * d_Pc_copy[j].nomal_y - d_Pc[tid].nomal_y * d_Pc_copy[j].nomal_x;
            d_norm = sqrtf(powf(X4_x, 2) + powf(X4_y, 2) + powf(X4_z, 2));

            dot = d_Pc[tid].nomal_x * d_Pc_copy[j].nomal_x + d_Pc[tid].nomal_y * d_Pc_copy[j].nomal_y +
                  d_Pc[tid].nomal_z * d_Pc_copy[j].nomal_z;

            f4 = atan2f(d_norm, dot);
            result_F4[tid * nx * ny + j] = int((f4 - g_other.Min_angle) / (g_other.Max_angle - g_other.Min_angle) *
                                               g_other.nAngle);
            result_hash[tid * nx * ny + j] += result_F4[tid * nx * ny + j] * (g_other.g_P4);

            //calculate hash
            result_hash[tid * nx * ny + j] %= g_other.max_hashIndex;

            //calculate alpha
            result_alpha[tid * nx * ny + j] = createRotAngle(d_Pc[tid], d_Pc_copy[j]);
        }
    }
}

__global__ void
getReferencePpfAndVoting(CudaPPFInfo *d_pointRef, CudaPPFInfo *d_pointScene, int *resultF1, int *resultF2,
                         int *resultF3, int *resultF4, float *resultAlpha, int *resultHash, int nx, int ny, int Num,
                         CudaOtherInfo g_other, PPInfo *d_hashValue, int *d_hashKeyIndex, int *d_accumSpace,
                         int *d_votingPoint, int *d_votingNumber, int *d_votingAngle) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float f1 = 0, f2 = 0, f3 = 0, f4 = 0;
    float d_norm, dot, d_length, d_x = 1, d_y = 1, d_z = 1;
    float X2_x = 1, X2_y = 1, X2_z = 1;
    float X3_x = 1, X3_y = 1, X3_z = 1;
    float X4_x = 1, X4_y = 1, X4_z = 1;

    if (tid < nx * ny) {

        for (int ii = 0; ii < g_other.modelPointsNum; ii++) {
            d_accumSpace[tid * g_other.modelPointsNum + ii] = 0;
        }
        for (int j = 0; j < Num; j++) {
            //calculate F1, f1 is the dis
            f1 = sqrtf(powf((d_pointRef[tid].x - d_pointScene[j].x), 2) +
                       powf((d_pointRef[tid].y - d_pointScene[j].y), 2) +
                       powf((d_pointRef[tid].z - d_pointScene[j].z), 2));
            resultF1[tid * Num + j] = int(f1 / g_other.d_distance);
            resultHash[tid * Num + j] = resultF1[tid * Num + j] * g_other.g_P1;

            //the d
            d_x = d_pointScene[j].x - d_pointRef[tid].x;
            d_y = d_pointScene[j].y - d_pointRef[tid].y;
            d_z = d_pointScene[j].z - d_pointRef[tid].z;

            //calculate F2,and f1 is the angel_d_n1
            //d.normlize
            d_length = sqrtf(powf(d_x, 2) + powf(d_y, 2) + powf(d_z, 2));
            d_x /= d_length;
            d_y /= d_length;
            d_z /= d_length;
            //X and X.norm (X指向量叉积,后期改进再调__device__)
            X2_x = d_y * d_pointRef[tid].nomal_z - d_z * d_pointRef[tid].nomal_y;
            X2_y = d_z * d_pointRef[tid].nomal_x - d_x * d_pointRef[tid].nomal_z;
            X2_z = d_x * d_pointRef[tid].nomal_y - d_y * d_pointRef[tid].nomal_x;
            d_norm = sqrtf(powf(X2_x, 2) + powf(X2_y, 2) + powf(X2_z, 2));

            //dot 点积
            dot = d_x * d_pointRef[tid].nomal_x + d_y * d_pointRef[tid].nomal_y + d_z * d_pointRef[tid].nomal_z;

            f2 = atan2f(d_norm, dot);
            resultF2[tid * Num + j] = int((f2 - g_other.Min_angle) / (g_other.Max_angle - g_other.Min_angle) *
                                          g_other.nAngle);
            resultHash[tid * Num + j] += resultF2[tid * Num + j] * (g_other.g_P2);

            //calculate F3,f3 is the angel_d_n2
            X3_x = d_y * d_pointScene[j].nomal_z - d_z * d_pointScene[j].nomal_y;
            X3_y = d_z * d_pointScene[j].nomal_x - d_x * d_pointScene[j].nomal_z;
            X3_z = d_x * d_pointScene[j].nomal_y - d_y * d_pointScene[j].nomal_x;
            d_norm = sqrtf(powf(X3_x, 2) + powf(X3_y, 2) + powf(X3_z, 2));

            dot = d_pointScene[j].nomal_x * d_x + d_pointScene[j].nomal_y * d_y + d_pointScene[j].nomal_z * d_z;

            f3 = atan2f(d_norm, dot);
            resultF3[tid * Num + j] = int((f3 - g_other.Min_angle) / (g_other.Max_angle - g_other.Min_angle) *
                                          g_other.nAngle);
            resultHash[tid * Num + j] += resultF3[tid * Num + j] * (g_other.g_P3);

            //calculate F4,f4 is the angel_n1_n2
            X4_x = d_pointRef[tid].nomal_y * d_pointScene[j].nomal_z -
                   d_pointRef[tid].nomal_z * d_pointScene[j].nomal_y;
            X4_y = d_pointRef[tid].nomal_z * d_pointScene[j].nomal_x -
                   d_pointRef[tid].nomal_x * d_pointScene[j].nomal_z;
            X4_z = d_pointRef[tid].nomal_x * d_pointScene[j].nomal_y -
                   d_pointRef[tid].nomal_y * d_pointScene[j].nomal_x;
            d_norm = sqrtf(powf(X4_x, 2) + powf(X4_y, 2) + powf(X4_z, 2));

            dot = d_pointRef[tid].nomal_x * d_pointScene[j].nomal_x +
                  d_pointRef[tid].nomal_y * d_pointScene[j].nomal_y +
                  d_pointRef[tid].nomal_z * d_pointScene[j].nomal_z;

            f4 = atan2f(d_norm, dot);
            resultF4[tid * Num + j] = int((f4 - g_other.Min_angle) / (g_other.Max_angle - g_other.Min_angle) *
                                          g_other.nAngle);
            resultHash[tid * Num + j] += resultF4[tid * Num + j] * (g_other.g_P4);

            //calculate hash
            resultHash[tid * Num + j] %= g_other.max_hashIndex;

            //calculate alpha
            resultAlpha[tid * Num + j] = createRotAngle(d_pointRef[tid], d_pointScene[j]);

            //find in the hash table and voting
            if (resultHash[tid * Num + j] > 0 &&
                (resultHash[tid * Num + j] <
                 g_other.hashTableSize - 1)) {         // -1(because we need to use the +1 next,
                // and didn't control the end!)
                int F1 = resultF1[tid * Num + j];
                int F2 = resultF2[tid * Num + j];
                int F3 = resultF3[tid * Num + j];
                int F4 = resultF4[tid * Num + j];
                float alpha = resultAlpha[tid * Num + j];

                int d_listsize = d_hashKeyIndex[resultHash[tid * Num + j] + 1] -        //+1!
                                 d_hashKeyIndex[resultHash[tid * Num + j]];
                for (int k = 0; k < d_listsize; k++) {

                    int cur_F1 = d_hashValue[d_hashKeyIndex[resultHash[tid * Num + j]] + k].F1;
                    int cur_F2 = d_hashValue[d_hashKeyIndex[resultHash[tid * Num + j]] + k].F2;
                    int cur_F3 = d_hashValue[d_hashKeyIndex[resultHash[tid * Num + j]] + k].F3;
                    int cur_F4 = d_hashValue[d_hashKeyIndex[resultHash[tid * Num + j]] + k].F4;
                    float cur_alpha = d_hashValue[d_hashKeyIndex[resultHash[tid * Num + j]] + k].alpha;
                    int pt_id = d_hashValue[d_hashKeyIndex[resultHash[tid * Num + j]] + k].pt1_id;

                    float dis_thresh = 0;
                    float angle_thresh = 0;

                    if (isFeatureSimilar(dis_thresh, angle_thresh, F1, cur_F1, F2, cur_F2, F3, cur_F3,
                                         F4, cur_F4)) {
                        float alpha_m2s = cur_alpha - alpha;
                        if (alpha_m2s < -M_PI)
                            alpha_m2s += M_PI * 2;
                        if (alpha_m2s > M_PI)
                            alpha_m2s -= M_PI * 2;

                        int angleID = (alpha_m2s - g_other.Min_angle) /
                                      (g_other.Max_angle - g_other.Min_angle) * g_other.nAngle;
                        d_accumSpace[tid * g_other.modelPointsNum * g_other.nAngle +
                                     pt_id * g_other.nAngle +
                                     angleID] += 1;
                    }
                }
            }
        }   //from reference point to other points

        int rowLen = g_other.modelPointsNum;
        int colLen = g_other.nAngle;
        int maxAccum = -1;
        int idy_max, idx_max;

        for (int idy = 0; idy < rowLen; idy++) {
            for (int idx = 0; idx < colLen; idx++) {
                int votingValue = d_accumSpace[tid * g_other.modelPointsNum * g_other.nAngle +
                                               idy * g_other.nAngle +
                                               idx];
                if (votingValue > maxAccum) {
                    maxAccum = votingValue;
                    idy_max = idy;
                    idx_max = idx;
                }
            }
        }
        d_votingPoint[tid] = idy_max;   //model point ID
        d_votingAngle[tid] = idx_max;   //rot angle
        d_votingNumber[tid] = maxAccum; //votes number
        //printf("i:%d idy_max:%d MaxAccum:%d Angle:%d\n",tid*5,d_votingPoint[tid],d_votingNumber[tid],d_votingAngle[tid]);
    }   //reference points
}

__global__ void
useVotingResultGetRot(int *d_votingPoint, int *d_votingNumber, int *d_votingAngle, CudaPPFInfo *d_pointRef,
                      CudaPPFInfo *d_pointModel, CudaOtherInfo g_other) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float3 pR = make_float3(d_pointRef[tid].x, d_pointRef[tid].y, d_pointRef[tid].z);
    float3 pM = make_float3(d_pointModel[d_votingPoint[tid]].x, d_pointModel[d_votingPoint[tid]].y,
                            d_pointModel[d_votingPoint[tid]].z);
    float3 pRn = make_float3(d_pointRef[tid].nomal_x, d_pointRef[tid].nomal_y, d_pointRef[tid].nomal_z);
    float3 pMn = make_float3(d_pointModel[d_votingPoint[tid]].nomal_x, d_pointModel[d_votingPoint[tid]].nomal_y,
                             d_pointModel[d_votingPoint[tid]].nomal_z);

    //Matrix4f Tmg = CreateTransformation2AlignNormWithX(pm);
    float3 rotAxisVecModel = make_float3(0, 0, 0);
    float3 xAxisModel = make_float3(1, 0, 0);
    float rotAngleModel = 0;
    //pMn dot xAxisModel
    float cosAngleModel = pMn.x * xAxisModel.x;
    if (fabsf(cosAngleModel - 1) < 0.000001) {
        rotAxisVecModel.x = 1;
    } else if (fabsf(cosAngleModel + 1) < 0.000001) {
        rotAxisVecModel.z = 1;
        rotAngleModel = M_PI;
    } else {
        //normal.corss(x_axis)
        rotAxisVecModel.x = 0;
        rotAxisVecModel.y = pMn.z;
        rotAxisVecModel.z = -pMn.y;
        //rotAxisVec.cross xAxis
        float3 newYVecModel = make_float3(0, rotAxisVecModel.z, -rotAxisVecModel.y);
        //x_part: normal.dot(x_axis)  y_part:normal.dot(newYVec)
        float xPartModel = pMn.x;
        float yPartModel = pMn.y * newYVecModel.y + pMn.z * newYVecModel.z;
        float modelAngleFromXToNormal = atan2f(yPartModel, xPartModel);
        rotAngleModel = -modelAngleFromXToNormal;
    }
    //rotAxisVec.normalise
    float vecLengthModel = sqrtf(powf(rotAxisVecModel.x, 2) + powf(rotAxisVecModel.y, 2) +
                                 powf(rotAxisVecModel.z, 2));
    rotAxisVecModel.x /= vecLengthModel;
    rotAxisVecModel.y /= vecLengthModel;
    rotAxisVecModel.z /= vecLengthModel;
    // compute rotation matrix
    float x = rotAxisVecModel.x;
    float y = rotAxisVecModel.y;
    float z = rotAxisVecModel.z;
    float c = cosf(rotAngleModel);
    float s = sinf(rotAngleModel);
    float v = 1 - c;

    float rotMatModel[4][4];  //Goldman formula
    rotMatModel[0][0] = x * x * v + c;
    rotMatModel[0][1] = x * y * v - z * s;
    rotMatModel[0][2] = x * z * v + y * s;
    rotMatModel[1][0] = x * y * v + z * s;
    rotMatModel[1][1] = y * y * v + c;
    rotMatModel[1][2] = y * z * v - x * s;
    rotMatModel[2][0] = x * z * v - y * s;
    rotMatModel[2][1] = y * z * v + x * s;
    rotMatModel[2][2] = z * z * v + c;
    rotMatModel[0][3] = rotMatModel[1][3] = rotMatModel[2][3] = 0;
    rotMatModel[3][0] = rotMatModel[3][1] = rotMatModel[3][2] = 0;
    rotMatModel[3][3] = 1;
    x = y = z = c = s = v = 0;

    float translationMatModel[4][4];
    //translation_mat.setIdentity();
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (i == j) {
                translationMatModel[i][j] = 1;
            } else {
                translationMatModel[i][j] = 0;
            }
        }
    }
    translationMatModel[0][3] = -pM.x;
    translationMatModel[1][3] = -pM.y;
    translationMatModel[2][3] = -pM.z;
    //rotMatModel * translationMatModel
    float transformMatModel[4][4];      ///This is Tmg
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            transformMatModel[i][j] = 0;
            for (int k = 0; k < 4; k++) {
                transformMatModel[i][j] += rotMatModel[i][k] * translationMatModel[k][j];
            }
        }
    }

    //Matrix4f Tsg = CreateTransformation2AlignNormWithX(ps);
    float3 rotAxisVecRef = make_float3(0, 0, 0);
    float3 xAxisRef = make_float3(1, 0, 0);
    float rotAngleRef = 0;
    //pR.normal dot xAxis
    float cosAngleRef = pRn.x * xAxisRef.x;
    if (fabsf(cosAngleRef - 1) < 0.000001) {
        rotAxisVecRef.x = 1;
    } else if (fabsf(cosAngleRef + 1) < 0.000001) {
        rotAxisVecRef.z = 1;
        rotAngleRef = M_PI;
    } else {
        //normal.corss(x_axis)
        rotAxisVecRef.x = 0;
        rotAxisVecRef.y = pRn.z;
        rotAxisVecRef.z = -pRn.y;
        //rotAxisVec.cross xAxis
        float3 newYVecRef = make_float3(0, rotAxisVecRef.z, -rotAxisVecRef.y);
        //x_part: normal.dot(x_axis)  y_part:normal.dot(newYVec)
        float xPartRef = pRn.x;
        float yPartRef = pRn.y * newYVecRef.y + pRn.z * newYVecRef.z;
        float referenceAngleFromXToNormal = atan2f(yPartRef, xPartRef);
        rotAngleRef = -referenceAngleFromXToNormal;
    }
    //rotAxisVec.normalise
    float vecLengthRef = sqrtf(powf(rotAxisVecRef.x, 2) + powf(rotAxisVecRef.y, 2) +
                               powf(rotAxisVecRef.z, 2));
    rotAxisVecRef.x /= vecLengthRef;
    rotAxisVecRef.y /= vecLengthRef;
    rotAxisVecRef.z /= vecLengthRef;
    // compute rotation matrix
    x = rotAxisVecRef.x;
    y = rotAxisVecRef.y;
    z = rotAxisVecRef.z;
    c = cosf(rotAngleRef);
    s = sinf(rotAngleRef);
    v = 1 - c;

    float rotMatRef[4][4];  //Goldman formula
    rotMatRef[0][0] = x * x * v + c;
    rotMatRef[0][1] = x * y * v - z * s;
    rotMatRef[0][2] = x * z * v + y * s;
    rotMatRef[1][0] = x * y * v + z * s;
    rotMatRef[1][1] = y * y * v + c;
    rotMatRef[1][2] = y * z * v - x * s;
    rotMatRef[2][0] = x * z * v - y * s;
    rotMatRef[2][1] = y * z * v + x * s;
    rotMatRef[2][2] = z * z * v + c;
    rotMatRef[0][3] = rotMatRef[1][3] = rotMatRef[2][3] = 0;
    rotMatRef[3][0] = rotMatRef[3][1] = rotMatRef[3][2] = 0;
    rotMatRef[3][3] = 1;
    x = y = z = c = s = v = 0;

    float translationMatRef[4][4];
    //translation_mat.setIdentity();
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (i == j) {
                translationMatRef[i][j] = 1;
            } else {
                translationMatRef[i][j] = 0;
            }
        }
    }
    translationMatRef[0][3] = -pR.x;
    translationMatRef[1][3] = -pR.y;
    translationMatRef[2][3] = -pR.z;
    //rotMatRef * translationMatRef
    float transformMatRef[4][4];        ///This is Tsg
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            transformMatRef[i][j] = 0;
            for (int k = 0; k < 4; k++) {
                transformMatRef[i][j] += rotMatRef[i][k] * translationMatRef[k][j];
            }
        }
    }

    //Matrix4f Rxa = RotateAboutAnyVector(Vector3f(1, 0, 0), alpha);
    float rotAngle = d_votingAngle[tid] * (g_other.Max_angle - g_other.Min_angle) / g_other.nAngle +
                     g_other.Min_angle;
    //rotAxisVec.normalise
    float3 vecRxa = make_float3(1, 0, 0);
    // compute rotation matrix
    x = vecRxa.x;
    y = vecRxa.y;
    z = vecRxa.z;
    c = cosf(rotAngle);
    s = sinf(rotAngle);
    v = 1 - c;

    float rotMatRxa[4][4];                ///This is Rxa
    rotMatRxa[0][0] = x * x * v + c;
    rotMatRxa[0][1] = x * y * v - z * s;
    rotMatRxa[0][2] = x * z * v + y * s;
    rotMatRxa[1][0] = x * y * v + z * s;
    rotMatRxa[1][1] = y * y * v + c;
    rotMatRxa[1][2] = y * z * v - x * s;
    rotMatRxa[2][0] = x * z * v - y * s;
    rotMatRxa[2][1] = y * z * v + x * s;
    rotMatRxa[2][2] = z * z * v + c;
    rotMatRxa[0][3] = rotMatRxa[1][3] = rotMatRxa[2][3] = 0;
    rotMatRxa[3][0] = rotMatRxa[3][1] = rotMatRxa[3][2] = 0;
    rotMatRxa[3][3] = 1;

    //1.Tsg.inverse()  2.then-->Matrix4f trans_mat = Tsg.inverse() * Rxa * Tmg;
    float detTsg; //hang lie shi
    float invTsg[4][4];   //ni jv zhen
    detTsg = transformMatRef[0][0] * transformMatRef[1][1] * transformMatRef[2][2] * transformMatRef[3][3] -
             transformMatRef[0][0] * transformMatRef[1][1] * transformMatRef[2][3] * transformMatRef[3][2] -
             transformMatRef[0][0] * transformMatRef[1][2] * transformMatRef[2][1] * transformMatRef[3][3] +
             transformMatRef[0][0] * transformMatRef[1][2] * transformMatRef[2][3] * transformMatRef[3][1] +
             transformMatRef[0][0] * transformMatRef[1][3] * transformMatRef[2][1] * transformMatRef[3][2] -
             transformMatRef[0][0] * transformMatRef[1][3] * transformMatRef[2][2] * transformMatRef[3][1] -
             transformMatRef[0][1] * transformMatRef[1][0] * transformMatRef[2][2] * transformMatRef[3][3] +
             transformMatRef[0][1] * transformMatRef[1][0] * transformMatRef[2][3] * transformMatRef[3][2] +
             transformMatRef[0][1] * transformMatRef[1][2] * transformMatRef[2][0] * transformMatRef[3][3] -
             transformMatRef[0][1] * transformMatRef[1][2] * transformMatRef[2][3] * transformMatRef[3][0] -
             transformMatRef[0][1] * transformMatRef[1][3] * transformMatRef[2][0] * transformMatRef[3][2] +
             transformMatRef[0][1] * transformMatRef[1][3] * transformMatRef[2][2] * transformMatRef[3][0] +
             transformMatRef[0][2] * transformMatRef[1][0] * transformMatRef[2][1] * transformMatRef[3][3] -
             transformMatRef[0][2] * transformMatRef[1][0] * transformMatRef[2][3] * transformMatRef[3][1] -
             transformMatRef[0][2] * transformMatRef[1][1] * transformMatRef[2][0] * transformMatRef[3][3] +
             transformMatRef[0][2] * transformMatRef[1][1] * transformMatRef[2][3] * transformMatRef[3][0] +
             transformMatRef[0][2] * transformMatRef[1][3] * transformMatRef[2][0] * transformMatRef[3][1] -
             transformMatRef[0][2] * transformMatRef[1][3] * transformMatRef[2][1] * transformMatRef[3][0] -
             transformMatRef[0][3] * transformMatRef[1][0] * transformMatRef[2][1] * transformMatRef[3][2] +
             transformMatRef[0][3] * transformMatRef[1][0] * transformMatRef[2][2] * transformMatRef[3][1] +
             transformMatRef[0][3] * transformMatRef[1][1] * transformMatRef[2][0] * transformMatRef[3][2] -
             transformMatRef[0][3] * transformMatRef[1][1] * transformMatRef[2][2] * transformMatRef[3][0] -
             transformMatRef[0][3] * transformMatRef[1][2] * transformMatRef[2][0] * transformMatRef[3][1] +
             transformMatRef[0][3] * transformMatRef[1][2] * transformMatRef[2][1] * transformMatRef[3][0];

    if (detTsg == 0) { // can not inverse
        invTsg[0][0] = invTsg[0][1] = invTsg[0][2] = invTsg[0][3] = 0;
        invTsg[1][0] = invTsg[1][1] = invTsg[1][2] = invTsg[1][3] = 0;
        invTsg[2][0] = invTsg[2][1] = invTsg[2][2] = invTsg[2][3] = 0;
        invTsg[3][0] = invTsg[3][1] = invTsg[3][2] = invTsg[3][3] = 0;
    } else {
        float adjTsg[4][4];    //ban sui jv zhen
        adjTsg[0][0] = transformMatRef[1][1] * transformMatRef[2][2] * transformMatRef[3][3] +
                       transformMatRef[2][1] * transformMatRef[3][2] * transformMatRef[1][3] +
                       transformMatRef[3][1] * transformMatRef[1][2] * transformMatRef[2][3] -
                       transformMatRef[3][1] * transformMatRef[2][2] * transformMatRef[1][3] -
                       transformMatRef[2][1] * transformMatRef[1][2] * transformMatRef[3][3] -
                       transformMatRef[1][1] * transformMatRef[3][2] * transformMatRef[2][3];
        adjTsg[1][0] = transformMatRef[1][0] * transformMatRef[2][2] * transformMatRef[3][3] +
                       transformMatRef[1][2] * transformMatRef[2][3] * transformMatRef[3][0] +
                       transformMatRef[1][3] * transformMatRef[2][0] * transformMatRef[3][2] -
                       transformMatRef[1][3] * transformMatRef[2][2] * transformMatRef[3][0] -
                       transformMatRef[1][2] * transformMatRef[2][0] * transformMatRef[3][3] -
                       transformMatRef[1][0] * transformMatRef[2][3] * transformMatRef[3][2];
        adjTsg[2][0] = transformMatRef[1][0] * transformMatRef[2][1] * transformMatRef[3][3] +
                       transformMatRef[1][1] * transformMatRef[2][3] * transformMatRef[3][0] +
                       transformMatRef[1][3] * transformMatRef[2][0] * transformMatRef[3][1] -
                       transformMatRef[1][3] * transformMatRef[2][1] * transformMatRef[3][0] -
                       transformMatRef[1][1] * transformMatRef[2][0] * transformMatRef[3][3] -
                       transformMatRef[1][0] * transformMatRef[2][3] * transformMatRef[3][1];
        adjTsg[3][0] = transformMatRef[1][0] * transformMatRef[2][1] * transformMatRef[3][2] +
                       transformMatRef[1][1] * transformMatRef[2][2] * transformMatRef[3][0] +
                       transformMatRef[1][2] * transformMatRef[2][0] * transformMatRef[3][1] -
                       transformMatRef[1][2] * transformMatRef[2][1] * transformMatRef[3][0] -
                       transformMatRef[1][1] * transformMatRef[2][0] * transformMatRef[3][2] -
                       transformMatRef[1][0] * transformMatRef[2][2] * transformMatRef[3][1];
        adjTsg[0][1] = transformMatRef[0][1] * transformMatRef[2][2] * transformMatRef[3][3] +
                       transformMatRef[0][2] * transformMatRef[2][3] * transformMatRef[3][1] +
                       transformMatRef[0][3] * transformMatRef[2][1] * transformMatRef[3][2] -
                       transformMatRef[0][3] * transformMatRef[2][2] * transformMatRef[3][1] -
                       transformMatRef[0][2] * transformMatRef[2][1] * transformMatRef[3][3] -
                       transformMatRef[0][1] * transformMatRef[2][3] * transformMatRef[3][2];
        adjTsg[1][1] = transformMatRef[0][0] * transformMatRef[2][2] * transformMatRef[3][3] +
                       transformMatRef[0][2] * transformMatRef[2][3] * transformMatRef[3][0] +
                       transformMatRef[0][3] * transformMatRef[2][0] * transformMatRef[3][2] -
                       transformMatRef[0][3] * transformMatRef[2][2] * transformMatRef[3][0] -
                       transformMatRef[0][2] * transformMatRef[2][0] * transformMatRef[3][3] -
                       transformMatRef[0][0] * transformMatRef[2][3] * transformMatRef[3][2];
        adjTsg[2][1] = transformMatRef[0][0] * transformMatRef[2][1] * transformMatRef[3][3] +
                       transformMatRef[0][1] * transformMatRef[2][3] * transformMatRef[3][0] +
                       transformMatRef[0][3] * transformMatRef[2][0] * transformMatRef[3][1] -
                       transformMatRef[0][3] * transformMatRef[2][1] * transformMatRef[3][0] -
                       transformMatRef[0][1] * transformMatRef[2][0] * transformMatRef[3][3] -
                       transformMatRef[0][0] * transformMatRef[2][3] * transformMatRef[3][1];
        adjTsg[3][1] = transformMatRef[0][0] * transformMatRef[2][1] * transformMatRef[3][2] +
                       transformMatRef[0][1] * transformMatRef[2][2] * transformMatRef[3][0] +
                       transformMatRef[0][2] * transformMatRef[2][0] * transformMatRef[3][1] -
                       transformMatRef[0][2] * transformMatRef[2][1] * transformMatRef[3][0] -
                       transformMatRef[0][1] * transformMatRef[2][0] * transformMatRef[3][2] -
                       transformMatRef[0][0] * transformMatRef[2][2] * transformMatRef[3][1];
        adjTsg[0][2] = transformMatRef[0][1] * transformMatRef[1][2] * transformMatRef[3][3] +
                       transformMatRef[0][2] * transformMatRef[1][3] * transformMatRef[3][1] +
                       transformMatRef[0][3] * transformMatRef[1][1] * transformMatRef[3][2] -
                       transformMatRef[0][3] * transformMatRef[1][2] * transformMatRef[3][1] -
                       transformMatRef[0][2] * transformMatRef[1][1] * transformMatRef[3][3] -
                       transformMatRef[0][1] * transformMatRef[1][3] * transformMatRef[3][2];
        adjTsg[1][2] = transformMatRef[0][0] * transformMatRef[1][2] * transformMatRef[3][3] +
                       transformMatRef[0][2] * transformMatRef[1][3] * transformMatRef[3][0] +
                       transformMatRef[0][3] * transformMatRef[1][0] * transformMatRef[3][2] -
                       transformMatRef[0][3] * transformMatRef[1][2] * transformMatRef[3][0] -
                       transformMatRef[0][2] * transformMatRef[1][0] * transformMatRef[3][3] -
                       transformMatRef[0][0] * transformMatRef[1][3] * transformMatRef[3][2];
        adjTsg[2][2] = transformMatRef[0][0] * transformMatRef[1][1] * transformMatRef[3][3] +
                       transformMatRef[0][1] * transformMatRef[1][3] * transformMatRef[3][0] +
                       transformMatRef[0][3] * transformMatRef[1][0] * transformMatRef[3][1] -
                       transformMatRef[0][3] * transformMatRef[1][1] * transformMatRef[3][0] -
                       transformMatRef[0][1] * transformMatRef[1][0] * transformMatRef[3][3] -
                       transformMatRef[0][0] * transformMatRef[1][3] * transformMatRef[3][1];
        adjTsg[3][2] = transformMatRef[0][0] * transformMatRef[1][1] * transformMatRef[3][2] +
                       transformMatRef[0][1] * transformMatRef[1][2] * transformMatRef[3][0] +
                       transformMatRef[0][2] * transformMatRef[1][0] * transformMatRef[3][1] -
                       transformMatRef[0][2] * transformMatRef[1][1] * transformMatRef[3][0] -
                       transformMatRef[0][1] * transformMatRef[1][0] * transformMatRef[3][2] -
                       transformMatRef[0][0] * transformMatRef[1][2] * transformMatRef[3][1];
        adjTsg[0][3] = transformMatRef[0][1] * transformMatRef[1][2] * transformMatRef[2][3] +
                       transformMatRef[0][2] * transformMatRef[1][3] * transformMatRef[2][1] +
                       transformMatRef[0][3] * transformMatRef[1][1] * transformMatRef[2][2] -
                       transformMatRef[0][3] * transformMatRef[1][2] * transformMatRef[2][1] -
                       transformMatRef[0][2] * transformMatRef[1][1] * transformMatRef[2][3] -
                       transformMatRef[0][1] * transformMatRef[1][3] * transformMatRef[2][2];
        adjTsg[1][3] = transformMatRef[0][0] * transformMatRef[1][2] * transformMatRef[2][3] +
                       transformMatRef[0][2] * transformMatRef[1][3] * transformMatRef[2][0] +
                       transformMatRef[0][3] * transformMatRef[1][0] * transformMatRef[2][2] -
                       transformMatRef[0][3] * transformMatRef[1][2] * transformMatRef[2][0] -
                       transformMatRef[0][2] * transformMatRef[1][0] * transformMatRef[2][3] -
                       transformMatRef[0][0] * transformMatRef[1][3] * transformMatRef[2][2];
        adjTsg[2][3] = transformMatRef[0][0] * transformMatRef[1][1] * transformMatRef[2][3] +
                       transformMatRef[0][1] * transformMatRef[1][3] * transformMatRef[2][0] +
                       transformMatRef[0][3] * transformMatRef[1][0] * transformMatRef[2][1] -
                       transformMatRef[0][3] * transformMatRef[1][1] * transformMatRef[2][0] -
                       transformMatRef[0][1] * transformMatRef[1][0] * transformMatRef[2][3] -
                       transformMatRef[0][0] * transformMatRef[1][3] * transformMatRef[2][1];
        adjTsg[3][3] = transformMatRef[0][0] * transformMatRef[1][1] * transformMatRef[2][2] +
                       transformMatRef[0][1] * transformMatRef[1][2] * transformMatRef[2][0] +
                       transformMatRef[0][2] * transformMatRef[1][0] * transformMatRef[2][1] -
                       transformMatRef[0][2] * transformMatRef[1][1] * transformMatRef[2][0] -
                       transformMatRef[0][1] * transformMatRef[1][0] * transformMatRef[2][2] -
                       transformMatRef[0][0] * transformMatRef[1][2] * transformMatRef[2][1];

        invTsg[0][0] = adjTsg[0][0] / detTsg;
        invTsg[1][0] = -adjTsg[1][0] / detTsg;
        invTsg[2][0] = adjTsg[2][0] / detTsg;
        invTsg[3][0] = -adjTsg[3][0] / detTsg;
        invTsg[0][1] = -adjTsg[0][1] / detTsg;
        invTsg[1][1] = adjTsg[1][1] / detTsg;
        invTsg[2][1] = -adjTsg[2][1] / detTsg;
        invTsg[3][1] = adjTsg[3][1] / detTsg;
        invTsg[0][2] = adjTsg[0][2] / detTsg;
        invTsg[1][2] = -adjTsg[1][2] / detTsg;
        invTsg[2][2] = adjTsg[2][2] / detTsg;
        invTsg[3][2] = -adjTsg[3][2] / detTsg;
        invTsg[0][3] = -adjTsg[0][3] / detTsg;
        invTsg[1][3] = adjTsg[1][3] / detTsg;
        invTsg[2][3] = -adjTsg[2][3] / detTsg;
        invTsg[3][3] = adjTsg[3][3] / detTsg;
    }
    //Matrix4f trans_mat = Tsg.inverse() * Rxa * Tmg;
    float transMat[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            transMat[i][j] = 0;
            for (int k = 0; k < 4; k++) {
                transMat[i][j] += invTsg[i][k] * rotMatRxa[k][j];
            }
        }
    }
    float trans_Mat[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            trans_Mat[i][j] = 0;
            for (int k = 0; k < 4; k++) {
                trans_Mat[i][j] += transMat[i][k] * transformMatModel[k][j];
            }
        }
    }


}

__global__ void
scenePPF(CudaPPFInfo *d_pointRef, CudaPPFInfo *d_pointScene, int *resultF1, int *resultF2,
         int *resultF3, int *resultF4, float *resultAlpha, int *resultHash, int nx, int ny, int Num,
         CudaOtherInfo g_other) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float f1 = 0, f2 = 0, f3 = 0, f4 = 0;
    float d_norm, dot, d_length, d_x = 1, d_y = 1, d_z = 1;
    float X2_x = 1, X2_y = 1, X2_z = 1;
    float X3_x = 1, X3_y = 1, X3_z = 1;
    float X4_x = 1, X4_y = 1, X4_z = 1;

    if (tid < nx * ny) {
        for (int j = 0; j < Num; j++) {
            //calculate F1, f1 is the dis
            f1 = sqrtf(powf((d_pointRef[tid].x - d_pointScene[j].x), 2) +
                       powf((d_pointRef[tid].y - d_pointScene[j].y), 2) +
                       powf((d_pointRef[tid].z - d_pointScene[j].z), 2));
            resultF1[tid * Num + j] = int(f1 / g_other.d_distance);
            resultHash[tid * Num + j] = resultF1[tid * Num + j] * g_other.g_P1;

            //the d
            d_x = d_pointScene[j].x - d_pointRef[tid].x;
            d_y = d_pointScene[j].y - d_pointRef[tid].y;
            d_z = d_pointScene[j].z - d_pointRef[tid].z;

            //calculate F2,and f1 is the angel_d_n1
            //d.normlize
            d_length = sqrtf(powf(d_x, 2) + powf(d_y, 2) + powf(d_z, 2));
            d_x /= d_length;
            d_y /= d_length;
            d_z /= d_length;
            //X and X.norm (X指向量叉积,后期改进再调__device__)
            X2_x = d_y * d_pointRef[tid].nomal_z - d_z * d_pointRef[tid].nomal_y;
            X2_y = d_z * d_pointRef[tid].nomal_x - d_x * d_pointRef[tid].nomal_z;
            X2_z = d_x * d_pointRef[tid].nomal_y - d_y * d_pointRef[tid].nomal_x;
            d_norm = sqrtf(powf(X2_x, 2) + powf(X2_y, 2) + powf(X2_z, 2));

            //dot 点积
            dot = d_x * d_pointRef[tid].nomal_x + d_y * d_pointRef[tid].nomal_y + d_z * d_pointRef[tid].nomal_z;

            f2 = atan2f(d_norm, dot);
            resultF2[tid * Num + j] = int((f2 - g_other.Min_angle) / (g_other.Max_angle - g_other.Min_angle) *
                                          g_other.nAngle);
            resultHash[tid * Num + j] += resultF2[tid * Num + j] * (g_other.g_P2);

            //calculate F3,f3 is the angel_d_n2
            X3_x = d_y * d_pointScene[j].nomal_z - d_z * d_pointScene[j].nomal_y;
            X3_y = d_z * d_pointScene[j].nomal_x - d_x * d_pointScene[j].nomal_z;
            X3_z = d_x * d_pointScene[j].nomal_y - d_y * d_pointScene[j].nomal_x;
            d_norm = sqrtf(powf(X3_x, 2) + powf(X3_y, 2) + powf(X3_z, 2));

            dot = d_pointScene[j].nomal_x * d_x + d_pointScene[j].nomal_y * d_y + d_pointScene[j].nomal_z * d_z;

            f3 = atan2f(d_norm, dot);
            resultF3[tid * Num + j] = int((f3 - g_other.Min_angle) / (g_other.Max_angle - g_other.Min_angle) *
                                          g_other.nAngle);
            resultHash[tid * Num + j] += resultF3[tid * Num + j] * (g_other.g_P3);

            //calculate F4,f4 is the angel_n1_n2
            X4_x = d_pointRef[tid].nomal_y * d_pointScene[j].nomal_z -
                   d_pointRef[tid].nomal_z * d_pointScene[j].nomal_y;
            X4_y = d_pointRef[tid].nomal_z * d_pointScene[j].nomal_x -
                   d_pointRef[tid].nomal_x * d_pointScene[j].nomal_z;
            X4_z = d_pointRef[tid].nomal_x * d_pointScene[j].nomal_y -
                   d_pointRef[tid].nomal_y * d_pointScene[j].nomal_x;
            d_norm = sqrtf(powf(X4_x, 2) + powf(X4_y, 2) + powf(X4_z, 2));

            dot = d_pointRef[tid].nomal_x * d_pointScene[j].nomal_x +
                  d_pointRef[tid].nomal_y * d_pointScene[j].nomal_y +
                  d_pointRef[tid].nomal_z * d_pointScene[j].nomal_z;

            f4 = atan2f(d_norm, dot);
            resultF4[tid * Num + j] = int((f4 - g_other.Min_angle) / (g_other.Max_angle - g_other.Min_angle) *
                                          g_other.nAngle);
            resultHash[tid * Num + j] += resultF4[tid * Num + j] * (g_other.g_P4);

            //calculate hash
            resultHash[tid * Num + j] %= g_other.max_hashIndex;

            //calculate alpha
            resultAlpha[tid * Num + j] = createRotAngle(d_pointRef[tid], d_pointScene[j]);

        }   //from reference point to other points
    }   //reference points
}

__global__ void
vote(int *resultF1, int *resultF2, int *resultF3, int *resultF4, float *resultAlpha, int *resultHash, int nx, int ny,
     int Num, CudaOtherInfo g_other, PPInfo *d_hashValue, int *d_hashKeyIndex, int *d_accumSpace) {

    int num = blockDim.x * gridDim.x;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = blockIdx.y * num + j;

    if(j < 10091){
        //find in the hash table and voting
        if (resultHash[blockIdx.x* Num + j] > 0 &&
            (resultHash[blockIdx.x * Num + j] < g_other.hashTableSize - 1)) {         // -1(because we need to use the +1 next,
            // and didn't control the end!)
            int F1 = resultF1[blockIdx.x * Num + j];
            int F2 = resultF2[blockIdx.x * Num + j];
            int F3 = resultF3[blockIdx.x * Num + j];
            int F4 = resultF4[blockIdx.x * Num + j];
            float alpha = resultAlpha[blockIdx.x * Num + j];

            int d_listsize = d_hashKeyIndex[resultHash[blockIdx.x * Num + j] + 1] -        //+1!
                             d_hashKeyIndex[resultHash[blockIdx.x * Num + j]];
            for (int k = 0; k < d_listsize; k++) {

                int cur_F1 = d_hashValue[d_hashKeyIndex[resultHash[blockIdx.x * Num + j]] + k].F1;
                int cur_F2 = d_hashValue[d_hashKeyIndex[resultHash[blockIdx.x * Num + j]] + k].F2;
                int cur_F3 = d_hashValue[d_hashKeyIndex[resultHash[blockIdx.x * Num + j]] + k].F3;
                int cur_F4 = d_hashValue[d_hashKeyIndex[resultHash[blockIdx.x * Num + j]] + k].F4;
                float cur_alpha = d_hashValue[d_hashKeyIndex[resultHash[blockIdx.x * Num + j]] + k].alpha;
                int pt_id = d_hashValue[d_hashKeyIndex[resultHash[blockIdx.x * Num + j]] + k].pt1_id;

                float dis_thresh = 0;
                float angle_thresh = 0;

                if (isFeatureSimilar(dis_thresh, angle_thresh, F1, cur_F1, F2, cur_F2, F3, cur_F3,
                                     F4, cur_F4)) {
                    float alpha_m2s = cur_alpha - alpha;
                    if (alpha_m2s < -M_PI)
                        alpha_m2s += M_PI * 2;
                    if (alpha_m2s > M_PI)
                        alpha_m2s -= M_PI * 2;

                    int angleID = (alpha_m2s - g_other.Min_angle) /
                                  (g_other.Max_angle - g_other.Min_angle) * g_other.nAngle;

                    atomicAdd(&d_accumSpace[blockIdx.y * g_other.modelPointsNum * g_other.nAngle + pt_id * g_other.nAngle +
                                            angleID], 1);
                }
            }
        }
    }
}

__global__ void
add(int *d_accumSpace, int *d_votingPoint, int *d_votingNumber, int *d_votingAngle, CudaOtherInfo g_other){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int rowLen = g_other.modelPointsNum;
    int colLen = g_other.nAngle;
    int maxAccum = -1;
    int idy_max, idx_max;

    if(tid<2019){
        for (int idy = 0; idy < rowLen; idy++) {
            for (int idx = 0; idx < colLen; idx++) {
                int votingValue = d_accumSpace[tid * g_other.modelPointsNum * g_other.nAngle +
                                               idy * g_other.nAngle +
                                               idx];
                if (votingValue > maxAccum) {
                    maxAccum = votingValue;
                    idy_max = idy;
                    idx_max = idx;
                }
            }
        }
        d_votingPoint[tid] = idy_max;   //model point ID
        d_votingAngle[tid] = idx_max;   //rot angle
        d_votingNumber[tid] = maxAccum; //votes number
        //printf("i:%d idy_max:%d MaxAccum:%d Angle:%d\n",tid*5,d_votingPoint[tid],d_votingNumber[tid],d_votingAngle[tid]);
    }
}

void modelPpf(CudaPPFInfo *h_Pc, CudaPPFInfo *h_Pc_copy, CudaOtherInfo h_other, int *h_hash, float *h_alpha,
              int *h_F1, int *h_F2, int *h_F3, int *h_F4) {
    cout << "GPUTraining start..." << endl;
    cudaSetDevice(0);
    int nx = 32;
    int ny = 128;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(CudaPPFInfo);

    CudaPPFInfo *d_Pc = NULL;
    CudaPPFInfo *d_Pc_copy = NULL;
    int *d_result_F1 = NULL;
    int *d_result_F2 = NULL;
    int *d_result_F3 = NULL;
    int *d_result_F4 = NULL;
    float *d_result_alpha = NULL;
    int *d_result_hash = NULL;

    cudaMalloc((void **) &d_Pc, nBytes);                                   ///test to change void as cuda_PPinfo
    cudaMalloc((void **) &d_Pc_copy, nBytes);

    cudaMalloc((int **) &d_result_F1, sizeof(int) * nxy * nxy);
    cudaMalloc((int **) &d_result_F2, sizeof(int) * nxy * nxy);
    cudaMalloc((int **) &d_result_F3, sizeof(int) * nxy * nxy);
    cudaMalloc((int **) &d_result_F4, sizeof(int) * nxy * nxy);
    cudaMalloc((float **) &d_result_alpha, sizeof(float) * nxy * nxy);
    cudaMalloc((int **) &d_result_hash, sizeof(int) * nxy * nxy);

    cudaMemcpy(d_Pc, h_Pc, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Pc_copy, h_Pc_copy, nBytes, cudaMemcpyHostToDevice);

    /*// start cuda timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);*/

    dim3 grid(ny);
    dim3 block(nx);

    modelPpf << < grid, block >> >
                        (d_Pc, d_Pc_copy, d_result_F1, d_result_F2, d_result_F3, d_result_F4, d_result_alpha, d_result_hash, nx, ny, h_other);

    /*// end cuda timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout <<"Time to generate PPFs: "<<elapsedTime<<" ms" <<endl;*/

    cudaDeviceSynchronize();

    cudaMemcpy(h_hash, d_result_hash, sizeof(int) * nxy * nxy, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_alpha, d_result_alpha, sizeof(float) * nxy * nxy, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_F1, d_result_F1, sizeof(int) * nxy * nxy, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_F2, d_result_F2, sizeof(int) * nxy * nxy, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_F3, d_result_F3, sizeof(int) * nxy * nxy, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_F4, d_result_F4, sizeof(int) * nxy * nxy, cudaMemcpyDeviceToHost);

    cudaFree(d_Pc);
    cudaFree(d_Pc_copy);
    cudaFree(d_result_F1);
    cudaFree(d_result_F2);
    cudaFree(d_result_F3);
    cudaFree(d_result_F4);
    cudaFree(d_result_alpha);
    cudaFree(d_result_hash);
    cudaDeviceReset();
}

void voting(CudaPPFInfo *pointRef, CudaPPFInfo *pointScene, CudaOtherInfo h_other, int *h_hash, float *h_alpha,
            int *h_F1, int *h_F2, int *h_F3, int *h_F4, PPInfo *modelHashValue, int *modelHashKeyIndex,
            CudaPPFInfo *pointModel,
            int *votingPoint, int *votingAngle, int *votingNumber) {
    cout << "GPU Voting start..." << endl;
    cudaSetDevice(0);
    int nx = 128;
    int ny = 16;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(CudaPPFInfo);
    int nu = 10091;
    int NumBytes = nu * sizeof(CudaPPFInfo);

    CudaPPFInfo *d_pointRef = NULL;
    CudaPPFInfo *d_pointScene = NULL;
    int *d_resultF1 = NULL;
    int *d_resultF2 = NULL;
    int *d_resultF3 = NULL;
    int *d_resultF4 = NULL;
    float *d_resultAlpha = NULL;
    int *d_resultHash = NULL;
    //points pair
    cudaMalloc((void **) &d_pointRef, nBytes);
    cudaMalloc((void **) &d_pointScene, NumBytes);
    cudaMemcpy(d_pointRef, pointRef, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pointScene, pointScene, NumBytes, cudaMemcpyHostToDevice);
    //point pair feature
    cudaMalloc((int **) &d_resultF1, sizeof(int) * nxy * nu);
    cudaMalloc((int **) &d_resultF2, sizeof(int) * nxy * nu);
    cudaMalloc((int **) &d_resultF3, sizeof(int) * nxy * nu);
    cudaMalloc((int **) &d_resultF4, sizeof(int) * nxy * nu);
    cudaMalloc((float **) &d_resultAlpha, sizeof(float) * nxy * nu);
    cudaMalloc((int **) &d_resultHash, sizeof(int) * nxy * nu);
    //calculate scene ppf
    dim3 block(nx);
    dim3 grid(ny);
    scenePPF <<< grid, block >>>
                        (d_pointRef, d_pointScene, d_resultF1, d_resultF2, d_resultF3, d_resultF4, d_resultAlpha,
                                d_resultHash, nx, ny, nu, h_other);
    cudaDeviceSynchronize();

    //hash table
    PPInfo *d_hashValue = NULL;
    cudaMalloc((void **) &d_hashValue, sizeof(PPInfo) * h_other.hashNum);
    cudaMemcpy(d_hashValue, modelHashValue, sizeof(PPInfo) * h_other.hashNum, cudaMemcpyHostToDevice);
    int *d_hashKeyIndex = NULL;
    cudaMalloc((int **) &d_hashKeyIndex, sizeof(int) * h_other.hashTableSize);
    cudaMemcpy(d_hashKeyIndex, modelHashKeyIndex, sizeof(int) * h_other.hashTableSize, cudaMemcpyHostToDevice);
    //accumSpace
    int *d_accumSpace = NULL;
    cudaMalloc((int **) &d_accumSpace, sizeof(int) * nxy * h_other.modelPointsNum * h_other.nAngle);

    block.x = 1024;
    grid.x = 10;
    grid.y = 2019;
    cout << "bolck size :" << block.x << " grid.x:" << grid.x << " grid.y:" << grid.y << endl;
    vote<<<grid, block>>>(d_resultF1, d_resultF2, d_resultF3, d_resultF4, d_resultAlpha, d_resultHash, nx, ny,
            nu, h_other, d_hashValue, d_hashKeyIndex, d_accumSpace);

    //voting result (point,votesNumber,angle)
    int *d_votingPoint = NULL;
    cudaMalloc((int **) &d_votingPoint, sizeof(int) * nxy);
    int *d_votingNumber = NULL;
    cudaMalloc((int **) &d_votingNumber, sizeof(int) * nxy);
    int *d_votingAngle = NULL;
    cudaMalloc((int **) &d_votingAngle, sizeof(int) * nxy);

    add<<<ny,nx>>>(d_accumSpace,d_votingPoint, d_votingNumber, d_votingAngle,h_other);

    cudaMemcpy(votingPoint, d_votingPoint, sizeof(int) * nxy, cudaMemcpyDeviceToHost);
    cudaMemcpy(votingAngle, d_votingAngle, sizeof(int) * nxy, cudaMemcpyDeviceToHost);
    cudaMemcpy(votingNumber, d_votingNumber, sizeof(int) * nxy, cudaMemcpyDeviceToHost);


    //useVotingResultGetRot<<<grid,block>>>(d_votingPoint, d_votingNumber ,d_votingAngle, d_pointRef, d_pointModel, h_other);

    cudaFree(d_pointRef);
    cudaFree(d_pointScene);
    cudaFree(d_resultF1);
    cudaFree(d_resultF2);
    cudaFree(d_resultF3);
    cudaFree(d_resultF4);
    cudaFree(d_resultAlpha);
    cudaFree(d_resultHash);
    cudaFree(d_hashValue);
    cudaFree(d_hashKeyIndex);
    cudaFree(d_accumSpace);
    cudaFree(d_votingPoint);
    cudaFree(d_votingAngle);
    cudaFree(d_votingNumber);
    cudaDeviceReset();
}

void getPpfAndVoting(CudaPPFInfo *pointRef, CudaPPFInfo *pointScene, CudaOtherInfo h_other, int *h_hash, float *h_alpha,
                     int *h_F1, int *h_F2, int *h_F3, int *h_F4, PPInfo *modelHashValue, int *modelHashKeyIndex,
                     CudaPPFInfo *pointModel,
                     int *votingPoint, int *votingAngle, int *votingNumber) {

    struct timeval timeEnd, timeMiddle, timeSystemStart;
    double systemRunTime;
    gettimeofday(&timeSystemStart, NULL);

    cout << "GPU Voting start..." << endl;
    cudaSetDevice(0);
    int nx = 128;
    int ny = 16;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(CudaPPFInfo);
    int nu = 10091;
    int NumBytes = nu * sizeof(CudaPPFInfo);

    CudaPPFInfo *d_pointRef = NULL;
    CudaPPFInfo *d_pointScene = NULL;
    CudaPPFInfo *d_pointModel = NULL;
    int *d_resultF1 = NULL;
    int *d_resultF2 = NULL;
    int *d_resultF3 = NULL;
    int *d_resultF4 = NULL;
    float *d_resultAlpha = NULL;
    int *d_resultHash = NULL;
    //points pair
    cudaMalloc((void **) &d_pointRef, nBytes);
    cudaMalloc((void **) &d_pointScene, NumBytes);
    cudaMalloc((void **) &d_pointModel, h_other.modelPointsNum * sizeof(CudaPPFInfo));
    cudaMemcpy(d_pointRef, pointRef, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pointScene, pointScene, NumBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pointModel, pointModel, h_other.modelPointsNum * sizeof(CudaPPFInfo), cudaMemcpyHostToDevice);
    //point pair feature
    cudaMalloc((int **) &d_resultF1, sizeof(int) * nxy * nu);
    cudaMalloc((int **) &d_resultF2, sizeof(int) * nxy * nu);
    cudaMalloc((int **) &d_resultF3, sizeof(int) * nxy * nu);
    cudaMalloc((int **) &d_resultF4, sizeof(int) * nxy * nu);
    cudaMalloc((float **) &d_resultAlpha, sizeof(float) * nxy * nu);
    cudaMalloc((int **) &d_resultHash, sizeof(int) * nxy * nu);
    //hash table
    PPInfo *d_hashValue = NULL;
    cudaMalloc((void **) &d_hashValue, sizeof(PPInfo) * h_other.hashNum);
    cudaMemcpy(d_hashValue, modelHashValue, sizeof(PPInfo) * h_other.hashNum, cudaMemcpyHostToDevice);
    int *d_hashKeyIndex = NULL;
    cudaMalloc((int **) &d_hashKeyIndex, sizeof(int) * h_other.hashTableSize);
    cudaMemcpy(d_hashKeyIndex, modelHashKeyIndex, sizeof(int) * h_other.hashTableSize, cudaMemcpyHostToDevice);
    //accumSpace
    int *d_accumSpace = NULL;
    cudaMalloc((int **) &d_accumSpace, sizeof(int) * nxy * h_other.modelPointsNum * h_other.nAngle);
    //voting result (point,votesNumber,angle)
    int *d_votingPoint = NULL;
    cudaMalloc((int **) &d_votingPoint, sizeof(int) * nxy);
    int *d_votingNumber = NULL;
    cudaMalloc((int **) &d_votingNumber, sizeof(int) * nxy);
    int *d_votingAngle = NULL;
    cudaMalloc((int **) &d_votingAngle, sizeof(int) * nxy);

    gettimeofday(&timeMiddle, NULL);
    systemRunTime = (timeMiddle.tv_sec - timeSystemStart.tv_sec) +
                    (double) (timeMiddle.tv_usec - timeSystemStart.tv_usec) / 1000000;
    cout << "the time usage of init memory is:" << systemRunTime << endl;

    dim3 grid(ny);
    dim3 block(nx);

    getReferencePpfAndVoting << < grid, block >> >
                                        (d_pointRef, d_pointScene, d_resultF1, d_resultF2, d_resultF3, d_resultF4,
                                                d_resultAlpha, d_resultHash, nx, ny, nu, h_other, d_hashValue, d_hashKeyIndex,
                                                d_accumSpace, d_votingPoint, d_votingNumber, d_votingAngle);
    //cudaDeviceSynchronize();


    cudaMemcpy(votingPoint, d_votingPoint, sizeof(int) * nxy, cudaMemcpyDeviceToHost);
    cudaMemcpy(votingAngle, d_votingAngle, sizeof(int) * nxy, cudaMemcpyDeviceToHost);
    cudaMemcpy(votingNumber, d_votingNumber, sizeof(int) * nxy, cudaMemcpyDeviceToHost);

    useVotingResultGetRot << < grid, block >> >
                                     (d_votingPoint, d_votingNumber, d_votingAngle, d_pointRef, d_pointModel, h_other);

    cudaFree(d_pointRef);
    cudaFree(d_pointScene);
    cudaFree(d_resultF1);
    cudaFree(d_resultF2);
    cudaFree(d_resultF3);
    cudaFree(d_resultF4);
    cudaFree(d_resultAlpha);
    cudaFree(d_resultHash);
    cudaFree(d_hashValue);
    cudaFree(d_hashKeyIndex);
    cudaFree(d_accumSpace);
    cudaFree(d_votingPoint);
    cudaFree(d_votingNumber);
    cudaFree(d_votingAngle);
    cudaDeviceReset();

    gettimeofday(&timeEnd, NULL);
    systemRunTime =
            (timeEnd.tv_sec - timeSystemStart.tv_sec) + (double) (timeEnd.tv_usec - timeSystemStart.tv_usec) / 1000000;
    cout << "the time usage of GPU voting is:" << systemRunTime << endl;
}
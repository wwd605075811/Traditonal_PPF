#include "kernel.cuh"

__global__ void kernelA(){
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

    if(globalThreadId==1){
        printf("KernelA\n");
    }

    printf("My threadIdx.x is %d, blockIdx.x is %d, blockDim.x is %d, Global thread id is %d\n",
            threadIdx.x, blockIdx.x, blockDim.x, globalThreadId);
}

__device__ float g_CreateRotAngle(cuda_PPFInfo p1,cuda_PPFInfo p2) {
    float rot_angle = 0;
    float rotAxisVec_x = 0, rotAxisVec_y = 0, rotAxisVec_z = 0;
    float newYVec_x = 0, newYVec_y = 0, newYVec_z = 0;
    //p1_normal dot X axis
    float cosAngle = p1.nomal_x * 1;
    if (fabsf(cosAngle - 1) < 0.000001) {
        rotAxisVec_x = 1;
    } else if (fabsf(cosAngle + 1) < 0.000001) {
        rotAxisVec_z = 1;
        rot_angle = 3.141592653;
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
    float vec_length = sqrtf(powf(rotAxisVec_x, 2) + powf(rotAxisVec_y, 2) + powf(rotAxisVec_z, 2));
    rotAxisVec_x /= vec_length;
    rotAxisVec_y /= vec_length;
    rotAxisVec_z /= vec_length;
    // compute rotation matrix
    float x=rotAxisVec_x;
    float y=rotAxisVec_y;
    float z=rotAxisVec_z;
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
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            if(i==j){
                translation_mat[i][j]=1;
            }
            else{
                translation_mat[i][j]=0;
            }
        }
    }
    translation_mat[0][3]= -p1.x;
    translation_mat[1][3]= -p1.y;
    translation_mat[2][3]= -p1.z;

    float transform_mat[4][4];
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            transform_mat[i][j]=0;
            for(int k=0;k<4;k++){
                transform_mat[i][j]+=rot_mat[i][k]*translation_mat[k][j];
            }
        }
    }
    float trans_p2_x=p2.x;
    float trans_p2_y=p2.y;
    float trans_p2_z=p2.z;
    float trans_p2_u=1.0;
    float rot_mat_y=0;
    float rot_mat_z=0;
    rot_mat_y=transform_mat[1][0]*trans_p2_x + transform_mat[1][1]*trans_p2_y + transform_mat[1][2]*trans_p2_z + transform_mat[1][3]*trans_p2_u;
    rot_mat_z=transform_mat[2][0]*trans_p2_x + transform_mat[2][1]*trans_p2_y + transform_mat[2][2]*trans_p2_z + transform_mat[2][3]*trans_p2_u;

    return atan2f(rot_mat_y,rot_mat_z);
}

__global__ void kernelA_1d(cuda_PPFInfo* d_Pc, cuda_PPFInfo* d_Pc_copy, float* result_F1,float* result_F2, float* result_F3,float* result_F4,int* result_hash,int nx,cuda_PPFotherInfo g_other){

    int tid =threadIdx.x ;
    printf("tid:%d\n",tid);
    float d_norm,dot,d_length,d_x=1,d_y=1,d_z=1;
    float X2_x=1,X2_y=1,X2_z=1;  //middle cha ji value
    float X3_x=1,X3_y=1,X3_z=1;
    float X4_x=1,X4_y=1,X4_z=1;

    //printf("blockIdx.x:%d * blockDim.x:%d + threadIdx.x:%d = tid:%d\n", blockIdx.x, blockDim.y, threadIdx.x, tid);
    if(tid<nx) {
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
kernelA_2dimension(cuda_PPFInfo *d_Pc, cuda_PPFInfo *d_Pc_copy, int *result_F1, int *result_F2, int *result_F3,
                   int *result_F4, float *result_alpha, int *result_hash, int nx, int ny, cuda_PPFotherInfo g_other) {

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
            result_alpha[tid * nx * ny + j] = g_CreateRotAngle(d_Pc[tid], d_Pc_copy[j]);
        }
    }
}

__global__ void
scene_ppf(cuda_PPFInfo *d_Pc, cuda_PPFInfo *d_Pc_Refer, int *result_F1, int *result_F2, int *result_F3,
                   int *result_F4, float *result_alpha, int *result_hash, int nxy, int nRefer, cuda_PPFotherInfo g_other) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float f1 = 0, f2 = 0, f3 = 0, f4 = 0;
    float d_norm, dot, d_length, d_x = 1, d_y = 1, d_z = 1;
    float X2_x = 1, X2_y = 1, X2_z = 1;
    float X3_x = 1, X3_y = 1, X3_z = 1;
    float X4_x = 1, X4_y = 1, X4_z = 1;

    if (tid < nxy) {
        for (int j = 0; j < nRefer; j++) {
            //calculate F1, f1 is the dis
            f1 = sqrtf(powf((d_Pc[tid].x - d_Pc_Refer[j].x), 2) + powf((d_Pc[tid].y - d_Pc_Refer[j].y), 2) +
                       powf((d_Pc[tid].z - d_Pc_Refer[j].z), 2));
            result_F1[tid * nRefer + j] = int(f1 / g_other.d_distance);
            result_hash[tid * nRefer + j] = result_F1[tid * nRefer + j] * g_other.g_P1;

            //the d
            d_x = d_Pc_Refer[j].x - d_Pc[tid].x;
            d_y = d_Pc_Refer[j].y - d_Pc[tid].y;
            d_z = d_Pc_Refer[j].z - d_Pc[tid].z;

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
            result_F2[tid * nRefer + j] = int((f2 - g_other.Min_angle) / (g_other.Max_angle - g_other.Min_angle) *
                                               g_other.nAngle);
            result_hash[tid * nRefer + j] += result_F2[tid * nRefer + j] * (g_other.g_P2);

            //calculate F3,f3 is the angel_d_n2
            X3_x = d_y * d_Pc_Refer[j].nomal_z - d_z * d_Pc_Refer[j].nomal_y;
            X3_y = d_z * d_Pc_Refer[j].nomal_x - d_x * d_Pc_Refer[j].nomal_z;
            X3_z = d_x * d_Pc_Refer[j].nomal_y - d_y * d_Pc_Refer[j].nomal_x;
            d_norm = sqrtf(powf(X3_x, 2) + powf(X3_y, 2) + powf(X3_z, 2));

            dot = d_Pc_Refer[j].nomal_x * d_x + d_Pc_Refer[j].nomal_y * d_y + d_Pc_Refer[j].nomal_z * d_z;

            f3 = atan2f(d_norm, dot);
            result_F3[tid * nRefer + j] = int((f3 - g_other.Min_angle) / (g_other.Max_angle - g_other.Min_angle) *
                                               g_other.nAngle);
            result_hash[tid * nRefer + j] += result_F3[tid * nRefer + j] * (g_other.g_P3);

            //calculate F4,f4 is the angel_n1_n2
            X4_x = d_Pc[tid].nomal_y * d_Pc_Refer[j].nomal_z - d_Pc[tid].nomal_z * d_Pc_Refer[j].nomal_y;
            X4_y = d_Pc[tid].nomal_z * d_Pc_Refer[j].nomal_x - d_Pc[tid].nomal_x * d_Pc_Refer[j].nomal_z;
            X4_z = d_Pc[tid].nomal_x * d_Pc_Refer[j].nomal_y - d_Pc[tid].nomal_y * d_Pc_Refer[j].nomal_x;
            d_norm = sqrtf(powf(X4_x, 2) + powf(X4_y, 2) + powf(X4_z, 2));

            dot = d_Pc[tid].nomal_x * d_Pc_Refer[j].nomal_x + d_Pc[tid].nomal_y * d_Pc_Refer[j].nomal_y +
                  d_Pc[tid].nomal_z * d_Pc_Refer[j].nomal_z;

            f4 = atan2f(d_norm, dot);
            result_F4[tid * nRefer + j] = int((f4 - g_other.Min_angle) / (g_other.Max_angle - g_other.Min_angle) *
                                               g_other.nAngle);
            result_hash[tid * nRefer + j] += result_F4[tid * nRefer + j] * (g_other.g_P4);

            //calculate hash
            result_hash[tid * nRefer + j] %= g_other.max_hashIndex;

            //calculate alpha
            result_alpha[tid * nRefer + j] = g_CreateRotAngle(d_Pc[tid], d_Pc_Refer[j]);

        } //tid 2 other points



    }  //tid point
}

__global__ void
scene_ppf2(cuda_PPFInfo *d_Pc, cuda_PPFInfo *d_Pc_copy, int *result_F1, int *result_F2, int *result_F3,
           int *result_F4, float *result_alpha, int *result_hash, int nx, int ny, int Num,
           cuda_PPFotherInfo g_other,
           PPInfo *d_hashValue,int *d_hashKeyIndex) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float f1 = 0, f2 = 0, f3 = 0, f4 = 0;
    float d_norm, dot, d_length, d_x = 1, d_y = 1, d_z = 1;
    float X2_x = 1, X2_y = 1, X2_z = 1;
    float X3_x = 1, X3_y = 1, X3_z = 1;
    float X4_x = 1, X4_y = 1, X4_z = 1;

    if (tid < nx * ny) {
        for (int j = 0; j < Num; j++) {
            //calculate F1, f1 is the dis
            f1 = sqrtf(powf((d_Pc[tid].x - d_Pc_copy[j].x), 2) + powf((d_Pc[tid].y - d_Pc_copy[j].y), 2) +
                       powf((d_Pc[tid].z - d_Pc_copy[j].z), 2));
            result_F1[tid * Num + j] = int(f1 / g_other.d_distance);
            result_hash[tid * Num + j] = result_F1[tid * Num + j] * g_other.g_P1;

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
            result_F2[tid * Num + j] = int((f2 - g_other.Min_angle) / (g_other.Max_angle - g_other.Min_angle) *
                                           g_other.nAngle);
            result_hash[tid * Num + j] += result_F2[tid * Num + j] * (g_other.g_P2);

            //calculate F3,f3 is the angel_d_n2
            X3_x = d_y * d_Pc_copy[j].nomal_z - d_z * d_Pc_copy[j].nomal_y;
            X3_y = d_z * d_Pc_copy[j].nomal_x - d_x * d_Pc_copy[j].nomal_z;
            X3_z = d_x * d_Pc_copy[j].nomal_y - d_y * d_Pc_copy[j].nomal_x;
            d_norm = sqrtf(powf(X3_x, 2) + powf(X3_y, 2) + powf(X3_z, 2));

            dot = d_Pc_copy[j].nomal_x * d_x + d_Pc_copy[j].nomal_y * d_y + d_Pc_copy[j].nomal_z * d_z;

            f3 = atan2f(d_norm, dot);
            result_F3[tid * Num + j] = int((f3 - g_other.Min_angle) / (g_other.Max_angle - g_other.Min_angle) *
                                           g_other.nAngle);
            result_hash[tid * Num + j] += result_F3[tid * Num + j] * (g_other.g_P3);

            //calculate F4,f4 is the angel_n1_n2
            X4_x = d_Pc[tid].nomal_y * d_Pc_copy[j].nomal_z - d_Pc[tid].nomal_z * d_Pc_copy[j].nomal_y;
            X4_y = d_Pc[tid].nomal_z * d_Pc_copy[j].nomal_x - d_Pc[tid].nomal_x * d_Pc_copy[j].nomal_z;
            X4_z = d_Pc[tid].nomal_x * d_Pc_copy[j].nomal_y - d_Pc[tid].nomal_y * d_Pc_copy[j].nomal_x;
            d_norm = sqrtf(powf(X4_x, 2) + powf(X4_y, 2) + powf(X4_z, 2));

            dot = d_Pc[tid].nomal_x * d_Pc_copy[j].nomal_x + d_Pc[tid].nomal_y * d_Pc_copy[j].nomal_y +
                  d_Pc[tid].nomal_z * d_Pc_copy[j].nomal_z;

            f4 = atan2f(d_norm, dot);
            result_F4[tid * Num + j] = int((f4 - g_other.Min_angle) / (g_other.Max_angle - g_other.Min_angle) *
                                           g_other.nAngle);
            result_hash[tid * Num + j] += result_F4[tid * Num + j] * (g_other.g_P4);

            //calculate hash
            result_hash[tid * Num + j] %= g_other.max_hashIndex;

            //calculate alpha
            result_alpha[tid * Num + j] = g_CreateRotAngle(d_Pc[tid], d_Pc_copy[j]);

            //find in the hash table

            int d_hashNum = d_hashKeyIndex[result_hash[tid * Num + j] + 1] -
                            d_hashKeyIndex[result_hash[tid * Num + j]];        //+1! so ......look the next line!
            if (result_hash[tid * Num + j] > 0 &&
                (result_hash[tid * Num + j] < g_other.hashTableSize - 1)) {     // -1(because didn't control the end!)
                for (int k = 0; k < d_hashNum; k++) {

                    int cur_F1 = d_hashValue[d_hashKeyIndex[result_hash[tid * Num + j]] + k].F1;
                    int cur_F2 = d_hashValue[d_hashKeyIndex[result_hash[tid * Num + j]] + k].F2;
                    int cur_F3 = d_hashValue[d_hashKeyIndex[result_hash[tid * Num + j]] + k].F3;
                    int cur_F4 = d_hashValue[d_hashKeyIndex[result_hash[tid * Num + j]] + k].F4;
                    float cur_alpha = d_hashValue[d_hashKeyIndex[result_hash[tid * Num + j]] + k].alpha;
                    int pt_id = d_hashValue[d_hashKeyIndex[result_hash[tid * Num + j]] + k].pt1_id;

                    float dis_thresh = 0;
                    float angle_thresh = 0;

                    if (fabsf(result_F1[tid * Num + j] - cur_F1) > dis_thresh || fabsf(result_F2[tid * Num + j] - cur_F2) > angle_thresh ||
                        fabsf(result_F3[tid * Num + j] - cur_F3) > angle_thresh || fabsf(result_F4[tid * Num + j] - cur_F4) > angle_thresh) {

                    } else {
                        float alpha_m2s = cur_alpha - result_alpha[tid * Num + j];

                        if (alpha_m2s < -3.141592653)
                            alpha_m2s += 3.141592653 * 2;
                        if (alpha_m2s > 3.141592653)
                            alpha_m2s -= 3.141592653 * 2;

                        int angleID =
                                (alpha_m2s - g_other.Min_angle) / (g_other.Max_angle - g_other.Min_angle) *
                                g_other.nAngle;

                        if (tid < 2 && j <20 && k <10) {
                            printf("hash:%d tid:%d j:%d *****%d\n",result_hash[tid * Num + j],tid,j,angleID);
                        }
                        //

                    }
                }
            }
        }   //reference point 2 other points
    }   //reference points
}


__global__ void kernelTest(PPInfo *d_hashValue, PPInfo **d_hashKey) {
    int tid = threadIdx.x;

    /*PPInfo *p = d_hashKey[0];
    for (int i = tid*10; i < (tid+1)*10; i++) {
        d_hashKey[i] = d_hashKey[i] - p + &(d_hashValue[0]);
    }
    __synchreads();*/

    /*PPInfo *p=d_hashKey[0];
    int k=0;
    for (int i = 0; i < 10000; i++) {
        //k= atomicSub(d_hashKey[i],p);
        //d_hashKey[i] = atomicAdd(&(d_hashValue[0]),k);
        k=atomicAdd(&k,1);
        //d_hashKey[i] = d_hashKey[i] - p + &(d_hashValue[0]);
    }*/
}

void ModelCuda(cuda_PPFInfo* h_Pc,cuda_PPFInfo* h_Pc_copy, cuda_PPFotherInfo h_other, int* h_fromGPU_hash, float* h_fromGPU_alpha,int* h_fromGPU_F1,int* h_fromGPU_F2,int* h_fromGPU_F3,int* h_fromGPU_F4)
{

    cout<<"GPU start:"<<endl;
    cudaSetDevice(0);
    int nx=128;
    int ny=32;
    int nxy=nx*ny;
    int nBytes=nxy*sizeof(cuda_PPFInfo);

    cuda_PPFInfo *d_Pc=NULL;
    cuda_PPFInfo *d_Pc_copy=NULL;
    int *d_result_F1=NULL;
    int *d_result_F2=NULL;
    int *d_result_F3=NULL;
    int *d_result_F4=NULL;
    float *d_result_alpha=NULL;
    int *d_result_hash=NULL;

    cudaMalloc((void **)&d_Pc,nBytes);                                   ///test to change void as cuda_PPinfo
    cudaMalloc((void **)&d_Pc_copy,nBytes);

    cudaMalloc((int **)&d_result_F1,sizeof(int)*nxy*nxy);
    cudaMalloc((int **)&d_result_F2,sizeof(int)*nxy*nxy);
    cudaMalloc((int **)&d_result_F3,sizeof(int)*nxy*nxy);
    cudaMalloc((int **)&d_result_F4,sizeof(int)*nxy*nxy);
    cudaMalloc((float **)&d_result_alpha,sizeof(float)*nxy*nxy);
    cudaMalloc((int **)&d_result_hash,sizeof(int)*nxy*nxy);

    cudaMemcpy(d_Pc,h_Pc,nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_Pc_copy,h_Pc_copy,nBytes,cudaMemcpyHostToDevice);

    dim3 grid(ny);
    dim3 block(nx);

    //kernelA_1dimension<<<grid, block>>>(d_Pc,d_Pc_copy,d_result_F1,d_result_F2,d_result_F3,d_result_F4,d_result_hash,nx,g_other);
    kernelA_2dimension<<<grid, block>>>(d_Pc,d_Pc_copy,d_result_F1,d_result_F2,d_result_F3,d_result_F4,d_result_alpha,d_result_hash,nx,ny,h_other);
    cudaDeviceSynchronize();

    cudaMemcpy(h_fromGPU_hash,d_result_hash,sizeof(int)*nxy*nxy,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fromGPU_alpha,d_result_alpha,sizeof(float)*nxy*nxy,cudaMemcpyDeviceToHost);

    cudaMemcpy(h_fromGPU_F1,d_result_F1,sizeof(int)*nxy*nxy,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fromGPU_F2,d_result_F2,sizeof(int)*nxy*nxy,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fromGPU_F3,d_result_F3,sizeof(int)*nxy*nxy,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fromGPU_F4,d_result_F4,sizeof(int)*nxy*nxy,cudaMemcpyDeviceToHost);

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

void SceneCuda(cuda_PPFInfo* h_Pc,cuda_PPFInfo* h_Pc_Refer,cuda_PPFotherInfo h_other, int* h_fromGPU_hash, float* h_fromGPU_alpha,int* h_fromGPU_F1,int* h_fromGPU_F2,int* h_fromGPU_F3,int* h_fromGPU_F4)
{
    cout<<"GPU start:"<<endl;
    cudaSetDevice(0);
    int nx=256;
    int ny=40;
    int nxy=nx*ny; //10240(10091)
    int nReference=2019;
    int nBytes=nxy*sizeof(cuda_PPFInfo);
    int nBytes_Refer=nReference*sizeof(cuda_PPFInfo);

    cuda_PPFInfo *d_Pc=NULL;
    cuda_PPFInfo *d_Pc_Refer=NULL;
    int *d_result_F1=NULL;
    int *d_result_F2=NULL;
    int *d_result_F3=NULL;
    int *d_result_F4=NULL;
    float *d_result_alpha=NULL;
    int *d_result_hash=NULL;

    cudaMalloc((void **)&d_Pc,nBytes);                                   ///test to change void as cuda_PPinfo
    cudaMalloc((void **)&d_Pc_Refer,nBytes_Refer);

    cudaMalloc((int **)&d_result_F1,sizeof(int)*nxy*nReference);
    cudaMalloc((int **)&d_result_F2,sizeof(int)*nxy*nReference);
    cudaMalloc((int **)&d_result_F3,sizeof(int)*nxy*nReference);
    cudaMalloc((int **)&d_result_F4,sizeof(int)*nxy*nReference);
    cudaMalloc((float **)&d_result_alpha,sizeof(float)*nxy*nReference);
    cudaMalloc((int **)&d_result_hash,sizeof(int)*nxy*nReference);

    cudaMemcpy(d_Pc,h_Pc,nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_Pc_Refer,h_Pc_Refer,nBytes_Refer,cudaMemcpyHostToDevice);

    dim3 grid(ny);
    dim3 block(nx);

    scene_ppf<<<grid, block>>>(d_Pc,d_Pc_Refer,d_result_F1,d_result_F2,d_result_F3,d_result_F4,d_result_alpha,d_result_hash,nxy,nReference,h_other);
    cudaDeviceSynchronize();

    cudaMemcpy(h_fromGPU_hash,d_result_hash,sizeof(int)*nxy*nReference,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fromGPU_alpha,d_result_alpha,sizeof(float)*nxy*nReference,cudaMemcpyDeviceToHost);

    cudaMemcpy(h_fromGPU_F1,d_result_F1,sizeof(int)*nxy*nReference,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fromGPU_F2,d_result_F2,sizeof(int)*nxy*nReference,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fromGPU_F3,d_result_F3,sizeof(int)*nxy*nReference,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fromGPU_F4,d_result_F4,sizeof(int)*nxy*nReference,cudaMemcpyDeviceToHost);

    cudaFree(d_Pc);
    cudaFree(d_Pc_Refer);
    cudaFree(d_result_F1);
    cudaFree(d_result_F2);
    cudaFree(d_result_F3);
    cudaFree(d_result_F4);
    cudaFree(d_result_alpha);
    cudaFree(d_result_hash);

    cudaDeviceReset();
}

void Reference2NumCuda(cuda_PPFInfo* h_Pc,cuda_PPFInfo* h_Pc_copy, cuda_PPFotherInfo h_other, int* h_fromGPU_hash, float* h_fromGPU_alpha,int* h_fromGPU_F1,int* h_fromGPU_F2,int* h_fromGPU_F3,int* h_fromGPU_F4, PPInfo* modelHashValue,int* modelHashKeyIndex)
{
    cout<<"GPU start:"<<endl;
   /* ofstream  outFile("Hash_train_model.txt");
    outFile << h_other.hashTableSize << endl;
    for (int i = 0; i < h_other.hashTableSize; i++)
    {
        outFile << i << " " << modelHashKey[i+1]-modelHashKey[i]<<" ";
        for (PPInfo *ptr =modelHashKey[i]; ptr< modelHashKey[i+1]; ptr++)
        {
            outFile << (*ptr).pt1_id << " " << (*ptr).pt2_id << " " << (*ptr).F1 << " " << (*ptr).F2 << " " << (*ptr).F3 << " " << (*ptr).F4<<" " << (*ptr).alpha << " ";
        }
        outFile << endl;
    }
    outFile.close();*/

    cudaSetDevice(0);
    int nx=128;
    int ny=16;
    int nxy=nx*ny;
    int nBytes=nxy*sizeof(cuda_PPFInfo);
    int nu=10091;
    int NumBytes=nu*sizeof(cuda_PPFInfo);

    cuda_PPFInfo *d_Pc=NULL;
    cuda_PPFInfo *d_Pc_copy=NULL;
    int *d_result_F1=NULL;
    int *d_result_F2=NULL;
    int *d_result_F3=NULL;
    int *d_result_F4=NULL;
    float *d_result_alpha=NULL;
    int *d_result_hash=NULL;
    //points pair
    cudaMalloc((void **)&d_Pc,nBytes);                                   ///test to change void as cuda_PPinfo
    cudaMalloc((void **)&d_Pc_copy,NumBytes);
    cudaMemcpy(d_Pc,h_Pc,nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_Pc_copy,h_Pc_copy,NumBytes,cudaMemcpyHostToDevice);
    //point pair feature
    cudaMalloc((int **)&d_result_F1,sizeof(int)*nxy*nu);
    cudaMalloc((int **)&d_result_F2,sizeof(int)*nxy*nu);
    cudaMalloc((int **)&d_result_F3,sizeof(int)*nxy*nu);
    cudaMalloc((int **)&d_result_F4,sizeof(int)*nxy*nu);
    cudaMalloc((float **)&d_result_alpha,sizeof(float)*nxy*nu);
    cudaMalloc((int **)&d_result_hash,sizeof(int)*nxy*nu);
    //hash table
    PPInfo* d_hashValue=NULL;
    cudaMalloc((void**)&d_hashValue, sizeof(PPInfo)*h_other.hashNum);
    cudaMemcpy(d_hashValue,modelHashValue,sizeof(PPInfo)*h_other.hashNum,cudaMemcpyHostToDevice);

    int *d_hashKeyIndex=NULL;
    cudaMalloc((int **)&d_hashKeyIndex, sizeof(int)*h_other.hashTableSize);
    cudaMemcpy(d_hashKeyIndex,modelHashKeyIndex, sizeof(int)*h_other.hashTableSize,cudaMemcpyHostToDevice);

    dim3 grid(ny);
    dim3 block(nx);
    //kernelTest<<<1,1>>>(d_hashValue,d_hashKey);
    //kernelA_1dimension<<<grid, block>>>(d_Pc,d_Pc_copy,d_result_F1,d_result_F2,d_result_F3,d_result_F4,d_result_hash,nx,g_other);
    scene_ppf2<<<grid, block>>>(d_Pc,d_Pc_copy,d_result_F1,d_result_F2,d_result_F3,d_result_F4,d_result_alpha,d_result_hash,nx,ny,nu,h_other,d_hashValue,d_hashKeyIndex);
    cudaDeviceSynchronize();

    cudaMemcpy(h_fromGPU_hash,d_result_hash,sizeof(int)*nxy*nu,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fromGPU_alpha,d_result_alpha,sizeof(float)*nxy*nu,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fromGPU_F1,d_result_F1,sizeof(int)*nxy*nu,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fromGPU_F2,d_result_F2,sizeof(int)*nxy*nu,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fromGPU_F3,d_result_F3,sizeof(int)*nxy*nu,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fromGPU_F4,d_result_F4,sizeof(int)*nxy*nu,cudaMemcpyDeviceToHost);

    cudaFree(d_Pc);
    cudaFree(d_Pc_copy);
    cudaFree(d_result_F1);
    cudaFree(d_result_F2);
    cudaFree(d_result_F3);
    cudaFree(d_result_F4);
    cudaFree(d_result_alpha);
    cudaFree(d_result_hash);
    cudaFree(d_hashValue);
    cudaFree(d_hashKeyIndex);

    cudaDeviceReset();
}


bool  InitCuda()
{
    //Pinfo pn1;
    cudaSetDevice(0);
    kernelA <<<2,33>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return true;
}
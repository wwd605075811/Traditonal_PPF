#include "../include/Kernel.h"
//#include "../include/vector_ops.h"

__host__ __device__ int4 discreteDisAndAngle(float4 ppf, float min_angle, float d_angle) {
    int4 r;

    if(ppf.x == -1) {
        r.x = -1;
        return r;
    }

    r.x = int(ppf.x);
    r.y = int((ppf.y - min_angle) * d_angle);
    r.z = int((ppf.z - min_angle) * d_angle);
    r.w = int((ppf.w - min_angle) * d_angle);
    return r;
}

__host__ __device__ float4 computePPF(float3 p1, float3 n1, float3 p2, float3 n2, float d_dist) {
    /*
    *这里的PPF计算、离散、计算Hash都采用 PPF.cpp 的计算公式，可对照cpu版公式理解涵义
    */
    float3 d;
    d.x = p2.x - p1.x;
    d.y = p2.y - p1.y;
    d.z = p2.z - p1.z;
    float4 f;
    f.x = norm(d) / d_dist;

    // because tau_d is 0.05
    if(f.x > 20) {
        f.x = -1;
        return f;
    }

    d=normalize(d);
    f.y = atan2f(norm(cross(d,n1)), dot(d,n1));;
    f.z = atan2f(norm(cross(d,n2)), dot(d,n2));
    f.w = atan2f(norm(cross(n1,n2)), dot(n1,n2));
    return f;
}

__host__ __device__ unsigned int hash(int4 ppf){
    return (ppf.x * P1 + ppf.y * P2 + ppf.z * P3 + ppf.w * P4) % MAX_HASHINDEX;
}

__host__ __device__ __forceinline__ void zeroMat4(float T[4][4]) {
    T[0][0] = 0;    T[0][1] = 0;    T[0][2] = 0;    T[0][3] = 0;
    T[1][0] = 0;    T[1][1] = 0;    T[1][2] = 0;    T[1][3] = 0;
    T[2][0] = 0;    T[2][1] = 0;    T[2][2] = 0;    T[2][3] = 0;
    T[3][0] = 0;    T[3][1] = 0;    T[3][2] = 0;    T[3][3] = 0;
}

__host__ __device__ void trans(float3 v, float T[4][4]) {
    //构建平移矩阵
    zeroMat4(T);
    T[0][0] = 1;
    T[1][1] = 1;
    T[2][2] = 1;
    T[3][3] = 1;
    T[0][3] = v.x;
    T[1][3] = v.y;
    T[2][3] = v.z;
}

__host__ __device__ void rotx(float theta, float T[4][4]) {
    //向X轴旋转的旋转矩阵
    zeroMat4(T);
    T[0][0] = 1;
    T[1][1] = cosf(theta);
    T[2][1] = sinf(theta);
    T[1][2] = -1*T[2][1];
    T[2][2] = T[1][1];
    T[3][3] = 1;
}

__host__ __device__ void roty(float theta, float T[4][4]) {
    //向Y轴旋转的旋转矩阵
    zeroMat4(T);
    T[0][0] = cosf(theta);
    T[0][2] = sinf(theta);
    T[1][1] = 1;
    T[2][0] = -1*T[0][2];
    T[2][2] = T[0][0];
    T[3][3] = 1;
}

__host__ __device__ void rotz(float theta, float T[4][4]) {
    //向Z轴旋转的旋转矩阵
    zeroMat4(T);
    T[0][0] = cosf(theta);
    T[1][0] = sinf(theta);
    T[0][1] = -1*T[1][0];
    T[1][1] = T[0][0];
    T[2][2] = 1;
    T[3][3] = 1;
}

__host__ __device__ float4 homogenize(float3 v) {
    float4 w = {v.x, v.y, v.z, 1};
    return w;
}

__host__ __device__ float3 dehomogenize(float4 v) {
    float3 w = {v.x, v.y, v.z};
    return w;
}

__host__ __device__ float4 mat4fVmul(const float A[4][4], const float4 b) {
    float4 *Af4 = (float4 *) A;
    float4 c;
    c.x = dot(Af4[0], b);
    c.y = dot(Af4[1], b);
    c.z = dot(Af4[2], b);
    c.w = dot(Af4[3], b);
    return c;
}

__host__ __device__ void mat4fMul(const float A[4][4], const float B[4][4], float C[4][4]) {
    zeroMat4(C);
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            for(int k = 0; k < 4; k++){
                C[i][j] += A[i][k]*B[k][j];
            }
        }
    }
}

__host__ __device__ float quantDownf(float x, float y) {
    return x - fmodf(x, y);
}

__device__ void transModelAndScene(float3 m_r, float3 n_r_m, float3 m_i,
                                  float3 s_r, float3 n_r_s, float3 s_i,
                                  float d_dist, unsigned int &alpha_idx){
    /*按照论文3.3节公式，计算转角：
    *用来计算把点对一端旋转到X轴上，另一端的夹角
    *后面不采用这个大函数，拆分为：transPointPair()  和  transPointPair()
    */
    float transm[4][4], rot_x[4][4], rot_y[4][4], rot_z[4][4], T_tmp[4][4], T_m_g[4][4], T_s_g[4][4],
            T_tmp2[4][4], T[4][4];
    float4 n_tmp;
    //旋转model中的点对
    m_r = -1 * m_r;
    trans(m_r, transm);
    roty(atan2f(n_r_m.z, n_r_m.x), rot_y);
    n_tmp = homogenize(n_r_m);
    n_tmp = mat4fVmul(rot_y, n_tmp);
    rotz(-1*atan2f(n_tmp.y, n_tmp.x), rot_z);
    mat4fMul(rot_z, rot_y, T_tmp);
    mat4fMul(T_tmp, transm, T_m_g);
    //旋转scene中的点对
    s_r = -1 * s_r;
    trans(s_r, transm);
    roty(atan2f(n_r_s.z, n_r_s.x), rot_y);
    n_tmp = homogenize(n_r_s);
    n_tmp = mat4fVmul(rot_y, n_tmp);
    rotz(-1*atan2f(n_tmp.y, n_tmp.x), rot_z);
    mat4fMul(rot_z, rot_y, T_tmp);
    mat4fMul(T_tmp, transm, T_s_g);
    //计算model夹角
    n_tmp = homogenize(m_i);
    n_tmp = mat4fVmul(T_m_g, n_tmp);
    float3 u = dehomogenize(n_tmp);
    //计算scene夹角
    n_tmp = homogenize(s_i);
    n_tmp = mat4fVmul(T_s_g, n_tmp);
    float3 v = dehomogenize(n_tmp);
    //计算夹角差值
    u.x = 0;
    v.x = 0;
    float alpha = atan2f(cross(u, v).x, dot(u, v));
    alpha = quantDownf(alpha + CUDART_PI_F, D_ANGLE0);
    alpha_idx = (unsigned int) (lrintf(alpha/D_ANGLE0));
    rotx(alpha, rot_x);
}

__device__ void transPointPair(float3 m_r, float3 n_r_m, float3 m_i, float &alpha){
    float transm[4][4], rot_x[4][4], rot_y[4][4], rot_z[4][4], T_tmp[4][4], T_m_g[4][4], T_s_g[4][4], T_tmp2[4][4], T[4][4];
    float4 n_tmp;
    float3 u;
    m_r = -1 * m_r;
    //计算平移矩阵
    trans(m_r, transm);
    //计算绕Y轴旋转的旋转矩阵rot_y
    roty(atan2f(n_r_m.z, n_r_m.x), rot_y);
    n_tmp = homogenize(n_r_m);
    n_tmp = mat4fVmul(rot_y, n_tmp);
    //计算绕Z轴旋转的旋转矩阵rot_z
    rotz(-1*atan2f(n_tmp.y, n_tmp.x), rot_z);
    //combina three transports into one T_m_g
    mat4fMul(rot_z, rot_y, T_tmp);
    mat4fMul(T_tmp, transm, T_m_g);

    n_tmp = homogenize(m_i);
    n_tmp = mat4fVmul(T_m_g, n_tmp);

    u = dehomogenize(n_tmp);
    float rot_angle = atan2f(u.y,u.z);
    alpha = rot_angle;
}

__device__ void transModelScene(float u, float v, unsigned int &alpha_idx) {
    /*float alpha = atan2f(cross(u, v).x, dot(u, v));
    alpha = quantDownf(alpha + CUDART_PI_F, D_ANGLE0);
    alpha_idx = (unsigned int) (lrintf(alpha/D_ANGLE0));*/
    float alpha_m2s = u - v;

    if (alpha_m2s < -M_PI)
        alpha_m2s += M_PI * 2;
    if (alpha_m2s > M_PI)
        alpha_m2s -= M_PI * 2;

    alpha_idx = (alpha_m2s - MIN_ANGLE)*D_ANGLE;
}

__device__ bool isFeatureSimilar(int4 scenePPF, int4 modelPPF, float dis_thresh, float angle_thresh) {
    /*
    *因为做了离散处理，两个点对要判等得完全相等
    *这个判等在CUDA十分影响计算时间，增加约一倍！后面可以采取策略消除掉
    */
    if (fabsf(scenePPF.x - modelPPF.x) > dis_thresh || fabsf(scenePPF.y - modelPPF.y) > angle_thresh || 
        fabsf(scenePPF.z - modelPPF.z) > angle_thresh || fabsf(scenePPF.w - modelPPF.w) > angle_thresh)
        return false;
    return true;
}

__device__ bool isSimilar(int4 scenePPF, int4 modelPPF) {
    if (scenePPF.x != modelPPF.x || scenePPF.y != modelPPF.y || scenePPF.z != modelPPF.z ||
        scenePPF.w != modelPPF.w)
        return false;
    return true;
}

__global__ void ppfKernel(float3 *points, float3 *norms, int4 *out, int count,
                           int refPointDownsampleFactor, float d_dist) {
/*
*这里的线层组织逻辑不好描述，建议按照流程一步一步在纸上走一遍便可以理解设计原因
*分块计算，提高处理效率。注意idx和各个索引(i,j)的计算
*/
    if(count <= 1) return;
    int ind = threadIdx.x;
    int idx = ind + blockIdx.x * blockDim.x;
    int bound;

    while(idx < count) {
        float3 thisPoint = points[idx];
        float3 thisNorm  = norms[idx];
        for(int i = 0; i < count; i+=BLOCK_SIZE){
            bound = MIN(count - i, BLOCK_SIZE);
            __syncthreads();
            for(int j = 0; j < bound; j++) {
                //this is to select the reference points,eg. refPointDownsampleFactor=1 means that:
                //every point need to calculate the PPF! Meanwhile refPointDownsampleFactor=5 means that:
                //every 5 points has one point to calculate the PPF
                if(idx % refPointDownsampleFactor != 0){
                    out[idx*count + j + i].x = CUDART_NAN_F;
                    continue;
                } ;
                // handle case of identical points in pair
                if((j + i - idx) == 0){
                    out[idx*count + j + i].x = CUDART_NAN_F;
                    continue;
                } ;
                float4 ppf = computePPF(thisPoint, thisNorm, points[i + j], norms[i + j], d_dist);

                out[idx*count + i + j] = discreteDisAndAngle(ppf, MIN_ANGLE, D_ANGLE);
            }
        }
        //grid stride
        __syncthreads();
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void ppfAngle(float3 *points, float3 *norms, float *out, int count,
                           int refPointDownsampleFactor, float d_dist){
    if(count <= 1)
        return;
    int ind = threadIdx.x;
    int idx = ind + blockIdx.x * blockDim.x;
    int bound;

    while(idx < count) {
        float3 thisPoint = points[idx];
        float3 thisNorm  = norms[idx];
        for(int i = 0; i < count; i+=BLOCK_SIZE){
            bound = MIN(count - i, BLOCK_SIZE);

            for(int j = 0; j < bound; j++) {
                if(idx % refPointDownsampleFactor != 0){
                    out[idx*count + j + i] = CUDART_NAN_F;
                    continue;
                } ;
                if((j + i - idx) == 0){
                    out[idx*count + j + i] = CUDART_NAN_F;
                    continue;
                } ;
                float alpha;
                transPointPair(thisPoint, thisNorm, points[i + j], alpha);
                out[idx*count + i + j] = alpha;
            }
        }
        __syncthreads();
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void ppf_hash_kernel(int4 *ppfs, unsigned int *codes, int count ) {
    if(count <= 1) return;

    int ind = threadIdx.x;
    int idx = ind + blockIdx.x * blockDim.x;

    while(idx < count) {
        if(ppfs[idx].x == CUDART_NAN_F || ppfs[idx].x == -1) {
            codes[idx] = 0;
        } else {
            codes[idx] = hash(ppfs[idx]);
        }
        //grid stride
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void ppf_vote_count_kernel(unsigned int *sceneKeys, unsigned int *sceneIndices,
                                      unsigned int *hashKeys, std::size_t *ppfCount,
                                      unsigned long *ppf_vote_counts, int count){
    if(count <= 1) return;

    int ind = threadIdx.x;
    int idx = ind + blockIdx.x * blockDim.x;

    while(idx < count) {
        unsigned int thisSceneKey = sceneKeys[idx];
        unsigned int thisSceneIndex = sceneIndices[idx];
        if(thisSceneKey == 0 ||
           thisSceneKey != hashKeys[thisSceneIndex]) {
            ppf_vote_counts[idx] = 0;
        } else {
            ppf_vote_counts[idx] = ppfCount[thisSceneIndex];
        }
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void ppf_vote_kernel(unsigned int *sceneKeys, unsigned int *sceneIndices,
                                unsigned int *hashKeys, std::size_t *ppfCount,
                                std::size_t *firstPPFIndex, std::size_t *key2ppfMap,
                                float3 *modelPoints, float3 *modelNormals, int modelSize,
                                float3 *scenePoints, float3 *sceneNormals, int sceneSize,
                                int *voteAccumSpace, int4 *modelPPFs, int4 *scenePPFs,
                                float *modelAngles, float *sceneAngles, int count, float d_dist) {
    /*参数列表：
    * 1.scene所有点对hash值 2.scene所有点对与Hash值索引 
    * 3.unique hashes  4.hash counts 5.StartingIndices 6.hashToDataMap (参考论文27页和技术文档)
    * 7.model点  8.model点法向量  9.model点数量
    * 10.scene点 11.scene点法向量 12.scene点数量
    * 13.累加数组 14.model中所有点对PPFs 15.scene中所有点对PPFs
    * 16.model中所有点对的旋转角 17.scene中所有点对的旋转角
    * 18.scene中Hash值数量 19.d_dist(没用上)
    *每个thread完成对于一个点对的投票
    */
    if(count <= 1) return;

    int ind = threadIdx.x;
    int idx = ind + blockIdx.x * blockDim.x;
    unsigned int alpha_idx;
    float dis_thresh = 0;
    float angle_thresh = 0;

    while(idx < count) {
        unsigned int thisSceneKey = sceneKeys[idx];
        unsigned int thisSceneIndex = sceneIndices[idx];
        if (thisSceneKey == 0 ||
            thisSceneKey != hashKeys[thisSceneIndex]) {
            idx += blockDim.x * gridDim.x;
            continue;
        }
        unsigned int thisPPFCount = ppfCount[thisSceneIndex];
        unsigned int thisFirstPPFIndex = firstPPFIndex[thisSceneIndex];
        //get reference point ID
        unsigned int scene_r_index = idx / sceneSize;
        //get scene point ID
        unsigned int scene_i_index = idx - scene_r_index*sceneSize;

        int4 thisScenePPF = scenePPFs[scene_r_index * sceneSize + scene_i_index];

        if(thisScenePPF.x == -1) {
            continue;
        }
        float thisSceneAngle = sceneAngles[scene_r_index * sceneSize + scene_i_index];

        unsigned int modelPPFIndex, model_r_index, model_i_index;
        /*
        *以上都是计算索引
        *以下是对当前点对的Hash值，在Model Hash表中查找，找到相同Hash值的好多点对s，
        *依次比较，相同则投票(原子操作，不用结果也没变，用上保险)
        */
        for(int i = 0; i < thisPPFCount; i++) {
            modelPPFIndex = key2ppfMap[thisFirstPPFIndex+i];
            model_r_index = modelPPFIndex / modelSize;
            model_i_index = modelPPFIndex - model_r_index*modelSize;

            int4 thisModelPPF = modelPPFs[model_r_index * modelSize + model_i_index];
            float thisModelAngle = modelAngles[model_r_index * modelSize + model_i_index];

            if(isFeatureSimilar(thisScenePPF,thisModelPPF,dis_thresh,angle_thresh)) {
                transModelScene(thisModelAngle, thisSceneAngle, alpha_idx);

                atomicAdd(&voteAccumSpace[scene_r_index/5 * modelSize * N_ANGLE +
                                          model_r_index * N_ANGLE +
                                          alpha_idx],1);

            }
        }
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void addVote(int *d_accumSpace, int *d_votingPoint, int *d_votingNumber, int *d_votingAngle,
    int modelSize, int sceneSize, int *secondPoint, int *secondNumber, int *secondAngle){
    /*
    *对投票结束后的累加数组进行统计，找出每个参考点所对应数组中的
    *最大值和次大值(值代表：参考点所对应的model点，以及旋转角度，以及获票数)
    *kernel为最简单的2维thread排列
    */
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int rowLen = modelSize;
    int colLen = N_ANGLE;
    int maxAccum = -1;
    int idy_max, idx_max;
    int refPointNum = (sceneSize / 5) + 1;
    if(tid<refPointNum){
        for (int idy = 0; idy < rowLen; idy++) {
            for (int idx = 0; idx < colLen; idx++) {
                int votingValue = d_accumSpace[tid * modelSize * N_ANGLE +
                                               idy * N_ANGLE +
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

        d_accumSpace[tid * modelSize * N_ANGLE +
                    idy_max * N_ANGLE +
                    idx_max] = -1;
        idx_max = 0;
        idy_max = 0;
        maxAccum = -1;
        for (int idy = 0; idy < rowLen; idy++) {
            for (int idx = 0; idx < colLen; idx++) {
                int votingValue = d_accumSpace[tid * modelSize * N_ANGLE +
                                               idy * N_ANGLE +
                                               idx];
                if (votingValue > maxAccum) {
                    maxAccum = votingValue;
                    idy_max = idy;
                    idx_max = idx;
                }
            }
        }
        secondPoint[tid] = idy_max;   //model point ID
        secondAngle[tid] = idx_max;   //rot angle
        secondNumber[tid] = maxAccum; //votes number
    }
}

__host__ __device__ float dot(float3 v1, float3 v2){
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

__host__ __device__ float dot(float4 v1, float4 v2){
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z + v1.w*v2.w;
}

__host__ __device__ float norm(float3 v){
    return sqrtf(dot(v, v));
}

__host__ __device__ float norm(float4 v){
    return sqrtf(dot(v, v));
}

__host__ __device__ float3 normalize(float3 v){
    float length = sqrtf(dot(v, v));
    float3 w = {v.x / length, v.y / length, v.z / length};
    return w;
}

__host__ __device__ float3 cross(float3 v1, float3 v2){
    float3 w ={v1.y * v2.z - v1.z * v2.y,
               v1.z * v2.x - v1.x * v2.z,
               v1.x * v2.y - v1.y * v2.x};
    return w;
}

__host__ __device__ float3 operator*(float a, float3 v){
    float3 w = {a*v.x, a*v.y, a*v.z};
    return w;
}

__host__ __device__ float4 operator*(float a, float4 v){
    float4 w = {a*v.x, a*v.y, a*v.z, a*v.z};
    return w;
}

__host__ __device__ float3 operator+(float3 u, float3 v){
    float3 w = {u.x+v.x, u.y+v.y, u.z+v.z};
    return w;
}

__host__ __device__ float4 operator+(float4 u, float4 v){
    float4 w = {u.x+v.x, u.y+v.y, u.z+v.z, u.w+v.w};
    return w;
}

__host__ __device__ float3 operator-(float3 u, float3 v){
    float3 w = {u.x-v.x, u.y-v.y, u.z-v.z};
    return w;
}

__host__ __device__ float4 operator-(float4 u, float4 v){
    float4 w = {u.x-v.x, u.y-v.y, u.z-v.z, u.w-v.w};
    return w;
}



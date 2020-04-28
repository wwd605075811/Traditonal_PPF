#ifndef __NEWKERNEL_H
#define __NEWKERNEL_H
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector_types.h>
#include <math_constants.h>
#include <iostream>

# define M_PI 3.14159265358979323846    /* pi */
//Launch configuration macros
#define BLOCK_SIZE 512
#define BLOCK_A 256
#define BLOCK_B 128
#define BLOCK_C 64
#define BLOCK_D 32
#define MAX_NBLOCKS 1024
#define N_ANGLE 30
#define MAX_HASHINDEX 20000
#define MAX_NDISTANCE 25       //int(1 / tau_d + 5)
//Algorithm macros
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MIN_ANGLE -M_PI
#define MAX_ANGLE M_PI
#define TWOPI (M_PI*2)
#define D_ANGLE (N_ANGLE / TWOPI)
#define P1 (N_ANGLE * N_ANGLE * MAX_NDISTANCE)   //P3 = N_ANGLE  P4 = 1
#define P2 (N_ANGLE * N_ANGLE)
#define P3 (N_ANGLE)
#define P4 1
#define D_ANGLE0 ((2.0f*float(CUDART_PI_F))/float(N_ANGLE))
//CUDA macros
#define RAW_PTR(V) thrust::raw_pointer_cast(V->data())
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__ ))
static void HandleError(cudaError_t err, const char *file, int line){
    if(err != cudaSuccess){
        fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err),
                file, line);
        exit(EXIT_FAILURE);
    }
}

/**
 * \简介 离散化f1~f4
 * \参数[in]
 *   1.未离散的ppf
 *   2. #define MIN_ANGLE -M_PI 
 *   3. #define D_ANGLE (N_ANGLE / TWOPI) 离散参数
 */
__host__ __device__ int4 discreteDisAndAngle(float4 ppf, float min_angle, float d_angle);
/**
 * \简介 计算PPFs值
 * \参数[in]
 *   1.参考点的坐标
 *   2. 参考点的法向量
 *   3. 其他点的坐标
 *   4. 其他点的法向量
 *   5. 距离离散参数
 * NOTE：计算公式都是采用CPU实现版本的计算公式
 */
__host__ __device__ float4 computePPF(float3 p1, float3 n1, float3 p2, float3 n2, float d_dist);
/**
 * \简介 求雨构造法-Hash值
 * \参数[in]
 *   1.离散后的ppf值
 *NOTE：github ppf-cuda在这里采用了另外一种hash值构造法，使hash值的碰撞更少，但是存储空间变大
 */
__host__ __device__ unsigned int hash(int4 ppf);

__host__ __device__ __forceinline__ void zeroMat4(float T[4][4]);

__host__ __device__ void trans(float3 v, float T[4][4]);

__host__ __device__ void rotx(float theta, float T[4][4]);

__host__ __device__ void roty(float theta, float T[4][4]);

__host__ __device__ void rotz(float theta, float T[4][4]);

__host__ __device__ float4 homogenize(float3 v);

__host__ __device__ float3 dehomogenize(float4 v);

__host__ __device__ float4 mat4fVmul(const float A[4][4], const float4 b);

__host__ __device__ void mat4fMul(const float A[4][4], const float B[4][4], float C[4][4]);

__host__ __device__ float quantDownf(float x, float y);

__device__ void transModelAndScene(float3 m_r, float3 n_r_m, float3 m_i,
                                  float3 s_r, float3 n_r_s, float3 s_i,
                                  float d_dist, unsigned int &alpha_idx);
/**
 * \简介 按照论文3.3节公式，计算转角。用来计算把点对一端旋转到X轴上后，另一端的夹角
 * \参数[in]
 *   1.m_r的坐标
 *   2.m_r的法向量
 *   3.m_i的坐标
 *   4.要得到的转角alpha
 * \NOTE 转角计算采用先转到X-Y平面再转到X轴，没有采用goldman绕轴旋转(测试有问题)
 */
__device__ void transPointPair(float3 m_r, float3 n_r_m, float3 m_i, float &alpha);
/**
 * \简介 计算αm 与 αs的差值，再离散化
 * \参数[in]
 *   1.αm
 *   2.αs
 *   3.角度差值
 *  note:  PCL库本身有bug，参考https://github.com/PointCloudLibrary/pcl/issues/1930
 *  目前没问题
 */
__device__ void transModelScene(float3 u, float3 v, unsigned int &alpha_idx);

__device__ bool isFeatureSimilar(int4 scenePPF, int4 modelPPF, float dis_thresh, float angle_thresh);

/**
 * \简介 计算点对的ppf
 * \参数[in]
 *   1.点的坐标
 *   2.点的法向量
 *   3.ppf结果值
 *   4.计数点数
 *   5.采样步长,model的为1，scene的为5
 *   6.离散参数
 */
__global__ void ppfKernel(float3 *points, float3 *norms, int4 *out, int count,
                           int refPointDownsampleFactor, float d_dist);
/**
 * \简介 计算点对的angle
 * \参数[in]
 *   1.点的坐标
 *   2.点的法向量
 *   3.ppf结果值
 *   4.计数点数
 *   5.采样步长,model的为1，scene的为5
 *   6.离散参数
 */
__global__ void ppfAngle (float3 *points, float3 *norms, float *out, int count, int refPointDownsampleFactor, float d_dist);
/**
 * \简介 计算点对的hash值
 * \参数[in]
 *   1.点对的ppfs
 *   2.点对的hash值
 *   3.点对数量计数
 */
__global__ void ppf_hash_kernel(int4 *ppfs, unsigned int *codes, int count );
/**
 * \简介 函数没用上，用来统计投票
 */
__global__ void ppf_vote_count_kernel(unsigned int *sceneKeys, unsigned int *sceneIndices,
                                      unsigned int *hashKeys, std::size_t *ppfCount,
                                      unsigned long *ppf_vote_counts, int count);
/**
 * \简介  逐点对投票
 * \参数[in]
 *  1.scene所有点对hash值 
 *  2.scene所有点对与Hash值索引 
 *  3.unique hashes  
 *  4.hash counts 
 *  5.StartingIndices 
 *  6.hashToDataMap (参考论文27页和技术文档)
 *  7.model点  
 *  8.model点法向量  
 *  9.model点数量
 *  10.scene点 
 *  11.scene点法向量 
 *  12.scene点数量
 *  13.累加数组 
 *  14.model中所有点对PPFs 
 *  15.scene中所有点对PPFs
 *  16.model中所有点对的旋转角 
 *  17.scene中所有点对的旋转角
 *  18.scene中Hash值数量 
 *  19.d_dist(没用上)
 *  \note 参数虽然多，但是逻辑很简单。(每个thread完成对于一个点对的投票)
 *   1.就是根据各种索引得到当前ppf点对的hash值
 *   2.在hash表查询所有相同的hash值
 *   3.判断是否相同，投票
 *   \问题 :再加入isSimilar判断点对是否相同使速度变慢，可以构思别的判等策略
 *   对于累加数组的原子操作，这里存在冲突，采用原子操作也会使速度变慢
 */
__global__ void ppf_vote_kernel(unsigned int *sceneKeys, unsigned int *sceneIndices,
                                unsigned int *hashKeys, std::size_t *ppfCount,
                                std::size_t *firstPPFIndex, std::size_t *key2ppfMap,
                                float3 *modelPoints, float3 *modelNormals, int modelSize,
                                float3 *scenePoints, float3 *sceneNormals, int sceneSize,
                                int *voteAccumSpace, int4 *modelPPFs, int4 *scenePPFs,
                                float *modelAngles, float *sceneAngles, int count, float d_dist);
/**
 * \简介 对投票函数的得到的累加数组进行统计，找到最大值和次大值
 * \参数[in]
 *   1.累加数组
 *   2.对应的最大model点
 *   3.对应的最大票数
 *   4.(最大)scene点与model的转角
 *   5.model点数
 *   6.scene点数
 *   7.对应的次大model点
 *   8.对应的次大票数
 *   9.(次大)scene点与model的转角
 * \note因为按顺序计算，所以存储2的索引就是对应的参考点索引
 */
__global__ void addVote(int *d_accumSpace, int *d_votingPoint, int *d_votingNumber, int *d_votingAngle,
                        int modelSize, int sceneSize, int *secondPoint, int *secondNumber, int *secondAngle);
__host__ __device__ float dot(float3 v1, float3 v2);

__host__ __device__ float dot(float4 v1, float4 v2);

__host__ __device__ float norm(float3 v);

__host__ __device__ float norm(float4 v);

__host__ __device__ float3 normalize(float3 v);

__host__ __device__ float3 cross(float3 v1, float3 v2);

__host__ __device__ float3 operator*(float a, float3 v);
__host__ __device__ float4 operator*(float a, float4 v);

__host__ __device__ float3 operator+(float3 u, float3 v);
__host__ __device__ float4 operator+(float4 u, float4 v);

__host__ __device__ float3 operator-(float3 u, float3 v);
__host__ __device__ float4 operator-(float4 u, float4 v);

#endif /* __NEWKERNEL_H */
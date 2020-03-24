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

__host__ __device__ int4 discreteAngle(float4 ppf, float min_angle, float d_angle);

__host__ __device__ float4 computePPF(float3 p1, float3 n1, float3 p2, float3 n2, float d_dist);

__host__ __device__ unsigned int hash(int4 ppf);


__host__ __device__ __forceinline__ void zeroMat4(float T[4][4]);

__host__ __device__ void trans(float3 v, float T[4][4]);

__host__ __device__ void rotx(float theta, float T[4][4]);

__host__ __device__ void roty(float theta, float T[4][4]);

__host__ __device__ void rotz(float theta, float T[4][4]);

__host__ __device__ float4 homogenize(float3 v);

__host__ __device__ float3 dehomogenize(float4 v);

__host__ __device__ float4 mat4f_vmul(const float A[4][4], const float4 b);

__host__ __device__ void mat4f_mul(const float A[4][4], const float B[4][4], float C[4][4]);

__host__ __device__ float quant_downf(float x, float y);

__device__ void trans_model_scene(float3 m_r, float3 n_r_m, float3 m_i,
                                  float3 s_r, float3 n_r_s, float3 s_i,
                                  float d_dist, unsigned int &alpha_idx);

__device__ void transPointPair(float3 m_r, float3 n_r_m, float3 m_i, float &alpha);

__device__ void transModelScene(float3 u, float3 v, unsigned int &alpha_idx);

__device__ bool isFeatureSimilar(int4 scenePPF, int4 modelPPF, float dis_thresh, float angle_thresh);

__global__ void ppfKernel(float3 *points, float3 *norms, int4 *out, int count,
                           int refPointDownsampleFactor, float d_dist);

__global__ void ppfAngle (float3 *points, float3 *norms, float *out, int count, int refPointDownsampleFactor, float d_dist);

__global__ void ppf_hash_kernel(int4 *ppfs, unsigned int *codes, int count );

__global__ void ppf_vote_count_kernel(unsigned int *sceneKeys, unsigned int *sceneIndices,
                                      unsigned int *hashKeys, std::size_t *ppfCount,
                                      unsigned long *ppf_vote_counts, int count);

__global__ void ppf_vote_kernel(unsigned int *sceneKeys, unsigned int *sceneIndices,
                                unsigned int *hashKeys, std::size_t *ppfCount,
                                std::size_t *firstPPFIndex, std::size_t *key2ppfMap,
                                float3 *modelPoints, float3 *modelNormals, int modelSize,
                                float3 *scenePoints, float3 *sceneNormals, int sceneSize,
                                int *voteAccumSpace, int4 *modelPPFs, int4 *scenePPFs,
                                float *modelAngles, float *sceneAngles, int count, float d_dist);

__global__ void addVote(int *d_accumSpace, int *d_votingPoint, int *d_votingNumber, int *d_votingAngle, int modelSize, int sceneSize);

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
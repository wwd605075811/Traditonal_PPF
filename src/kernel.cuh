#ifndef CIRCLE_H
#define CIRCLE_H

#pragma once
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>
#include <stdlib.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <vector>
#include<fstream>

#include <iostream>
using namespace std;

struct cuda_PPFInfo
{
    float x;
    float y;
    float z;
    float nomal_x;
    float nomal_y;
    float nomal_z;
};

struct PPInfo
{
    int pt1_id; //index of one point in modelWithNormal
    int pt2_id; //index of the other point in modelWithNormal

    int F1;
    int F2;
    int F3;
    int F4;

    float alpha;
};

struct cuda_PPFotherInfo
{
    float Min_angle;
    float Max_angle;
    int   nAngle;
    int   g_P1;
    int   g_P2;
    int   g_P3;
    int   g_P4;
    float   d_distance;
    int   max_hashIndex;
    int hashTableSize;
    int hashNum;
};

extern "C" {
    bool  InitCuda();
    void ModelCuda(cuda_PPFInfo* h_Pc, cuda_PPFInfo* h_Pc_copy,cuda_PPFotherInfo h_other, int* h_fromGPU_hash, float* h_fromGPU_alpha,int* h_fromGPU_F1,int* h_fromGPU_F2,int* h_fromGPU_F3,int* h_fromGPU_F4);
    void SceneCuda(cuda_PPFInfo* h_Pc,cuda_PPFInfo* h_Pc_Refer,cuda_PPFotherInfo h_other, int* h_fromGPU_hash, float* h_fromGPU_alpha,int* h_fromGPU_F1,int* h_fromGPU_F2,int* h_fromGPU_F3,int* h_fromGPU_F4);
    void Reference2NumCuda(cuda_PPFInfo* h_Pc, cuda_PPFInfo* h_Pc_copy,cuda_PPFotherInfo h_other, int* h_fromGPU_hash, float* h_fromGPU_alpha,int* h_fromGPU_F1,int* h_fromGPU_F2,int* h_fromGPU_F3,int* h_fromGPU_F4, PPInfo* modelHashValue,int* modelHashKeyIndex);
}



#endif
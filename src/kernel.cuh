#ifndef PPFMATCHING_CUDAPPF_KERNEL_H
#define PPFMATCHING_CUDAPPF_KERNEL_H
#pragma once
#include <stdio.h>
#include <iostream>
#include <vector>
#include<fstream>
#include <time.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
using namespace std;
struct CudaPPFInfo {
    float x;
    float y;
    float z;
    float nomal_x;
    float nomal_y;
    float nomal_z;
};

struct PPInfo {
    int pt1_id; //index of one point in modelWithNormal
    int pt2_id; //index of the other point in modelWithNormal
    int F1;
    int F2;
    int F3;
    int F4;
    float alpha;
};

struct CudaOtherInfo {
    float Min_angle;
    float Max_angle;
    int nAngle;
    int g_P1;
    int g_P2;
    int g_P3;
    int g_P4;
    float d_distance;
    int max_hashIndex;
    int hashTableSize;
    int hashNum;  //N*(N-1)
    int modelPointsNum;
};

extern "C" {
void ModelCuda(CudaPPFInfo *h_Pc, CudaPPFInfo *h_Pc_copy, CudaOtherInfo h_other, int *h_hash,
               float *h_alpha, int *h_F1, int *h_F2, int *h_F3, int *h_F4);

void Reference2NumCuda(CudaPPFInfo *h_Pc, CudaPPFInfo *h_Pc_copy, CudaOtherInfo h_other,
                       int *h_hash,float *h_alpha, int *h_F1, int *h_F2, int *h_F3,int *h_F4,
                       PPInfo *modelHashValue, int *modelHashKeyIndex, int *h_accumSpace);
}
#endif
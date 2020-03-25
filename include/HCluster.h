#ifndef PPF_HCLUSTER_H
#define PPF_HCLUSTER_H
#pragma once
#include<iostream>
#include<vector>
#include<map>
#include<cstdlib>
#include<algorithm>
#include<fstream>
#include<stdio.h>
#include<string.h>
#include<string>
#include<time.h>  //for srand
#include<limits.h> //for INT_MIN INT_MAX

#include <math.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
using namespace Eigen;
using namespace std;

class HCluster
{
public:
    HCluster();
    ~HCluster();

private:
    vector< vector<float> > dataSet;//the data set
    vector< int > votingValueSet;//votingValueList_
    int colLen, rowLen;//colLen:the dimension of vector;rowLen:the number of vectors
    int classNumber;
    vector< vector<float> > centroids;
    vector< int > clusterLabels;
    float angle_thresh;
    float trans_thresh;

public:
    void SetThresholds(float angle_thresh, float trans_thresh);
    void CreateDataSet(vector< vector<float> >&  transformationDataSet, vector<int>&  votingValueList_);
    void CreateCluster();
    float distEclud(vector<float> &v1, vector<float> &v2);
    bool isSimilar(vector<float> &v1, vector<float> &v2);
    void print();
    void GetCentroidsAndClusterLabels(vector< vector<float> >&  centroidList, vector< int >& clusterLabelList);

};

#endif //PPF_HCLUSTER_H

#include "../include/HCluster.h"

struct Vote {
    int id;
    int value;
};

bool VoteCompare(Vote v1, Vote v2) {
    return v1.value > v2.value;
}

HCluster::HCluster() {
}

HCluster::~HCluster() {
}

void HCluster::CreateDataSet(vector< vector<float> >&  transformationDataSet, vector<int>&  votingValueList_) {//从内存加载
    //init dataSet
    for(int i = 0; i<transformationDataSet.size(); i++) {
        dataSet.push_back(transformationDataSet[i]);
        votingValueSet.push_back(votingValueList_[i]);
    }

    //init colLen,rowLen
    colLen = 0;
    rowLen = 0;
    if (dataSet.size() > 0) {
        colLen = dataSet[0].size();  //quaternions + Trans_Mat =7.
        rowLen = dataSet.size();  //poses'size
        cout<< "In the CreateDataSet. colLen:" <<colLen <<" rowLen:" <<rowLen <<endl;
    }
}

void HCluster::SetThresholds(float angle_thresh, float trans_thresh) {
    this->angle_thresh = angle_thresh;
    this->trans_thresh = trans_thresh;
}

bool HCluster::isSimilar(vector<float> &v1, vector<float> &v2) {
    Vector4f angle1(v1[0], v1[1], v1[2], v1[3]);
    Vector4f angle2(v2[0], v2[1], v2[2], v2[3]);


    if ((angle1 - angle2).norm() > angle_thresh)
        return false;

    Vector3f trans1(v1[4], v1[5], v1[6]);
    Vector3f trans2(v2[4], v2[5], v2[6]);
    if ((trans1 - trans2).norm() > trans_thresh)
        return false;

    return true;
}

void HCluster::print() {
    ofstream fout;
    fout.open("res.txt");
    if (!fout)
    {
        cout << "file res.txt open failed" << endl;
        exit(0);
    }

    vector< vector<float> > ::iterator it = dataSet.begin();
    vector< int > ::iterator itt = clusterLabels.begin();
    for (int i = 0; i<rowLen; i++)
    {
        vector<float> ::iterator it2 = (*it).begin();
        while (it2 != (*it).end())
        {
            fout << *it2 << " ";
            it2++;
        }
        fout <<"   "<< (*itt)<< endl;
        itt++;
        it++;
    }
}

float HCluster::distEclud(vector<float> &v1, vector<float> &v2) {
    float sum = 0;
    int size = v1.size();
    for (int i = 0; i<size; i++)
    {
        sum += (v1[i] - v2[i])*(v1[i] - v2[i]);
    }
    return sum;
}

void HCluster::CreateCluster() {
    vector<Vote> voteList;
    for (int i = 0; i < rowLen; i++) {
        Vote tempVote;
        tempVote.id = i;
        tempVote.value = votingValueSet[i];
        voteList.push_back(tempVote);
    }
    // sort votelist by decreasing order
    std::sort(voteList.begin(), voteList.end(), VoteCompare);

    vector<vector<float> > tempDataSet;
    for (int i = 0; i<rowLen; i++) {
        int id = voteList[i].id;
        tempDataSet.push_back(dataSet[id]);
    }

    int cur_clusterID = -1;
    //
    vector<int> tempClusterList(rowLen, cur_clusterID);
    vector<vector<float> > tempCentroidList;

    for (int i = 0; i < rowLen; i++) { //for the each pose
        bool isFind = false;
        int findClusterID = -1;
        float dis = 9999999;
        for (int j = 0; j < tempCentroidList.size() ; j++) {//for the each cluster
            if (isSimilar(tempDataSet[i], tempCentroidList[j])) {
                isFind = true;
                float cur_dis = distEclud(tempDataSet[i], tempCentroidList[j]);
                if(isnan(cur_dis)) { //this nan is come from the scene point normal
                    isFind = false;
                    continue;
                }
                if(cur_dis < dis) { //one pose just go into one cluster
                    dis = cur_dis;
                    findClusterID = j;
                }
            }
        }
        if (isFind) {
            tempClusterList[i] = findClusterID;
            int cnt = 0;
            vector<float> vec(colLen, 0);

            // TODO: 待优化 ↓↓↓↓↓ ↑
            for (int id = 0; id < rowLen; id++) {
                if (tempClusterList[id] == findClusterID) {
                    ++cnt;
                    for (int col_id = 0; col_id < colLen; col_id++) {
                        vec[col_id] += tempDataSet[id].at(col_id);
                    }
                }
            }
            //mean of the vector and update the centroids[findClusterID]
            for (int col_id = 0; col_id < colLen; col_id++) {
                if (cnt != 0)	vec[col_id] /= cnt;
                tempCentroidList[findClusterID].at(col_id) = vec[col_id];
            }
            // ↑↑↑↑↑
        }
        else {
            cur_clusterID++;
            tempCentroidList.push_back(tempDataSet[i]);
            tempClusterList[i] = cur_clusterID;
        }
    }//for i

    clusterLabels.clear();
    for (int i = 0; i < rowLen; i++) {
        clusterLabels.push_back(-1);
    }
    for (int i = 0; i < rowLen; i++) {
        int id = voteList[i].id;
        clusterLabels[id]=(tempClusterList[i]);
    }

    classNumber = tempCentroidList.size();
    ///note:
    if( classNumber == 0 ) {
        cerr<< "the most possible reason is there aren't normals in pointCloud!" <<endl;
    }

    cout<<"Centroid size:"<< classNumber <<endl;
    centroids.clear();
    for (int i = 0; i < classNumber; i++) {
        centroids.push_back(tempCentroidList[i]);
    }

    tempCentroidList.clear();
    tempClusterList.clear();
    tempDataSet.clear();
    voteList.clear();

    print();
}

void HCluster::GetCentroidsAndClusterLabels(vector< vector<float> >&  centroidList, vector< int >& clusterLabelList) {
    centroidList.clear();
    clusterLabelList.clear();

    for (int i = 0; i<centroids.size(); i++) {
        centroidList.push_back(centroids[i]);
    }

    for (int i = 0; i<clusterLabels.size(); i++) {
        int clusterNo = clusterLabels[i];
        clusterLabelList.push_back(clusterNo);
    }
}
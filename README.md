## 环境搭建

1.安装nvidia驱动

2.安装基本的编译环境

```
sudo apt install build-essential cmake git
```

3.boost 安装

```
cd /tmp
wget https://dl.bintray.com/boostorg/release/1.71.0/source/boost_1_71_0.tar.gz
tar -xvf boost_1_71_0.tar.gz
cd boost_1_71_0
./bootstrap.sh --prefix=/usr/local  --with-python=/usr/bin/python3 --with-python-version=3.6 --with-python-root=/usr
sudo ./b2 install
```

​	创建软链接

```
sudo ln -s /usr/local/lib/libboost_numpy36.so /usr/local/lib/libboost_numpy3.so
sudo ln -s /usr/local/lib/libboost_python36.so /usr/local/lib/libboost_python3.so
```

4.PCL安装

```
//安装依赖项
sudo apt install libeigen3-dev libflann-dev
sudo apt install libvtk7-dev
//编译pcl 1.91
cd /tmp
wget https://github.com/PointCloudLibrary/pcl/archive/pcl-1.9.1.tar.gz
tar -xvf pcl-1.9.1.tar.gz
cd pcl-pcl-1.9.1
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
sudo make install
```

5.安装cuda10.2

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```

​		配置环境变量

```
sudo vim ~/.bashrc
//在末尾写入并保存
export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
//使其生效
source ~/.bashrc
```

​		测试CUDA

```
//First
nvidia-smi
//Second
nvcc -V
//Third
cd ~/NVIDIA_CUDA-10.0_Samples/1_Utilities/deviceQuery
sudo make
./deviceQuery
```

6.安装Eigen库

```
sudo apt install libeigen3-dev
//添加映射
sudo ln -s /usr/include/eigen3/Eigen  /usr/include/Eigen
//修改Eigen源文件 根据：https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/Core 24-27行
cd /usr/include/eigen3/Eigen
sudo vim Core
//修改42行的  #include <math_functions.hpp> 为： (新版这里做了更新)
#include<cuda_runtime.h>
```

7.修改cuda头文件

```
cd /usr/local/cuda/include/crt/
sudo vim common_functions.h
//修改74 75行为：
#define __CUDACC_VER_MAJOR /*"__CUDACC_VER__ is no longer supported.  Use __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, and __CUDACC_VER_BUILD__ instead."*/

```

## 使用：

1.下载

```
git clone -b cuda_ppf http://gitlab.qianyi.ai/lihk11/ppf-matching.git
```

2.从NAS下载dataTest数据集  

```
网址 http://nas.qianyi.ai:5000/
路径 /tf_shared/智能机器人部/CUDA_ppf测试点云/dataTest
复制到clone路径下，与data model modelScene sonor2018 在同一文件夹下
```

3.运行

```
cd ppf-matching
mkdir build && cd build
cmake ..
make -j4
./main
```



## PPF参数介绍与推荐：

```c++
详情请参阅参数文档
```
## 参考论文与项目:

[1] Bertram Drost et al. “Model globally, match locally: Efficient and robust 3D object recognition”. In: Computer Vision and Pattern Recognition (CVPR), 2010 IEEE Conference on. IEEE, 2010, pp. 998–1005.

[2]https://github.com/nicolasavru/ppf-registration-spatial-hashing    (GPU实现)
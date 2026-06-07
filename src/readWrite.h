//
// Created by 叶卓杨 on 2021/5/8.
//

#pragma once


#include <string>
#include <vector>
#include <Eigen/Core>

using namespace std;

#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/centroid.h>
#include <pcl/io/pcd_io.h>

class readWrite {
public:
    static bool readData(const string& filename, vector<Eigen::Vector3d>& points);

    static bool writeDate(const string& filename,const vector<Eigen::Vector3d>& points, bool append);
    static bool writeDate(const string &filename, const pcl::PointCloud<pcl::PointXYZ>::Ptr points, bool append);
};

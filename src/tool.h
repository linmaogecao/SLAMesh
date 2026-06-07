//
// Created by albus on 2026/2/15.
//

#ifndef SPLINE_FITTING_TOOL_H
#define SPLINE_FITTING_TOOL_H
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
typedef Eigen::Matrix4d Transf;
bool readKitti(const std::string & file_dataset, const std::string & seq, int line_num,
               pcl::PointCloud<pcl::PointXYZ> & laser_cloud,Eigen::Matrix<double, 3, Eigen::Dynamic>& tmp_point_expend);
void transformPoints(Transf & trans,Eigen::Matrix<double, 3, Eigen::Dynamic>& points);
#endif //SPLINE_FITTING_TOOL_H
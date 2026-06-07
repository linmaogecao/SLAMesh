//
// Created by 叶卓杨 on 2021/5/8.
//

#include "readWrite.h"

#include <fstream>
#include <iomanip>
#include <iostream>

using namespace Eigen;


bool readWrite::readData(const string &filename, vector<Vector3d>& points) {
    std::ifstream fin(filename.c_str());
    std::vector<double> coefficients;
    std::string line;
    int nRow = 0;
    int nCol = 0;
    if( fin.fail() )
        return false;


    while( std::getline( fin, line ) )
    {
        std::stringstream ss(line);
        double d = 0;
        nCol = 0;
        while( ss >> d)
        {
            coefficients.push_back( d);
            ++nCol;
        }
        ++nRow;
    }
    fin.close();

    if( nCol < 3)
        return false;

    points.resize( nRow);
    int idx = 0;
    for( int i = 0; i!= nRow; ++i )
    {
        points[i] = Vector3d( coefficients[idx+0], coefficients[idx+1], coefficients[idx+2]);
        idx += nCol;
    }
    return true;
}

bool readWrite::writeDate(const string &filename, const vector<Vector3d>& points, bool append) {
    // 根据 append 参数决定打开模式
    std::ios_base::openmode mode = std::ios::out;
    if (append) {
        mode |= std::ios::app; // 追加模式
    } else {
        mode |= std::ios::trunc; // 覆盖模式 (默认)
    }

    ofstream fout(filename.c_str(), mode);
    if( fout.fail() )
        return false;

    // 设置精度，防止浮点数记录不准
    fout << std::fixed << std::setprecision(6);

    for( int i = 0; i!= points.size(); ++i )
    {
        fout << points[i].x() << " " << points[i].y() << " " << points[i].z()  << endl;
    }
    fout.close();

    return true;
}

bool readWrite::writeDate(const string &filename, const pcl::PointCloud<pcl::PointXYZ>::Ptr points, bool append) {
    // 根据 append 参数决定打开模式
    std::ios_base::openmode mode = std::ios::out;
    if (append) {
        mode |= std::ios::app; // 追加模式
    } else {
        mode |= std::ios::trunc; // 覆盖模式 (默认)
    }

    ofstream fout(filename.c_str(), mode);
    if( fout.fail() )
        return false;

    // 设置精度，防止浮点数记录不准
    fout << std::fixed << std::setprecision(6);

    for( int i = 0; i!= points->size(); ++i )
    {
        const auto& p = points->points[i];
        fout << p.x << " " << p.y << " " << p.z << "\n";
    }
    fout.close();

    return true;
}
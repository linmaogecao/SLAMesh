//
// Created by albus on 2026/2/15.
//
#include "tool.h"

bool readKitti(const std::string & file_dataset, const std::string & seq, int line_num,
               pcl::PointCloud<pcl::PointXYZ> & laser_cloud,Eigen::Matrix<double, 3, Eigen::Dynamic>& tmp_point_expend){//partially refer to A-LOAM
    std::stringstream bin_point_cloud_path;
    bin_point_cloud_path << file_dataset << seq << "/velodyne/" << std::setfill('0') << std::setw(6) << line_num << ".bin";
    std::cout<<"path:"<<bin_point_cloud_path.str()<<std::endl;
    std::cout<<std::setprecision(7)<<setiosflags(std::ios::fixed);
    std::ifstream bin_point_cloud_file(bin_point_cloud_path.str(), std::ifstream::in | std::ifstream::binary);
    if(!bin_point_cloud_file.good()){
        return false;
    }
    bin_point_cloud_file.seekg(0, std::ios::end);
    const size_t num_elements = bin_point_cloud_file.tellg() / sizeof(float);
    bin_point_cloud_file.seekg(0, std::ios::beg);
    std::vector<float> lidar_data(num_elements);
    bin_point_cloud_file.read(reinterpret_cast<char*>(&lidar_data[0]), num_elements * sizeof(float));
    std::cout << "totally " << int(lidar_data.size() / 4.0) << " points in this lidar frame \n";

    std::vector<Eigen::Vector3d> lidar_points;
    std::vector<float> lidar_intensities;
    int num_points = lidar_data.size() / 4;
    tmp_point_expend.resize(3, num_points);
    laser_cloud.clear();
    laser_cloud.reserve(num_points);
    int idx = 0;
    for (std::size_t i = 0; i < lidar_data.size(); i += 4)
    {
        lidar_points.emplace_back(lidar_data[i], lidar_data[i+1], lidar_data[i+2]);
        lidar_intensities.push_back(lidar_data[i+3]);

        pcl::PointXYZ point;
        point.x = lidar_data[i];
        point.y = lidar_data[i + 1];
        point.z = lidar_data[i + 2];
        //if(point.z > -2.5){//there are some underground outliers in kitti dataset, remove them
        laser_cloud.push_back(point);
        tmp_point_expend(0, idx) = point.x;
        tmp_point_expend(1, idx) = point.y;
        tmp_point_expend(2, idx) = point.z;
        idx++;
        //}
    }
    // transformPoints(trans, tmp_point_expend);
    return true;
}

void transformPoints(Transf & trans, Eigen::Matrix<double, 3, Eigen::Dynamic>& points){
    //transform points
    if (trans.block<3,1>(0,3).isZero(1e-12)) {
        std::cerr << "ERROR: Wrong Trans!" << std::endl;
    }

    points =
        (trans.topLeftCorner<3,3>() * points).eval()
        + trans.topRightCorner<3,1>().replicate(1, points.cols());
}

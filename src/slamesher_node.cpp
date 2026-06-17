/* This code is the implementation of our paper "SLAMesh: Real-time LiDAR
 Simultaneous Localization and Meshing".

Author: Jianyuan Ruan, Email: <jianyuan.ruan@connect.polyu.hk>

If you find our research helpful, please cite our paper:
[1] Jianyuan Ruan, Bo Li, Yibo Wang, and Yuxiang Sun, "SLAMesh: Real-time
 LiDAR Simultaneous Localization and Meshing" ICRA 2023.

Other related papers:
[2] Jianyuan Ruan, Bo Li, Yinqiang Wang and Zhou Fang, "GP-SLAM+: real-time
 3D lidar SLAM based on improved regionalized Gaussian process map
 reconstruction," IROS 2020.
[3] Bo Li, Yinqiang Wang, Yu Zhang. Wenjie Zhao, Jianyuan Ruan, and Pin Li,
 "GP-SLAM: laser-based SLAM approach based on regionalized Gaussian process
 map reconstruction". Auton Robot 2020.

For commercial use, please contact Dr. Yuxiang SUN < yx.sun@polyu.edu.hk >.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from this
   software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

/***
 * This file run a SLAMesh node.
 * Please use this abbreviation dictionary to help you understand our code:
 * param: parameter
 * g_data: global data
 * trf: transformation, 6dof
 * odom: odometry
 * dir: direction, the coordinate that serve as the prediction in gaussian
    process function, like z in z = f(x, y).
 * num: number
 * Idx: index
 * 3Dir: inside one cell, there can be 3 different gaussian process function
    to model complex local surfaces, they have different prediction directions,
    that is, x, y, or z. Functions without "3Dir" means they only allow one
    gaussian process function inside one cell, like paper [3].
 * glb: global map, means in the world frame
 * now: means the current scan
 * avg: average
 * thr: threshold
 * posi: position
 * ary: array
 *
    */

# include "slamesher_node.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
Parameter param;//parameters
Log g_data;//global variables

Log::Log(){
    log_length =  param.max_steps;
    num_cells_now = num_cells_glb = num_cells_new =
    time_cost = time_cost_draw_map = time_find_overlap = time_find_overlap_points = time_pub_odom =
    time_get_pcl = time_gp = time_compute_rt = time_update = time_down_sample =
    not_a_surface_cell_num = overlap_point_num = gp_times = rg_times =
        Eigen::MatrixXd::Zero(1, log_length);
    pose = Eigen::MatrixXd::Zero(3, log_length);
    //first_transf = lidar_install_transf = grt_first_transf = transf_odom_now = transf_odom_last = Eigen::MatrixXd::Identity(4,4);
    path.header.frame_id      = "map";
    path_odom.header.frame_id = "map";
    path_grt.header.frame_id  = "map";
    pcl_raw_accumulated.height = 1;
    pcl_raw_accumulated.width = 0;
    g << 0.0, 0.0, 9.82;
}
void Log::extendLog(){
    if(step >= log_length){
        int extend_length = 5000;
        extendEigen1dVector(num_cells_now, extend_length);
        extendEigen1dVector(num_cells_glb, extend_length);
        extendEigen1dVector(num_cells_new, extend_length);

        extendEigen1dVector(time_cost, extend_length);
        extendEigen1dVector(time_cost_draw_map, extend_length);
        extendEigen1dVector(time_find_overlap, extend_length);
        extendEigen1dVector(time_find_overlap_points, extend_length);
        extendEigen1dVector(time_pub_odom, extend_length);
        extendEigen1dVector(time_get_pcl, extend_length);
        extendEigen1dVector(time_gp, extend_length);
        extendEigen1dVector(time_compute_rt, extend_length);
        extendEigen1dVector(time_update, extend_length);
        extendEigen1dVector(time_down_sample, extend_length);

        extendEigen1dVector(not_a_surface_cell_num, extend_length);
        extendEigen1dVector(overlap_point_num, extend_length);
        extendEigen1dVector(gp_times, extend_length);
        extendEigen1dVector(rg_times, extend_length);

        pose.conservativeResize(Eigen::NoChange_t(1), pose.cols() + extend_length);
        pose.rightCols(extend_length).fill(0);

        log_length += extend_length;
    }
}
void Log::initLog(std::string & log_file_path){
    //open file to record log
    ROS_DEBUG("Log::initLog");
    //init file path
    std::cout << "Try to save report in: " << log_file_path << std::endl;
    auto now = std::time(nullptr);
    char buf[sizeof("YYYY-MM-DD-HH:MM:SS")];
    std::string run_time (buf, buf + std::strftime(buf, sizeof(buf), "%F-%T", std::gmtime(&now)));
    std::string log_file_path_report   = log_file_path + "_" + run_time + "_report.txt";
    std::string log_file_path_odom     = log_file_path + "_" + run_time + "_path_odom.txt";
    std::string log_file_path_grt      = log_file_path + "_" + run_time + "_path_grt.txt";
    log_file_path_GP_map_points = log_file_path + "_" + run_time + "_map_point.pcd";
    log_file_path_raw_pcl       = log_file_path + "_" + run_time + "_raw_pcl.pcd";
    log_file_path.erase(log_file_path.end() - 5, log_file_path.end());
    //std::cout << log_file_path << std::endl;
    std::string log_file_path_path          = log_file_path + param.seq.substr(1, 2) + "_pred.txt";
    //open
    file_loc_report_wrt.open (log_file_path_report, std::ios::out);
    if(!file_loc_report_wrt){
        ROS_WARN("Can not open Report file");
    }
    file_loc_path_wrt.open(log_file_path_path, std::ios::out);
    if(!file_loc_path_wrt){
        ROS_WARN("Can not open Path file");
    }
    if(param.grt_available){
        file_loc_path_odom_wrt.open(log_file_path_odom, std::ios::out);
        if(!file_loc_path_grt_wrt){
            ROS_WARN("Can not open Path Grt file");
        }
    }
    if(param.odom_available){
        file_loc_path_grt_wrt.open(log_file_path_grt, std::ios::out);
        if(!file_loc_path_odom_wrt){
            ROS_WARN("Can not open Path Odom file");
        }
    }
}
void Log::saveResult(double code_whole_time, const PointMatrix & map_glb_point_filtered, Map &  map_glb){
    //save report
    ROS_DEBUG("saveResult");
    std::cout<<"Saving result" <<std::endl;
    double time_sum_all_step;
    time_sum_all_step = time_cost.leftCols(step).sum();//unit: ms, last step is not counted
    double sum_now_cells = num_cells_now.leftCols(step).sum();
    double raw_point_num_before_voxel_filter = -1;
    double steps_include_first = step+1 ;
    //print
    std::cout
            << "TIME ALL STEPS: " << (time_sum_all_step)/1000.0 << " s" << std::endl
            << "TRJ LENGTH    : " << trajectory_length << std::endl;
    std::cout << "average_time" << "\n"
              << "time_average       : " << time_cost.leftCols(step+1).sum()/steps_include_first << "\n"
              << "time_get_pcl       : " << time_get_pcl.leftCols(step+1).sum()/steps_include_first << "\n"
              << "time_down_sample   : " << time_down_sample.leftCols(step+1).sum()/steps_include_first << "\n"
              << "time_overlap_region: " << time_find_overlap.leftCols(step+1).sum()/steps_include_first <<"\n"
              << "time_overlap_points: " << time_find_overlap_points.leftCols(step+1).sum()/steps_include_first <<"\n"
              << "time_gp            : " << time_gp.leftCols(step+1).sum()/steps_include_first <<"\n"
              << "time_compute_rt    : " << time_compute_rt.leftCols(step+1).sum()/steps_include_first <<"\n"
              << "time_update        : " << time_update.leftCols(step+1).sum()/steps_include_first <<"\n"
              //<< "time_pub_odom      : " << time_pub_odom.leftCols(step+1).sum()/steps_include_first <<"\n"
              << "time_cost_draw_map : " << time_cost_draw_map.leftCols(step+1).sum()/steps_include_first <<"\n" ;
    //save report
    if(file_loc_report_wrt){
        file_loc_report_wrt
            << "step: " << step << "\n"
            << "TIME PER STEP:  " <<(time_sum_all_step)/(step-1) << " ms" << "\n"
            << "TIME ALL STEPS: " <<(time_sum_all_step)/1000.0 << " s" << "\n"
            << "TIME TOTAL RUN: " <<(code_whole_time)/1000.0 << " s" << "\n"
            << "NOW_CELL AVERAGE: " << (sum_now_cells) / step << "\n"
            << "DISPLACEMENT  : " << sqrt(pow(pose(0,step),2)+pow(pose(1,step),2)+pow(pose(2,step),2)) << "\n"

            << "time_cost: " << time_cost.leftCols(step+1).sum()/steps_include_first << "\n" << time_cost.leftCols(step+1) << "\n"
            << "time_get_pcl: " << time_get_pcl.leftCols(step+1).sum()/steps_include_first << "\n" << time_get_pcl.leftCols(step+1) << "\n"
            << "time_down_sample: " << time_down_sample.leftCols(step+1).sum()/steps_include_first << "\n" << time_down_sample.leftCols(step+1) << "\n"
            << "time_gp: " << time_gp.leftCols(step+1).sum()/steps_include_first << "\n" << time_gp.leftCols(step+1) << "\n"
            << "time_find_overlap: " << time_find_overlap.leftCols(step+1).sum()/steps_include_first << "\n" << time_find_overlap.leftCols(step+1) << "\n"
            << "time_find_overlap_points: " << time_find_overlap_points.leftCols(step+1).sum()/steps_include_first << "\n" << time_find_overlap_points.leftCols(step+1) << "\n"
            << "time_compute_rt: " << time_compute_rt.leftCols(step+1).sum()/steps_include_first << "\n" << time_compute_rt.leftCols(step+1) << "\n"
            << "time_update: " << time_update.leftCols(step+1).sum()/steps_include_first << "\n" << time_update.leftCols(step+1) << "\n"
            << "time_cost_draw_map: " << time_cost_draw_map.leftCols(step+1).sum()/steps_include_first << "\n" << time_cost_draw_map.leftCols(step+1) << "\n"

            << "num_cells_glb:" << num_cells_glb.leftCols(step + 1).sum() / steps_include_first << "\n" << num_cells_glb.leftCols(step + 1) << "\n"
            << "num_cells_now:" << num_cells_now.leftCols(step + 1).sum() / steps_include_first << "\n" << num_cells_now.leftCols(step + 1) << "\n"
            << "num_cells_new:" << num_cells_new.leftCols(step + 1).sum() / steps_include_first << "\n" << num_cells_new.leftCols(step + 1) << "\n"
            << "not_a_surface_cell_num:" << not_a_surface_cell_num.leftCols(step + 1).sum() / steps_include_first << "\n" << not_a_surface_cell_num.leftCols(step + 1) << "\n"
            << "gp_times:" << gp_times.leftCols(step + 1).sum() / steps_include_first << "\n" << gp_times.leftCols(step + 1) << "\n"
            << "overlap_point_num:" << overlap_point_num.leftCols(step+1).sum()/steps_include_first <<"\n" << overlap_point_num.leftCols(step+1) << "\n"
            << "register_times:" << rg_times.leftCols(step+1).sum()/steps_include_first <<"\n" << rg_times.leftCols(step+1) << "\n"
            << "pose: " << "\n" << pose.leftCols(step+1) << "\n"
            << std::endl;
        file_loc_report_wrt << "raw_point_num_before_voxel_filter: "<<raw_point_num_before_voxel_filter;
        std::cout << "Report saved in: " << param.file_loc_report << std::endl;
        file_loc_report_wrt.close();
    }
    else{
        std::cout <<"Result not saved, did you creat the folder?" << std::endl;
    }
    //save path
    savePath2TxtKitti(file_loc_path_wrt, path);
    if(param.grt_available){
        savePathTxt(file_loc_path_grt_wrt, path_grt);
    }
    //save mesh map
    if(param.save_mesh_map){
        //at the end, publish global mesh map, may cost seconds
        std::string file_loc_mesh_ply = param.file_loc_report + param.seq + "_mesh.ply";
        if(map_glb.outputMeshAsPly(file_loc_mesh_ply, map_glb.mesh_msg)){
            std::cout << "Mesh map saved in: " << file_loc_mesh_ply << std::endl;
        }
    }
    //save raw pcl
    if(param.save_raw_point_clouds){
        double voxel = 0.3;
        raw_point_num_before_voxel_filter = pcl_raw_accumulated.width;
        pcl::PointCloud<pcl::PointXYZ> raw_pcl_store;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPointer(new pcl::PointCloud<pcl::PointXYZ>);
        cloudPointer = pcl_raw_accumulated.makeShared();
        raw_pcl_store = pclVoxelFilter(cloudPointer, voxel);
        pcl::io::savePCDFileASCII(log_file_path_raw_pcl, raw_pcl_store);
    }
    g_data.file_loc_path_wrt.close();
    g_data.file_loc_path_grt_wrt.close();
}
void Log::pose_print(ros::Publisher & cloud_pub) const{//ok
    sensor_msgs::PointCloud cloud1 = matrix3DtoPclMsg(pose, step);
    cloud_pub.publish(cloud1);
}
void Log::savePath2TxtKitti(std::ofstream & file_out, nav_msgs::Path & path_msg){
    //write kitti format pose txt file, from path_msg, before write, transform to the camera frame using provided extrinsic
    if(file_out){
        for(int i = 0; i<= step-1; i++){
            Transf T_rectified;
            Transf T_velo2cam = Eigen::Matrix4d::Identity();
            if(param.seq == "/00" || param.seq == "/01" || param.seq == "/02" || param.seq == "/13" || param.seq == "/14" ||
               param.seq == "/15" || param.seq == "/16" || param.seq == "/17" || param.seq == "/18" || param.seq == "/19" ||
               param.seq == "/20" || param.seq == "/21" ) {
                T_velo2cam << 4.276802385584e-04,-9.999672484946e-01,-8.084491683471e-03,-1.198459927713e-02,
                              -7.210626507497e-03,8.081198471645e-03,-9.999413164504e-01,-5.403984729748e-02,
                              9.999738645903e-01,4.859485810390e-04,-7.206933692422e-03,-2.921968648686e-01,
                              0.0,0.0,0.0,1.0;
            }
            else if(param.seq == "/03") {
                T_velo2cam << 2.347736981471e-04, -9.999441545438e-01, -1.056347781105e-02, -2.796816941295e-03,
                              1.044940741659e-02, 1.056535364138e-02, -9.998895741176e-01, -7.510879138296e-02,
                              9.999453885620e-01, 1.243653783865e-04, 1.045130299567e-02, -2.721327964059e-01,
                              0.0,0.0,0.0,1.0;
            }
            else if(param.seq == "/04" || param.seq == "/05" || param.seq == "/06" || param.seq == "/07" || param.seq == "/08" ||
                    param.seq == "/09" || param.seq == "/10" || param.seq == "/11" || param.seq == "/12" ) {
                T_velo2cam << -1.857739385241e-03, -9.999659513510e-01, -8.039975204516e-03, -4.784029760483e-03,
                              -6.481465826011e-03, 8.051860151134e-03, -9.999466081774e-01, -7.337429464231e-02,
                              9.999773098287e-01, -1.805528627661e-03, -6.496203536139e-03, -3.339968064433e-01,
                              0.0,0.0,0.0,1.0;
            }
            T_rectified = T_velo2cam * T_seq[i] * T_velo2cam.inverse();
            //T_rectified = T_seq[i];
            file_out << T_rectified(0, 0) << " " << T_rectified(0, 1) << " " << T_rectified(0, 2) << " " << T_rectified(0, 3) << " "
                     << T_rectified(1, 0) << " " << T_rectified(1, 1) << " " << T_rectified(1, 2) << " " << T_rectified(1, 3) << " "
                     << T_rectified(2, 0) << " " << T_rectified(2, 1) << " " << T_rectified(2, 2) << " " << T_rectified(2, 3)
                     << "\n";
        }
    }
    else{
        std::cout<<"Can not open file: " <<"\n";
    }
}
void Log::savePathTxt(std::ofstream & file_out, nav_msgs::Path & path_msg){
    //write path msg into disk as txt file, only x y z
    if(file_out){
        for(const auto& path_pose : path_msg.poses){
            double roll, pitch, yaw;
            tf::Quaternion quat;
            tf::quaternionMsgToTF(path_pose.pose.orientation, quat);
            tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);

            file_out << path_pose.pose.position.x << ","
                     << path_pose.pose.position.y << ","
                     << path_pose.pose.position.z << ","
                     << roll << "," << pitch << "," << yaw << "\n";
        }
    }else {
        std::cout<<"Can not open file: " <<"\n";
    }
}
void Log::accumulateRawPoint(pcl::PointCloud<pcl::PointXYZ> pcl_raw, Transf& transf_this_step){
    //accumulate raw point cloud in the world frame
    //std::cout<<"num_point_all_raw_point last: \n"<<pcl_raw_accumulated.width * pcl_raw_accumulated.height<<"\n";//std::endl;
    static int skip_i = 0;
    int skip_step = 1;
    skip_i ++;
    if(skip_i % skip_step == 0){
        pcl::transformPointCloud(pcl_raw, pcl_raw, transf_this_step.cast<float>());
        pcl_raw_accumulated = pcl_raw_accumulated + pcl_raw;
        std::cout << "num_point_all_raw_point: " << pcl_raw_accumulated.width * pcl_raw_accumulated.height << "\n";//std::endl;
    }
}
Transf Log::initFirstTransf(){
    // get the first initial guess transform, can use it to correct the whole map when the sensor is not horizontal installed
    // usually grt and imu odometry are gravity aligned, so we can use them to correct the whole map
    ROS_DEBUG("initFirstTransf");

    Transf first_odom = Eigen::MatrixXd::Identity(4, 4);
    if(step != 0){
        ROS_ERROR("Incorrect use of initFirstTransf!");
        return first_odom;
    }
    //three ways
    //1, use ground truth
    if(param.grt_available){
        bool first = true;
        while(path_grt.poses.empty() && ros::ok()){
            ros::spinOnce();//wait for first imu
            if(first) ROS_INFO("Waiting for first grt...");
            usleep(100);
            first = false;
        }
        transf_odom_now = grt_first_transf;
    }
    //2, use first odom
    else if(param.odom_available){
        if(param.read_offline_pcd){
            transf_odom_now = state2quat2trans3(g_data.odom_offline[0]);
        }
        else{
            bool first = true;
            while(imu_msg_buf.empty() && odometry_msg_buf.empty() && ros::ok()){
                ros::spinOnce();//wait for first imu
                if(first) ROS_INFO("Waiting for first odom or imu...");
                usleep(100);
                first = false;
            }
            // imu or odometry callback function will update transf_odom_now
        }
    }
    //3, give a manual T
    else{
        transf_odom_now = createTrans(param.correction_x, param.correction_y, param.correction_z,
                                   param.correction_roll_degree, param.correction_pitch_degree,
                                   param.correction_yaw_degree);
    }
    if_first_trans_init = true;//stop update grt_first_transf

    //set odom
    g_data.recordPoseToPath(Odom, g_data.transf_odom_now);//save path

    //set imu start point
    imu_Bg.fill(0);
    imu_Ba.fill(0);
    imu_pos = transf_odom_now.block(0, 3, 3, 1);
    imu_rot = transf_odom_now.block(0, 0, 3, 3);
    imu_vel.fill(0);
    imu_init = true;

    //result
    transf_odom_last = transf_odom_now;//once read
    std::cout << "first_transf:\n" << transf_odom_now << "\n";
    return transf_odom_now;
}
void Log::updatePose(Transf & now_slam_transf){
    //After scan registration, if use imu, here try to compensate the bias of imu. If no imu, just save transformation
    ROS_DEBUG("updatePose");
    static Transf last_imu_transf, now_imu_transf, last_slam_transf;
    static double bias_sum_x, bias_sum_y, update_count;
    static double last_time;
    static bool is_first = true;

    static std::queue<double> ba_x;
    static std::queue<double> ba_y;
    int fix_lag = 1000;//50Hz = 20s

    //store Tsep
    T_seq.push_back(now_slam_transf);
    Point tmp_pose;
    tmp_pose = trans3Dpoint(0, 0, 0, now_slam_transf);
    pose.col(step) = tmp_pose;
    if(step > 0){
        trajectory_length += sqrt(pow(tmp_pose(0, 0) - pose(0, step - 1), 2) +
                pow(tmp_pose(1, 0) - pose(1, step - 1), 2) +
                pow(tmp_pose(2, 0) - pose(2, step - 1), 2));
    }

    recordPoseToPath(Slam, now_slam_transf);
    //save path and grt_path to txt
    //savePathEveryStep2Txt(file_loc_path_gdt_wrt, path_grt);
    //savePathEveryStep2Txt(file_loc_path_wrt, path);

    //update imu translation and imu bias
    if(!param.read_offline_pcd && param.imu_feedback){

        now_imu_transf << imu_rot, imu_pos, 0, 0, 0, 1;
        if(is_first){
            last_time  = ros::Time::now().toSec();
            last_time  = g_data.pcl_msg_buff.header.stamp.toSec();
            //last_time = imu_msg_buf.back()->header.timestamp.toSec();
            last_imu_transf = now_imu_transf;
            last_slam_transf = now_slam_transf;
            bias_sum_x = bias_sum_y = update_count = 0;
            is_first = false;
            imu_vel.fill(0);
        }
        else{
            //update imu transformPoints
            imu_pos(0, 0) = tmp_pose(0, 0);
            imu_pos(1, 0) = tmp_pose(1, 0);
            //imu_vel.fill(0);

            //save imu pose every time update and predict
            Point tmp_point;
            tmp_point << g_data.imu_pos.topRows(2), 0;
            g_data.pose_imu.addPoint(tmp_point);

            //update imu bias
            double now_time  = g_data.pcl_msg_buff.header.stamp.toSec();
            double dt = now_time - last_time;
            if(dt == 0){
                dt = 0.1;
            }
            double ds_slam_x = now_slam_transf(0, 3) - last_slam_transf(0, 3),
                    ds_slam_y = now_slam_transf(1, 3) - last_slam_transf(1, 3);
            double ds_imu_x  = now_imu_transf(0, 3) - last_imu_transf(0, 3),
                    ds_imu_y  = now_imu_transf(1, 3) - last_imu_transf(1, 3);
            double ds_x = ds_slam_x - ds_imu_x,
                    ds_y = ds_slam_y - ds_imu_y;

            //imu_vel(0, 0) = ds_slam_x / dt;
            //imu_vel(1, 0) = ds_slam_y / dt;

            bool fix_lag_ba_estimate = false;//false true
            if(fix_lag_ba_estimate){
                if(ba_x.size() < fix_lag){
                    ba_x.push(- ds_x * 2 / pow(dt, 2));
                    ba_y.push(- ds_y * 2 / pow(dt, 2));
                    bias_sum_x += ba_x.back();
                    bias_sum_y += ba_y.back();
                }
                else{
                    while(ba_x.size() >= fix_lag){
                        bias_sum_x -= ba_x.front();
                        bias_sum_y -= ba_y.front();
                        ba_x.pop();
                        ba_y.pop();
                    }
                    ba_x.push(- ds_x * 2 / pow(dt, 2));
                    ba_y.push(- ds_y * 2 / pow(dt, 2));
                    bias_sum_x += ba_x.back();
                    bias_sum_y += ba_y.back();
                    imu_Ba(0, 0) = bias_sum_x / ba_x.size();
                    imu_Ba(1, 0) = bias_sum_y / ba_y.size();
                }
                std::cout << "bias: x " << imu_Ba(0, 0) << "   y: " << imu_Ba(1, 0) << "   dt: " << dt << "\n";
            }
            else{
                update_count ++;
                bias_sum_x += - ds_x * 2 / pow(dt, 2);
                bias_sum_y += - ds_y * 2 / pow(dt, 2);
                imu_Ba(0, 0) = bias_sum_x / update_count;
                imu_Ba(1, 0) = bias_sum_y / update_count;
                std::cout << "bias: x " << imu_Ba(0, 0) << "   y: " << imu_Ba(1, 0) << "   dt: " << dt << "\n";
            }

            last_time = now_time;
            last_imu_transf = now_imu_transf;
            last_slam_transf = now_slam_transf;
        }
    }
}

void Parameter::initParameter(ros::NodeHandle & nh){
    //<!--  read param  -->
    nh.param("slamesher/grt_available", grt_available, false);
    nh.param("slamesher/odom_available", odom_available, false);
    nh.param("slamesher/read_offline_pcd", read_offline_pcd, false);
    nh.param("slamesher/imu_feedback", imu_feedback, false);//? TO DO
    nh.param("slamesher/file_loc_dataset", file_loc_dataset, std::string("/not_set"));
    nh.param("slamesher/dataset", dataset, 6);
    nh.param("slamesher/seq", seq, std::string(""));
    nh.param("slamesher/max_steps", max_steps, 1);
    nh.param("slamesher/max_frames", max_frames, 0);
    nh.param("slamesher/dump_frame", dump_frame, 0);
    std::cout<<"max_steps: "<<max_steps<<std::endl;
    std::cout<<"max_frames: "<<max_frames<<(max_frames > 0 ? " (debug stop)" : " (run full sequence)")<<std::endl;
    std::cout<<"dump_frame: "<<dump_frame<<(dump_frame > 0 ? " (save debug point clouds)" : " (off)")<<std::endl;
    std::cout<<"ground_w_rp: "<<ground_w_rp<<"  ground_w_z: "<<ground_w_z<<std::endl;
    nh.param("slamesher/file_loc_report", file_loc_report, std::string("not_set"));
    nh.param("slamesher/save_mesh_map", save_mesh_map, false);
    nh.param("slamesher/save_surface_samples", save_surface_samples, false);

    //<!--  register param  -->-
    nh.param("slamesher/range_max",  range_max,  100.0);
    nh.param("slamesher/range_min",  range_min,  1.0);
    nh.param("slamesher/range_unit", range_unit, 1.0);
    nh.param("slamesher/register_times", register_times, 5);
    nh.param("slamesher/ground_w_rp", ground_w_rp, 0.0);
    nh.param("slamesher/ground_w_z",  ground_w_z,  0.0);
    nh.param("slamesher/cross_overlap", cross_overlap, false);
    nh.param("slamesher/cross_cell_overlap_length", cross_cell_overlap_length, 0);
    nh.param("slamesher/num_margin_old_cell", num_margin_old_cell, -1);
    nh.param("slamesher/point2mesh", point2mesh, false);
    nh.param("slamesher/residual_combination", residual_combination, true);
    //<!--  visualize parameter  -->
    nh.param("slamesher/meshing_tsdf", meshing_tsdf, false);
    nh.param("slamesher/full_cover", full_cover, false);
    nh.param("slamesher/visualisation_type", visualisation_type, 1);

    //<!--  gp param  -->
    nh.param("slamesher/num_thread", num_thread, 1);
    nh.param("slamesher/grid", grid, 1.0);
    nh.param("slamesher/min_points_num_to_gp", min_points_num_to_gp, 8);
    nh.param("slamesher/num_test",  num_test, 10);
    //nh.param("slamesher/voxel_size", voxel_size, 0.05);
    voxel_size = grid*1.0/num_test;
    std::cout<<"voxel_size: "<<voxel_size<<std::endl;

    nh.param("slamesher/variance_register", variance_register, 0.1);
    nh.param("slamesher/variance_map_update", variance_map_update, 0.1);
    nh.param("slamesher/variance_map_show", variance_map_show, 0.1);
    nh.param("slamesher/variance_min", variance_min, 5.0);
    nh.param("slamesher/variance_sensor", variance_sensor, 0.1);

    nh.param("slamesher/test_param", test_param, 0.0);

    nh.param("slamesher/correction_x", correction_x, 0.0);
    nh.param("slamesher/correction_y", correction_y, 0.0);
    nh.param("slamesher/correction_z", correction_z, 0.0);
    nh.param("slamesher/correction_roll_degree",   correction_roll_degree, 0.0);
    nh.param("slamesher/correction_pitch_degree", correction_pitch_degree, 0.0);
    nh.param("slamesher/correction_yaw_degree",     correction_yaw_degree, 0.0);

    eigen_1 = 48;
    eigen_2 = 0.95;
    eigen_3 = 0.2;
    converge_thr = 0.00001;
    std::cout<<"====ROS INIT DONE===="<<std::endl;
}

Transf SLAMesher::getOdom(){
    //before scan registration, obtain initial guess of transformation from motion prior or odometry msg
    Transf odom, dT, odom_now, odom_pre;
    if(param.odom_available){
        if(param.read_offline_pcd){
            //use odometry from file
            int step_offset = 0;
            g_data.transf_odom_now =  state2quat2trans3(g_data.odom_offline[g_data.step + step_offset]);}
        else{
            //use odometry from topic
            ros::spinOnce();
        }

        odom_now = g_data.transf_odom_now;
        odom_pre = g_data.transf_odom_last;
        dT = odom_pre.inverse() * odom_now;
        odom.block(0, 0, 3, 3) = odom.block(0, 0, 3, 3) * dT.block(0, 0, 3, 3);//only use orientation of imu odom
        odom.block(0, 3, 1, 3) = g_data.transf_odom_now.block(0, 3, 1, 3);
        //store odom_offline path
        g_data.recordPoseToPath(Odom, odom);
        g_data.transf_odom_last = g_data.transf_odom_now;//once the odom_offline was read, the odom_buff_last will be updated
    }
    else{
        if(g_data.step == 1){
            odom = g_data.T_seq[g_data.step - 1];
        }
        else if (g_data.step > 1){
            //const motion prior
            odom = g_data.T_seq[g_data.step - 1] * g_data.T_seq[g_data.step - 2].inverse() * g_data.T_seq[g_data.step - 1];
            // no motion prior
            // odom = g_data.T_seq[g_data.step-1];
        }
    }
    std::cout << "Pose Odom:" << "x: " << odom(0, 3) << "  y: " << odom(1, 3) << "  z: " << odom(2, 3) << "\n";
    return odom;//use as initial guess
}
void SLAMesher::imuIntegration(const sensor_msgs::ImuConstPtr & imu_msg){
    //a very simple integration of imu to provide odometry, neglect this part if no imu is used
    //ROS_INFO("imuIntegration");
    static double time_last;

    double time_now = imu_msg->header.stamp.toSec();
    if( !g_data.receive_first_imu){
        //initialize
        g_data.imu_pos.fill(0);
        g_data.imu_vel.fill(0);
        g_data.imu_Ba << param.bias_acc_x, param.bias_acc_y, 0;

        bool uav_cl = false;// false
        if(uav_cl){
            //only used in ZJU UAV dataset because the initial velocity is not zero
            g_data.imu_pos << 0.276397, -18.4999, 7.92001;
            g_data.imu_vel << -0.2932305336, -0.586515367031, 0.180226743221;
            g_data.imu_Ba << param.bias_acc_x, param.bias_acc_y, 0;
        }

        double dx = imu_msg->linear_acceleration.x;
        double dy = imu_msg->linear_acceleration.y;
        double dz = imu_msg->linear_acceleration.z;
        Eigen::Vector3d linear_acceleration{dx, dy, dz};

        g_data.acc_0 = linear_acceleration;

        tf::Quaternion temp_quaternion;
        tf::quaternionMsgToTF(imu_msg->orientation, temp_quaternion);
        tf::Matrix3x3 matrix(temp_quaternion);

        g_data.imu_rot << matrix[0][0], matrix[0][1], matrix[0][2],
                matrix[1][0], matrix[1][1], matrix[1][2],
                matrix[2][0], matrix[2][1], matrix[2][2];

        time_last = time_now;
        g_data.receive_first_imu = true;
    }
    else{
        double dt = time_now - time_last;
        time_last = time_now;

        double dx = imu_msg->linear_acceleration.x;
        double dy = imu_msg->linear_acceleration.y;
        double dz = imu_msg->linear_acceleration.z;
        Eigen::Vector3d linear_acceleration{dx, dy, dz};

        Eigen::Vector3d _un_acc_0;
        _un_acc_0 = g_data.imu_rot * (g_data.acc_0 - g_data.imu_Ba) - g_data.g;

        tf::Quaternion quat;
        tf::quaternionMsgToTF(imu_msg->orientation, quat);
        tf::Matrix3x3 matrix(quat);
        g_data.imu_rot << matrix[0][0], matrix[0][1], matrix[0][2],
                matrix[1][0], matrix[1][1], matrix[1][2],
                matrix[2][0], matrix[2][1], matrix[2][2];

        Eigen::Vector3d _un_acc_1;
        _un_acc_1 = g_data.imu_rot * (linear_acceleration - g_data.imu_Ba) - g_data.g;


        Eigen::Vector3d _un_acc = 0.5 * (_un_acc_0 + _un_acc_1);

        g_data.imu_pos = g_data.imu_pos + dt * g_data.imu_vel + 0.5 * dt * dt * _un_acc;
        g_data.imu_vel = g_data.imu_vel + dt * _un_acc;
        g_data.acc_0 = linear_acceleration;
    }

    //save imu pose every time update and predict
    Point tmp_point;
    tmp_point << g_data.imu_pos.topRows(2), 0;
    g_data.pose_imu .addPoint(tmp_point);

    g_data.transf_odom_now.block(0, 0, 3, 3) = g_data.imu_rot;
    g_data.transf_odom_now(0, 3) = g_data.imu_pos(0, 0);
    g_data.transf_odom_now(1, 3) = g_data.imu_pos(1, 0);
}
void SLAMesher::groundTruthCallback(const geometry_msgs::PoseStamped::ConstPtr & ground_truth_msg){
    //used in motion capture system
    //the ground truth is only use to align the first frame with the ground frame, and save both ground truth and slam path for easy comparison.
    ROS_DEBUG("GroundTruth seq: [%d]", ground_truth_msg->header.seq);

    //update grt_first_transf
    if(!g_data.if_first_trans_init){
        tf::Quaternion quat;
        tf::quaternionMsgToTF(ground_truth_msg->pose.orientation, quat);
        tf::Matrix3x3 matrix(quat);
        g_data.grt_first_transf << matrix[0][0], matrix[0][1], matrix[0][2], ground_truth_msg->pose.position.x,
                matrix[1][0], matrix[1][1], matrix[1][2], ground_truth_msg->pose.position.y,
                matrix[2][0], matrix[2][1], matrix[2][2], ground_truth_msg->pose.position.z,
                0, 0, 0, 1;
    }
    //save grt path
    g_data.path_grt.poses.push_back(*ground_truth_msg);
}
void SLAMesher::groundTruthUavCallback(const nav_msgs::Odometry::ConstPtr & odom_msg){
    //used in uav system
    ROS_DEBUG("GroundTruthUAV seq: [%d]", odom_msg->header.seq);
    geometry_msgs::PoseStamped ground_truth_msg;
    ground_truth_msg.pose = odom_msg->pose.pose;
    ground_truth_msg.header = odom_msg->header;
    //ROS_INFO("GroundTruthUav UAV seq: [%d]", ground_truth_msg.header.seq);

    //update grt_first_transf
    if(!g_data.if_first_trans_init) {
        tf::Quaternion quat;
        tf::quaternionMsgToTF(ground_truth_msg.pose.orientation, quat);
        tf::Matrix3x3 matrix(quat);
        g_data.grt_first_transf << matrix[0][0], matrix[0][1], matrix[0][2], ground_truth_msg.pose.position.x,
                matrix[1][0], matrix[1][1], matrix[1][2], ground_truth_msg.pose.position.y,
                matrix[2][0], matrix[2][1], matrix[2][2], ground_truth_msg.pose.position.z,
                0, 0, 0, 1;
    }
    //save grt path
    g_data.path_grt.poses.push_back(ground_truth_msg);
    Point tmp_point;
    tmp_point << ground_truth_msg.pose.position.x, ground_truth_msg.pose.position.y, ground_truth_msg.pose.position.z;
    g_data.pose_grt.addPoint(tmp_point);
    //std::cout<<"size_of_pose_grt"<<g_data.path_grt.poses.size()<<std::endl;
}
void SLAMesher::altitudeCallback(const std_msgs::Float64 & alt_msg){
    //receive altitude msg from barometer
    //ROS_INFO("alt_msg");
    static double last_alt = 0;
    static bool first_alt = true;
    double now_alt = alt_msg.data, delta_alt;
    if(first_alt){
        last_alt = now_alt;
        first_alt = false;
        g_data.if_first_alt = true;
    }
    else{
        delta_alt = now_alt - last_alt;
        g_data.transf_odom_now(2, 3) = g_data.transf_odom_now(2, 3) + delta_alt;
        last_alt = now_alt;
    }
}
void SLAMesher::imuCallback(const sensor_msgs::Imu::ConstPtr & imu_msg){
    //receive imu data
    g_data.imu_msg_buf.push(imu_msg);
    imuIntegration(imu_msg);
}
void SLAMesher::odomCallback(const nav_msgs::Odometry::ConstPtr & odom_msg){
    //receive odometry data
    ROS_INFO("Odometry seq: [%d]", odom_msg->header.seq);
    g_data.odometry_msg_buf.push(odom_msg);
    g_data.transf_odom_now = PoseWithCovariance2transf(odom_msg->pose);
}
void SLAMesher::pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr & pcl_msg){
    //receive point cloud
    ROS_INFO("PointCloud seq: [%d]", pcl_msg->header.seq);
    g_data.pcl_msg_buff_deque.push_back(*pcl_msg);
}
bool SLAMesher::visualize(Map & map_glb, Map & map_now, int option){
    //visualization. some types may be a heavy load for SLAMesh or rviz.
    TicToc t_visualize;
    // publish path
    posePrint(pose_pub, g_data.pose, g_data.step);
    //posePrint(pose_imu_pub, g_data.pose_imu.point, g_data.pose_imu.num_point);
    path_pub.     publish(g_data.path);
    path_odom_pub.publish(g_data.path_odom);
    path_grt_pub. publish(g_data.path_grt);

//    Because mesh-tools rviz plugin do not support incremental mesh intersection, three visualization mode are provided
//    visualisation_type:
//    0, publish registered raw_points_in_world, like fast-lio, lio sam
//    1, + publish the vertices of mesh as point cloud, each scan + (1/n) map global
//    2, + visualize local updated mesh, each scan
//    3, + visualize global mesh, (1/n) frame
//    after finish whole process, the global mesh map will be visualized in any mode

    //pub current scan
    //pub aligned raw points in the world frame
    scanPrint3D(raw_points_in_world_pub, map_now.points_turned, 1);

    if(option == 0){
        return true;
    }

    //pub current scan vertices as pcl
    map_now.filterVerticesByVariance(param.variance_register);
    scanPrint3D(map_vertices_now_pub, map_now.vertices_filted, 1);

    //pub overlapped vertices in registration, for debug
    //only overlap, for debug
    //map_glb.filterVerticesByVarianceOverlap(param.variance_map_show, map_now);//needed?
    //scanPrint3D(overlap_point_glb, map_glb.vertices_filted, 1);
    //scanPrint3D(overlap_point_now, map_now.vertices_filted, 1);

    if(option == 2 || option == 3){
        // mesh_tool updated cells
        map_now.filterMeshLocal();
        mesh_pub_local.publish(map_now.mesh_msg);
    }

    // after each pub_map_glb_count frame, publish the global map
    static int pub_map_glb_count = 0;
    int skip_map_glb_pub = 50, skip_map_glb_point = 1;
    pub_map_glb_count ++;
    if(pub_map_glb_count % skip_map_glb_pub == 0 ){//&& g_data.trajectory_length > 0
        if(option == 1){
            //all map_glb pcl
            map_glb.filterVerticesByVariance(param.variance_map_show);
            scanPrint3D(map_vertices_glb_pub, map_glb.vertices_filted, skip_map_glb_point);
        }
        if(option == 3){
            // mesh_tool total map
            map_glb.filterMeshGlb();
            mesh_pub.publish(map_glb.mesh_msg);
        }

        //for debug: normal
        //visualizeArrow(arrow_pub, map_glb.ary_overlap_vertices[0], map_glb.ary_overlap_vertices[1]);
        //scanPrint3D(overlap_point_glb, map_glb.ary_overlap_vertices[1], 1);
        pub_map_glb_count = 0;
    }
    std::cout << "t_visualize: " << t_visualize.toc() << "ms" << std::endl;
    return true;
}
void SLAMesher::pubTf(){
    //publish tf and odometry message from
    Transf transf_now = g_data.T_seq[g_data.step];
    //pub odometry msg
    nav_msgs::Odometry odom_msg;
    odom_msg.header.frame_id = "/map";
    odom_msg.child_frame_id = "/slamesher_odom";
    if(param.read_offline_pcd){
        ros::Time now_time = ros::Time::now();
        odom_msg.header.stamp = now_time;
    }
    else{
        odom_msg.header.stamp = g_data.pcl_msg_buff.header.stamp;
    }
    odom_msg.pose = transf2PoseWithCovariance(transf_now);
    odom_pub.publish(odom_msg);
    //pub tf
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odom_msg.pose.pose.position.x,
                                    odom_msg.pose.pose.position.y,
                                    odom_msg.pose.pose.position.z));
    q.setW(odom_msg.pose.pose.orientation.w);
    q.setX(odom_msg.pose.pose.orientation.x);
    q.setY(odom_msg.pose.pose.orientation.y);
    q.setZ(odom_msg.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odom_msg.header.stamp, "/map", "/slamesher_odom"));
}

// ========== process 子步骤 ==========

int SLAMesher::chooseControlGridSize(int num_fitting_points)
{
    constexpr int kMinCp = 4;
    constexpr int kMaxCp = 15;

    int n;
    if (num_fitting_points < 50)
        n = 4;
    else if (num_fitting_points < 100)
        n = 5;
    else if (num_fitting_points < 150)
        n = 6;
    else if (num_fitting_points < 250)
        n = 8;
    else if (num_fitting_points < 400)
        n = 10;
    else if (num_fitting_points < 700)
        n = 12;
    else
        n = 15;

    return std::max(kMinCp, std::min(n, kMaxCp));
}

void SLAMesher::processFirstFrame(const pcl::PointCloud<pcl::PointXYZ>& scan_local,
                                  Transf& T_world,
                                  RangeImageProcessor& range_proc,
                                  BSplineMap& bspline_map,
                                  double& z_ground_ref,
                                  double ground_z_thr)
{
    pcl::PointCloud<pcl::PointXYZ> scan_world;
    pcl::transformPointCloud(scan_local, scan_world, T_world.cast<float>());

    range_proc.generateRangeImage(scan_world);
    SegmentationResult seg = range_proc.segmentRangeImage(5, 0.1, 30);
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters = range_proc.generateClusterClouds(seg);
    range_proc.saveClustersToTxt(seg, "/home/albus/slam-math/Bspline/build/output_clusters");

    for (int cid = 0; cid < (int)clusters.size(); cid++) {
        if (!clusters[cid] || clusters[cid]->size() < 20) continue;

        const int num_cp = chooseControlGridSize((int)clusters[cid]->size());
        auto init_cp = range_proc.computeInitControlPoints(seg, cid, num_cp, num_cp, 2);
        if (init_cp.empty()) continue;

        auto surf = std::make_shared<BSplineSurface>(3, 3, num_cp, num_cp, 0.25);

        surf->setExternalInitControls(init_cp);
        surf->apply(clusters[cid], 50, 1, 1, 0.05);
        bspline_map.addSurface(surf, clusters[cid]);
    }

    if (std::isnan(z_ground_ref)) {
        double sum_z = 0.0;
        int cnt = 0;
        for (auto& pt : scan_local) {
            if (pt.z < ground_z_thr) { sum_z += pt.z; cnt++; }
        }
        z_ground_ref = (cnt > 10) ? (sum_z / cnt) : -1.73;
        std::cout << "  [Ground] z_ground_ref = " << z_ground_ref << " m" << std::endl;
    }

    g_data.updatePose(T_world);
    pubTf();
    path_pub.publish(g_data.path);
}

SLAMesher::GroundConstraint SLAMesher::computeGroundConstraint(
    const pcl::PointCloud<pcl::PointXYZ>& scan_local,
    double z_ground_ref,
    double ground_z_thr) const
{
    GroundConstraint gnd;
    gnd.c_local = Eigen::Vector3d(0, 0, z_ground_ref);

    if (std::isnan(z_ground_ref)) return gnd;

    std::vector<Eigen::Vector3d> gpts;
    gpts.reserve(2000);
    for (size_t gi = 0; gi < scan_local.size(); gi += 2) {
        if (scan_local[gi].z < ground_z_thr)
            gpts.emplace_back(scan_local[gi].x, scan_local[gi].y, scan_local[gi].z);
    }
    if ((int)gpts.size() < 20) return gnd;

    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (auto& p : gpts) centroid += p;
    centroid /= (double)gpts.size();

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (auto& p : gpts) {
        Eigen::Vector3d d = p - centroid;
        cov += d * d.transpose();
    }
    cov /= (double)gpts.size();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
    Eigen::Vector3d n = eig.eigenvectors().col(0);
    if (n.z() < 0) n = -n;

    gnd.n_local = n;
    gnd.c_local = centroid;
    gnd.valid   = true;
    return gnd;
}

Transf SLAMesher::registerScanToMap(const pcl::PointCloud<pcl::PointXYZ>& scan_local,
                                    Transf T_guess,
                                    BSplineMap& bspline_map,
                                    const GroundConstraint& ground,
                                    double z_ground_ref,
                                    double ground_w_rp,
                                    double ground_w_z,
                                    int max_iters,
                                    double converge_thr,
                                    double match_dist_thr,
                                    int skip_points,
                                    double match_min_z)
{
    Transf T_curr = T_guess;
    double delta_scale = 100.0;
    std::vector<RegMatch> last_matches;

    for (int iter = 0; iter < max_iters && delta_scale > converge_thr; iter++) {
        std::unordered_map<int, std::vector<int>> surf_to_pts;
        for (int i = 0; i < (int)scan_local.size(); i += skip_points) {
            if (scan_local[i].z < match_min_z) continue;
            Eigen::Vector3d p_w = T_curr.block<3,3>(0,0) *
                Eigen::Vector3d(scan_local[i].x, scan_local[i].y, scan_local[i].z) +
                T_curr.block<3,1>(0,3);
            for (int sid : bspline_map.queryCandidates(p_w, 1))
                surf_to_pts[sid].push_back(i);
        }

        std::vector<RegMatch> matches;
        matches.reserve(scan_local.size() / skip_points);

        for (auto& [sid, indices] : surf_to_pts) {
            const BSplineMapEntry* entry = bspline_map.getEntry(sid);
            if (!entry || !entry->surface) continue;
            auto& surf = entry->surface;

            const auto& knU = surf->getKnotsU();
            const auto& knV = surf->getKnotsV();
            const auto& cps = surf->getControls();
            int num_cpv     = surf->getNumCpV();

            std::vector<Eigen::Vector3d> pts_world;
            pts_world.reserve(indices.size());
            for (int idx : indices) {
                pts_world.push_back(
                    T_curr.block<3,3>(0,0) *
                    Eigen::Vector3d(scan_local[idx].x, scan_local[idx].y, scan_local[idx].z) +
                    T_curr.block<3,1>(0,3));
            }

            std::vector<std::pair<BSplineSurface::Parameter, BSplineSurface::Parameter>> footprints;
            std::vector<double> dists;
            surf->findFootPrint(pts_world, footprints, dists);

            for (int k = 0; k < (int)indices.size(); k++) {
                double d = std::sqrt(std::abs(dists[k]));
                if (d > match_dist_thr || d < 1e-6) continue;

                auto [paraU, paraV] = footprints[k];
                SurfaceCurvature curv = surf->getCurvature(paraU, paraV, knU, knV, cps, num_cpv);
                if (curv.normal.norm() < 1e-9) continue;
                curv.normal.normalize();

                RegMatch m;
                m.p_local   = Eigen::Vector3d(scan_local[indices[k]].x,
                                              scan_local[indices[k]].y,
                                              scan_local[indices[k]].z);
                m.p_world   = pts_world[k];
                m.curvature = curv;
                m.scan_idx  = indices[k];
                matches.push_back(m);
            }
        }

        if ((int)matches.size() < 10) {
            ROS_WARN("Registration iter %d: only %d matches, skip", iter, (int)matches.size());
            break;
        }
        last_matches = matches;

        Eigen::Quaterniond q_init(T_curr.block<3,3>(0,0));
        double parameters[7];
        parameters[0] = q_init.x();
        parameters[1] = q_init.y();
        parameters[2] = q_init.z();
        parameters[3] = q_init.w();
        parameters[4] = T_curr(0, 3);
        parameters[5] = T_curr(1, 3);
        parameters[6] = T_curr(2, 3);

        ceres::LossFunction* loss = new ceres::HuberLoss(0.5);
        ceres::Problem problem;
        problem.AddParameterBlock(parameters, 7, new PoseSE3Parameterization());

        for (auto& m : matches) {
            ceres::CostFunction* cost = new SDMRegistrationCostFunction(
                m.p_local, m.p_world, m.curvature);
            problem.AddResidualBlock(cost, loss, parameters);
        }

        if (ground.valid) {
            ceres::CostFunction* gnd_cost = new GroundPlaneConstraint(
                ground.n_local, ground.c_local, z_ground_ref,
                ground_w_rp, ground_w_z);
            problem.AddResidualBlock(gnd_cost, nullptr, parameters);
        }

        ceres::Solver::Options opts;
        opts.linear_solver_type = ceres::DENSE_QR;
        opts.max_num_iterations = 10;
        opts.minimizer_progress_to_stdout = false;
        opts.num_threads = 4;
        ceres::Solver::Summary summary;
        ceres::Solve(opts, &problem, &summary);

        Eigen::Map<Eigen::Quaterniond> q_opt(parameters);
        Eigen::Map<Eigen::Vector3d> t_opt(parameters + 4);
        Transf T_new = Eigen::Matrix4d::Identity();
        T_new.block<3,3>(0,0) = q_opt.normalized().toRotationMatrix();
        T_new.block<3,1>(0,3) = t_opt;

        delta_scale = (T_new.block<3,1>(0,3) - T_curr.block<3,1>(0,3)).norm()
                    + 5.0 * (T_new.block<3,3>(0,0) - T_curr.block<3,3>(0,0)).norm();
        T_curr = T_new;

        std::cout << "  iter " << iter << ": matches=" << matches.size()
                  << " delta=" << delta_scale << std::endl;
    }

    if (param.dump_frame > 0 && g_data.step == param.dump_frame && !last_matches.empty()) {
        const std::string base = "/home/albus/slam-math/Bspline/build/";
        const std::string tag  = std::to_string(g_data.step);
        const Eigen::Matrix3d R = T_curr.block<3, 3>(0, 0);
        const Eigen::Vector3d t = T_curr.block<3, 1>(0, 3);

        std::ofstream f_scan(base + tag + "_scan_world.txt", std::ios::out | std::ios::trunc);
        f_scan << std::fixed << std::setprecision(6);
        for (const auto& pt : scan_local.points) {
            const Eigen::Vector3d p_w = R * Eigen::Vector3d(pt.x, pt.y, pt.z) + t;
            f_scan << p_w.x() << " " << p_w.y() << " " << p_w.z() << "\n";
        }
        f_scan.close();

        std::ofstream f_matched(base + tag + "_matched.txt", std::ios::out | std::ios::trunc);
        f_matched << std::fixed << std::setprecision(6);
        for (const auto& m : last_matches)
            f_matched << m.p_world.x() << " " << m.p_world.y() << " " << m.p_world.z() << "\n";
        f_matched.close();

        std::unordered_set<int> matched_set;
        for (const auto& m : last_matches) matched_set.insert(m.scan_idx);

        std::ofstream f_unmatched(base + tag + "_unmatched.txt", std::ios::out | std::ios::trunc);
        f_unmatched << std::fixed << std::setprecision(6);
        for (int i = 0; i < (int)scan_local.size(); i += skip_points) {
            if (scan_local[i].z < match_min_z) continue;
            if (matched_set.count(i)) continue;
            const Eigen::Vector3d p_w = R * Eigen::Vector3d(scan_local[i].x, scan_local[i].y, scan_local[i].z) + t;
            f_unmatched << p_w.x() << " " << p_w.y() << " " << p_w.z() << "\n";
        }
        f_unmatched.close();

        std::cout << "  [Dump] frame=" << g_data.step
                  << " matched=" << last_matches.size()
                  << " unmatched=" << (scan_local.size() / skip_points - matched_set.size())
                  << " -> " << base << tag << "_*.txt" << std::endl;
    }
    return T_curr;
}

void SLAMesher::runMapUpdate(const pcl::PointCloud<pcl::PointXYZ>& scan_local,
                             const Transf& T_world,
                             RangeImageProcessor& range_proc,
                             BSplineMap& bspline_map,
                             double match_dist_thr)
{
    const int UPDATE_INTERVAL  = 20;
    const int MIN_CLUSTER_PTS  = 30;
    const int MAX_NEW_SURFACES = 70;

    if (g_data.step % UPDATE_INTERVAL != 0) return;

    TicToc t_upd;
    range_proc.generateRangeImage(scan_local);
    SegmentationResult seg_upd = range_proc.segmentRangeImage(5, 0.1, MIN_CLUSTER_PTS);

    const Eigen::Matrix3d R_w = T_world.block<3,3>(0,0);
    const Eigen::Vector3d t_w = T_world.block<3,1>(0,3);

    int n_added = 0;
    for (int cid = 0; cid < (int)seg_upd.clusters.size() && n_added < MAX_NEW_SURFACES; cid++) {
        const auto& pixels = seg_upd.clusters[cid];
        if ((int)pixels.size() < MIN_CLUSTER_PTS) continue;

        std::unordered_map<int, std::vector<int>> surf_to_kidx;
        std::vector<Eigen::Vector3d> pix_world(pixels.size(), Eigen::Vector3d::Zero());
        std::vector<bool> pix_valid(pixels.size(), false);

        for (int k = 0; k < (int)pixels.size(); ++k) {
            const auto& px = range_proc.range_image_[pixels[k]];
            if (!px.valid) continue;
            Eigen::Vector3d pw = R_w * Eigen::Vector3d(px.x, px.y, px.z) + t_w;
            pix_world[k] = pw;
            pix_valid[k] = true;
            for (int sid : bspline_map.queryCandidates(pw, 1))
                surf_to_kidx[sid].push_back(k);
        }

        std::vector<bool> confirmed_matched(pixels.size(), false);
        for (auto& [sid, kidxs] : surf_to_kidx) {
            const BSplineMapEntry* entry = bspline_map.getEntry(sid);
            if (!entry || !entry->surface) continue;

            std::vector<Eigen::Vector3d> batch;
            batch.reserve(kidxs.size());
            for (int k : kidxs) batch.push_back(pix_world[k]);

            std::vector<std::pair<BSplineSurface::Parameter, BSplineSurface::Parameter>> fps;
            std::vector<double> dists;
            entry->surface->findFootPrint(batch, fps, dists);

            for (int i = 0; i < (int)kidxs.size(); ++i) {
                double d = std::sqrt(std::abs(dists[i]));
                if (d > 1e-6 && d <= match_dist_thr)
                    confirmed_matched[kidxs[i]] = true;
            }
        }

        auto cloud_local = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        cloud_local->reserve(pixels.size());
        for (int k = 0; k < (int)pixels.size(); ++k) {
            if (!pix_valid[k] || confirmed_matched[k]) continue;
            const auto& px = range_proc.range_image_[pixels[k]];
            cloud_local->push_back(pcl::PointXYZ(px.x, px.y, px.z));
        }
        if ((int)cloud_local->size() < MIN_CLUSTER_PTS) continue;

        const int num_cp = chooseControlGridSize((int)cloud_local->size());

        auto cloud_world = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::transformPointCloud(*cloud_local, *cloud_world, T_world.cast<float>());

        auto init_cp = range_proc.computeInitControlPoints(seg_upd, cid, num_cp, num_cp, 2);
        if (init_cp.empty()) continue;
        for (auto& cp : init_cp) cp = R_w * cp + t_w;

        auto surf = std::make_shared<BSplineSurface>(3, 3, num_cp, num_cp, 0.25);
        surf->setExternalInitControls(init_cp);
        surf->apply(cloud_world, 50, 1, 1, 0.05);
        bspline_map.addSurface(surf, cloud_world);
        n_added++;
    }

    std::cout << "  [MapUpdate] step=" << g_data.step
              << " clusters=" << seg_upd.clusters.size()
              << " +surfaces=" << n_added
              << " total=" << bspline_map.size()
              << " (" << t_upd.toc() << " ms)" << std::endl;
}

void SLAMesher::printMapSummary(const BSplineMap& bspline_map) const
{
    int total_surfaces = bspline_map.size();
    int total_control_points = 0;
    for (int sid = 0; sid < total_surfaces; ++sid) {
        const BSplineMapEntry* e = bspline_map.getEntry(sid);
        if (e && e->surface)
            total_control_points += (int)e->surface->getControls().size();
    }
    std::cout << "Map summary: "
              << total_surfaces << " surfaces, "
              << total_control_points << " control points total" << std::endl;
}

void SLAMesher::saveControlPointsToTxt(const BSplineMap& bspline_map, bool save_surface_samples) const
{
    const std::string out_dir = "/home/albus/slam-math/Bspline/build/controls";
    const std::string sample_dir = "/home/albus/slam-math/Bspline/build/global_map";
    constexpr int kSampleGridU = 8;
    constexpr int kSampleGridV = 8;

    auto findSpan = [](double t, const std::vector<double>& knots, int num_cp) {
        if (t >= 1.0 - 1e-9) return num_cp - 1;
        for (int k = 3; k < num_cp; ++k)
            if (knots[k] <= t && t < knots[k + 1]) return k;
        return 3;
    };

    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);
    if (ec) {
        std::cerr << "Failed to create directory: " << out_dir << " (" << ec.message() << ")\n";
        return;
    }
    if (save_surface_samples) {
        std::filesystem::create_directories(sample_dir, ec);
        if (ec) {
            std::cerr << "Failed to create directory: " << sample_dir << " (" << ec.message() << ")\n";
            return;
        }
    }

    int n_saved = 0;
    int n_samples_saved = 0;
    for (int sid = 0; sid < bspline_map.size(); ++sid) {
        const BSplineMapEntry* e = bspline_map.getEntry(sid);
        if (!e || !e->surface) continue;

        const std::string out_path = out_dir + "/" + std::to_string(sid) + ".txt";
        std::ofstream out(out_path, std::ios::out | std::ios::trunc);
        if (!out) {
            std::cerr << "Failed to open: " << out_path << std::endl;
            continue;
        }

        out << std::fixed << std::setprecision(6);
        for (const auto& cp : e->surface->getControls())
            out << cp.x() << " " << cp.y() << " " << cp.z() << "\n";
        ++n_saved;

        if (!save_surface_samples) continue;

        const auto& surf = e->surface;
        const auto& knU = surf->getKnotsU();
        const auto& knV = surf->getKnotsV();
        const auto& cps = surf->getControls();
        const int num_cpv = surf->getNumCpV();
        const int num_cpu = surf->getNumCpU();

        const std::string sample_path = sample_dir + "/" + std::to_string(sid) + ".txt";
        std::ofstream sout(sample_path, std::ios::out | std::ios::trunc);
        if (!sout) {
            std::cerr << "Failed to open: " << sample_path << std::endl;
            continue;
        }
        sout << std::fixed << std::setprecision(6);
        for (int iu = 0; iu < kSampleGridU; ++iu) {
            const double u = (kSampleGridU > 1) ? double(iu) / double(kSampleGridU - 1) : 0.0;
            const BSplineSurface::Parameter paraU(findSpan(u, knU, num_cpu), u);
            for (int iv = 0; iv < kSampleGridV; ++iv) {
                const double v = (kSampleGridV > 1) ? double(iv) / double(kSampleGridV - 1) : 0.0;
                const BSplineSurface::Parameter paraV(findSpan(v, knV, num_cpv), v);
                const Eigen::Vector3d p = surf->getPos(paraU, paraV, knU, knV, cps, num_cpv);
                sout << p.x() << " " << p.y() << " " << p.z() << "\n";
            }
        }
        ++n_samples_saved;
    }

    std::cout << "B-spline control points saved: " << n_saved
              << " files in " << out_dir << std::endl;
    if (save_surface_samples)
        std::cout << "B-spline surface samples saved: " << n_samples_saved
                  << " files in " << sample_dir << std::endl;
}

void SLAMesher::process(){
    TicToc t_whole;

    // ========== 全局地图 & 工具初始化 ==========
    BSplineMap bspline_map(param.grid);
    RangeImageProcessor range_proc;

    g_data.extendLog();
    Transf T_world = g_data.initFirstTransf();
    g_data.updatePose(T_world);

    // 配准参数
    const int    max_rg_iters   = param.register_times;  // 外层迭代次数
    const double converge_thr   = param.converge_thr;
    const double match_dist_thr = 0.3;  // 点到曲面最大容许距离 (m)
    const int    skip_points    = 20;    // 每隔几个点取一个用于配准 (降低计算量)
    static std::ofstream traj_file;
    if (!traj_file.is_open()) {
        traj_file.open("/tmp/bspline_traj_xyz.txt", std::ios::out | std::ios::trunc);
        traj_file << std::fixed << std::setprecision(6);
    }

    // 地面约束参考高度 (世界系下地面质心 z，由第一帧初始化)
    static double z_ground_ref   = std::numeric_limits<double>::quiet_NaN();
    // 地面点 z 阈值（传感器系）：KITTI lidar 距地面约 1.73m，取 -1.0m 为截止
    const double ground_z_thr    = -1.0;
    while(nh.ok()){
        g_data.step++;
        if (param.max_frames > 0 && g_data.step > param.max_frames) {
            std::cout << "Reached max_frames=" << param.max_frames << ", stop." << std::endl;
            g_data.step--;
            break;
        }
        g_data.extendLog();
        TicToc t_step;

        pcl::PointCloud<pcl::PointXYZ> scan_local;
        if(!range_proc.getPointCloud(scan_local, 0)){
            std::cout << "No more point cloud, exit." << std::endl;
            break;
        }
        std::cout << "===STEP " << g_data.step << "=== points: " << scan_local.size() << std::endl;

        if(g_data.step == 1){
            processFirstFrame(scan_local, T_world, range_proc, bspline_map,
                              z_ground_ref, ground_z_thr);
            continue;
        }

        TicToc t_register;
        Transf T_guess = getOdom();
        GroundConstraint ground = computeGroundConstraint(scan_local, z_ground_ref, ground_z_thr);

        T_world = registerScanToMap(scan_local, T_guess, bspline_map, ground,
                                    z_ground_ref, param.ground_w_rp, param.ground_w_z,
                                    max_rg_iters, converge_thr, match_dist_thr, skip_points,
                                    range_proc.MIN_Z);

        g_data.updatePose(T_world);
        pubTf();

        std::cout << "  Registration: " << t_register.toc() << " ms | Pose: "
                  << T_world(0,3) << " " << T_world(1,3) << " " << T_world(2,3) << std::endl;

        if (traj_file.is_open()) {
            traj_file << T_world(0,3) << " "
                      << T_world(1,3) << " "
                      << T_world(2,3) << "\n";
            traj_file.flush();
        }

        runMapUpdate(scan_local, T_world, range_proc, bspline_map, match_dist_thr);

        path_pub.publish(g_data.path);
        std::cout << "===STEP " << g_data.step << "=== Total: " << t_step.toc() << " ms===" << std::endl;
    }

    std::cout << "Process finished. Total time: " << t_whole.toc() / 1000.0 << " s" << std::endl;

    printMapSummary(bspline_map);
    saveControlPointsToTxt(bspline_map, param.save_surface_samples);
    g_data.savePath2TxtKitti(g_data.file_loc_path_wrt, g_data.path);
    g_data.file_loc_path_wrt.close();
    std::cout << "KITTI trajectory saved." << std::endl;
}
SLAMesher::SLAMesher(ros::NodeHandle & nh_, Parameter & param_, Log & g_data_) : nh (nh_), param(param_), g_data(g_data_){
    odom_pub          = nh.advertise<nav_msgs::Odometry>("/lidar_odometry", 1);

    map_vertices_glb_pub         = nh.advertise<sensor_msgs::PointCloud>("/map_vertices_glb", 1);
    map_vertices_now_pub         = nh.advertise<sensor_msgs::PointCloud>("/map_vertices_now", 1);
    raw_points_in_world_pub = nh.advertise<sensor_msgs::PointCloud>("/raw_points_in_world", 1);
    overlap_point_glb   = nh.advertise<sensor_msgs::PointCloud>("/overlap_point_glb", 1);
    overlap_point_now   = nh.advertise<sensor_msgs::PointCloud>("/overlap_point_now", 1);

    pose_pub      = nh.advertise<sensor_msgs::PointCloud>("/pose_pub", 1);
    pose_imu_pub  = nh.advertise<sensor_msgs::PointCloud>("/pose_imu_pub", 1);
    path_pub      = nh.advertise<nav_msgs::Path>("/path_pub", 1);
    path_odom_pub = nh.advertise<nav_msgs::Path>("/path_odom_pub", 1);
    path_grt_pub  = nh.advertise<nav_msgs::Path>("/path_grt_pub", 1);

    mesh_pub = nh.advertise<mesh_msgs::MeshGeometryStamped>("mesh_msg", 1); //visualize the mesh using mesh tool msg
    mesh_pub_local = nh.advertise<mesh_msgs::MeshGeometryStamped>("mesh_msg_local", 1);

    arrow_pub = nh.advertise<visualization_msgs::Marker>( "arrow", 0 ); //visualize the normal

    param.initParameter(nh);

    if(param.grt_available){
        //if there is ground truth msg, the SLAM trajectory will be aligned to the ground truth trajectory using the
        // first pose, then both trajectories will be visualized for comparison
        ground_truth_sub = nh.subscribe("/pose", 500, &SLAMesher::groundTruthCallback, this);//vicon
        ground_truth_uav_sub = nh.subscribe("/mavros/local_position/odom", 500, &SLAMesher::groundTruthUavCallback, this);//gps+imu
    }
    if(param.odom_available){
        //odometry may be provided by those ways, choose one from odom_sub and imu_sub
        odom_sub = nh.subscribe("/odom", 1, & SLAMesher::odomCallback, this);
        //imu_sub = nh.subscribe("/imu/data", 500, & SLAMesher::imuCallback, this);
        alt_sub = nh.subscribe("/mavros/global_position/rel_alt", 100, & SLAMesher::altitudeCallback, this);
    }

    pointcloud_sub = nh.subscribe("/velodyne_points", 100, &SLAMesher::pointCloudCallback, this);
    g_data.initLog(param.file_loc_report);
}
int main(int argc, char **argv){
    std::cout<<"====START===="<<std::endl;
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);
    std::cout<<std::setprecision(5)<<setiosflags(std::ios::fixed);
    ros::init(argc, argv, "slamesher");

    ros::NodeHandle nh;
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);//Debug Info
    ros::Rate r(1);

    SLAMesher slamesher(nh, param, g_data);
    r.sleep();//waiting for the topic registration
    slamesher.process();

    return 0;
}

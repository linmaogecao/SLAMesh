//This file is for the class SLAMesh, it is the main class of the project, and it is the node of ROS.
#ifndef SLAMESH_SLAMESHER_NODE_H
#define SLAMESH_SLAMESHER_NODE_H

#endif //SLAMESH_SLAMESHER_NODE_H
#include "cell.h"
#include "map.h"
#include "RangeMap.h"
#include "BSplineMap.h"
#include "BSplineSDMErr.h"
#include "GroundGridMap.h"
#include <sensor_msgs/point_cloud_conversion.h>
#include <pcl/registration/icp.h>
#include <unordered_set>
class Parameter{
    //algorithm parameter
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int    max_steps{10000}, max_frames{0}, dump_frame{0}, dump_cluster_step{0}, dump_occluded_step{0}, register_times, num_test, min_points_num_to_gp, num_thread, cross_cell_overlap_length, dataset;
    int    map_update_interval{20}, ground_build_interval{20};
    int    map_save_step_begin{0}, map_save_step_end{0};  // 0=不限；导出 created_step 在此闭区间内的曲面
    int    all_surfaces_max_step{0};  // 0=不限；>0 时 all_surfaces.txt 只含 created_step<=此值的曲面
    double range_max, range_min, range_unit;
    double variance_register, variance_map_update, variance_map_show, variance_min, variance_sensor;
    double grid, voxel_size, converge_thr;
    double map_unmatched_ratio_min{0.30}; // 簇内未匹配比例 >= 此值才新建障碍曲面
    int    cluster_ds_min_pts{100};     // cluster 像素数 <= 此值不抽稀
    int    cluster_ds_target_max{200};  // 大 cluster 目标保留像素上限（0=关闭）
    // 两层 range image 参数
    double range_image_split{30.0};      // 近/远层分界距离（米）；0=禁用远层
    double range_image_far_z_floor_offset{20.0}; // 远层 z_floor = ground_z_min - offset（建图/配准/审计统一）
    int    range_image_far_min_cluster{20}; // 远层 cluster 最小点数（可比近层宽松）
    // 障碍配准采样（range image 模式）
    int    obs_rimg_col_step{3};           // range image 列方向采样间隔（1=全取，3=每3列取1）
    int    obs_match_per_surf_max{50};     // 每个障碍曲面最多保留的匹配点数（0=不限）
    // XY 栅格地面地图
    double ground_z_min{-3.0};        // 传感器系地面点 z 下界（建图/配准共用）
    double ground_z_max{-1.5};        // 传感器系地面点 z 上界（建图/配准共用）
    double ground_cell_size{6.0};     // 每格 XY 边长（米）
    int    ground_cell_min_pts{80};   // 格内点数达此值才拟合曲面
    int    ground_cell_num_cp{5};     // BSpline 每维控制点数
    int    ground_query_radius{1};    // 配准查询格半径（格数）
    int    ground_map_skip_points{8}; // 地面建图：雷达系 z 带内每隔 N 点取 1 点投格
    int    ground_cell_max_pts{400};  // 每格最多保留点数（超出均匀下采样）
    int    ground_fit_max_pts{150};   // 每格 BSpline 拟合最多用点数
    int    ground_skip_points{40};    // 地面配准采样间隔（已被 XY 格子采样替代，仅作初筛备用）
    double ground_clear_dist{150.0};  // 超过此距离（米）的旧格被清除
    double ground_reg_cell_size{3.0}; // 地面配准 XY 格子采样边长（米）；0=退化回 skip 模式
    int    ground_reg_cell_max_pts{30}; // 每个 XY 采样格最多保留点数

    double correction_x{0}, correction_y{0}, correction_z{0},
    correction_roll_degree{0}, correction_pitch_degree{0}, correction_yaw_degree{0};
    // 无 odom 时 step2 冷启动：相对上一帧的 Velodyne 系平移先验 (m)
    double bootstrap_step2_tx{0.5}, bootstrap_step2_ty{0.0}, bootstrap_step2_tz{0.0};
    double test_param;
    double eigen_1, eigen_2, eigen_3;//PCA

    std::string file_loc_report, file_loc_dataset, seq, console_log_path;
    bool three_dir;//features fixed
    bool odom_available, read_offline_pcd, cross_overlap, grt_available, imu_feedback,
            meshing_tsdf, full_cover, save_raw_point_clouds, point2mesh{true},
            residual_combination{true}, save_mesh_map, save_surface_samples{false};
    int visualisation_type;
    int num_margin_old_cell;
    double bias_acc_x, bias_acc_y;

    Parameter(){
    };
    void initParameter(ros::NodeHandle& nh);
};
class Log{
    //global variable, for recording and runtime
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    //record
    std::ofstream file_loc_report_wrt, file_loc_path_wrt, file_loc_path_odom_wrt, file_loc_path_grt_wrt;
    std::string log_file_path_GP_map_points, log_file_path_raw_pcl;
    double t_gp{0}, t_compute_rt{0};
    int log_length{0};
    Eigen::Matrix<double, 1, Eigen::Dynamic> num_cells_now, num_cells_glb, num_cells_new ,
            time_cost, time_get_pcl, time_gp, time_compute_rt, time_update,
            time_cost_draw_map, time_find_overlap, time_find_overlap_points, time_pub_odom,
            time_down_sample, time_devide_point,
            not_a_surface_cell_num, overlap_point_num, gp_times, rg_times;
    Eigen::Matrix<double, 3, Eigen::Dynamic> pose;
    double trajectory_length = 0;

    nav_msgs::Path path;
    nav_msgs::Path path_odom;
    nav_msgs::Path path_grt;
    PointMatrix pose_imu, pose_grt;

    pcl::PointCloud<pcl::PointXYZ> pcl_raw_accumulated;
    //for debug MSE error
    std::queue<PointMatrix> ary_last_raw_points;
    PointMatrix last_raw_points;

    //runtime
    //init
    Transf grt_first_transf;
    bool receive_first_imu = false, if_first_alt = false, imu_init = false, if_first_trans_init = false;
    int step = 0;
    //transformPoints
    std::vector<Transf> T_seq;

    Transf transf_odom_last  = Eigen::MatrixXd::Identity(4, 4),
           transf_odom_now   = Eigen::MatrixXd::Identity(4, 4),//used to calculate incremental transformation between two odometry frames
           transf_slam       = Eigen::MatrixXd::Identity(4, 4);
    std::vector<State> odom_offline;//not used
    std::queue<nav_msgs::OdometryConstPtr> odometry_msg_buf;
    //imu
    std::queue<sensor_msgs::ImuConstPtr> imu_msg_buf;
    Eigen::Vector3d imu_pos, imu_vel, imu_Ba, imu_Bg, acc_0, gyr_0;
    Eigen::Matrix3d imu_rot;
    Eigen::Vector3d g;
    //lidar
    std::deque<sensor_msgs::PointCloud2> pcl_msg_buff_deque;
    sensor_msgs::PointCloud2 pcl_msg_buff;

    Log();
    void initLog(std::string & log_file_path);
    void extendLog();
    Transf initFirstTransf();
    void updatePose(Transf & now_slam_transf);
    inline void recordPoseToPath(enum WhosPath whos_path, const Transf & transf){
        //save transformation to path msg
        //pose
        geometry_msgs::PoseStamped_<std::allocator<void>> path_tmp;
        path_tmp.pose.position.x = transf(0, 3);
        path_tmp.pose.position.y = transf(1, 3);
        path_tmp.pose.position.z = transf(2, 3);

        //orientation
        //transformPoints->tf::matrix->roll yaw pitch->tf::Q, there is no direct way from tf::matrix to tf::Q
        tf::Matrix3x3 tmp_m(transf(0, 0), transf(0, 1), transf(0, 2),
                            transf(1, 0), transf(1, 1), transf(1, 2),
                            transf(2, 0), transf(2, 1), transf(2, 2));
        double roll, yaw, pitch;
        tmp_m.getEulerYPR(yaw, pitch, roll);
        tf::Quaternion tmp_q;
        tmp_q.setRPY(roll, pitch, yaw);
        path_tmp.pose.orientation.x = tmp_q.x();
        path_tmp.pose.orientation.y = tmp_q.y();
        path_tmp.pose.orientation.z = tmp_q.z();
        path_tmp.pose.orientation.w = tmp_q.w();

        //save
        switch (whos_path){
            case Slam:{
                path.poses.push_back(path_tmp); break;
            }
            case Odom:{
                path_odom.poses.push_back(path_tmp); break;
            }
            case Grt:{
                path_grt.poses.push_back(path_tmp); break;
            }
            default: ROS_INFO("wrong usage of recordPoseToPath");
        }
    }
    void savePath2TxtKitti(std::ofstream & file_out, nav_msgs::Path & path_msg);
    void savePathTxt(std::ofstream & file_out, nav_msgs::Path & path_msg);
    void saveResult(double code_whole_time, const PointMatrix & map_glb_point_filtered, Map & map_glb);
    void pose_print(ros::Publisher& cloud_pub) const;
    void accumulateRawPoint(pcl::PointCloud<pcl::PointXYZ> pcl_raw, Transf& transf_this_step);
};
class SLAMesher{
    // node of this project
public:
    Parameter & param;//initialize ros parameters
    Log & g_data;//initialize global variables
    ros::NodeHandle & nh;

    ros::Publisher odom_pub,
    map_vertices_glb_pub, map_vertices_now_pub, raw_points_in_world_pub, overlap_point_glb, overlap_point_now,
    pose_pub, pose_imu_pub, path_pub, path_odom_pub, path_grt_pub,
    mesh_pub , mesh_pub_local, mesh_voxblox_pub, arrow_pub;

    ros::Subscriber pointcloud_sub, laser_sub, odom_sub, imu_sub, imu_raw_sub, alt_sub, ground_truth_sub, ground_truth_uav_sub;

    void groundTruthCallback(const geometry_msgs::PoseStamped::ConstPtr & ground_truth_msg);
    void groundTruthUavCallback(const nav_msgs::Odometry::ConstPtr & odom_msg);
    void altitudeCallback(const std_msgs::Float64 & alt_msg);
    void imuCallback(const sensor_msgs::Imu::ConstPtr & imu_msg);
    void odomCallback(const nav_msgs::Odometry::ConstPtr & odom_msg);
    void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr & pcl_msg);

    Transf getOdom();
    void imuIntegration(const sensor_msgs::ImuConstPtr & imu_msg);
    bool visualize(Map & map_glb, Map & map_now, int option);
    void pubTf();
    SLAMesher(ros::NodeHandle & nh_, Parameter & param_, Log & g_data_);
    void process();

private:
    struct RegMatch {
        Eigen::Vector3d p_local;
        Eigen::Vector3d p_world;
        SurfaceCurvature curvature;
        int scan_idx  = -1;
        bool is_ground = false;
    };

    // 第 1 帧：初始化位姿并建图
    void processFirstFrame(Transf& T_world);

    Transf registerScanToMap(const pcl::PointCloud<pcl::PointXYZ>& scan_local,
                             Transf T_guess,
                             BSplineMap& bspline_map,
                             GroundGridMap& ground_grid,
                             int max_iters,
                             double converge_thr,
                             double match_dist_thr,
                             int skip_points,
                             double match_min_z,
                             double ground_z_min,
                             double ground_z_max,
                             const RangeImageProcessor& rp_near,
                             const RangeImageProcessor& rp_far,
                             bool use_far);

    // 统一建图：近/远两层 range image 分割 → z 分流地面/障碍 → 按各自 interval 决定是否建图
    void runMapBuild(const pcl::PointCloud<pcl::PointXYZ>& scan_local,
                     const Transf& T_world,
                     RangeImageProcessor& range_proc,
                     RangeImageProcessor& range_proc_far,
                     BSplineMap& bspline_map,
                     GroundGridMap& ground_grid,
                     double match_dist_thr,
                     double ground_z_min,
                     double ground_z_max);

    void printMapSummary(const BSplineMap& bspline_map) const;
    void saveGroundGridZ(const GroundGridMap& ground_grid) const;
    void saveGndSurfacesToTxt(const GroundGridMap& ground_grid) const;
    void saveControlPointsToTxt(const BSplineMap& bspline_map,
                                bool save_surface_samples,
                                int step_begin = 0,
                                int step_end = 0) const;

    // 3 阶 B-spline：u/v 同尺寸，范围 [4, 15]
    static int chooseControlGridSize(int num_fitting_points);
};
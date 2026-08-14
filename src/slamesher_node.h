//This file is for the class SLAMesh, it is the main class of the project, and it is the node of ROS.
#ifndef SLAMESH_SLAMESHER_NODE_H
#define SLAMESH_SLAMESHER_NODE_H

#endif //SLAMESH_SLAMESHER_NODE_H
#include "cell.h"
#include "map.h"
#include "RangeMap.h"
#include "BSplineMap.h"
#include "BSplineSDMErr.h"
#include "MultiResGroundMap.h"
#include "PatchworkppGround.h"
#include <sensor_msgs/point_cloud_conversion.h>
#include <pcl/registration/icp.h>
#include <unordered_set>
class Parameter{
    //algorithm parameter
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int    max_steps{10000}, max_frames{0}, dump_frame{0}, dump_cluster_step{0}, dump_occluded_step{0}, register_times, num_test, min_points_num_to_gp, num_thread, cross_cell_overlap_length, dataset;
    int    dump_gnd_scan_begin{0}, dump_gnd_scan_end{0};  // 闭区间：gnd_scan/+whole_gnd/+scan_world/；建地面时再写 all_surfaces/all_surfaces_<step>.txt（累计）；0=关
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
    int    obs_cand_target{1500};          // 逐曲面截顶后的候选目标；不足则补采偏移列（0=关闭）
    // Patchwork++ 地面分割：每帧一份 mask，建图链与配准链共用
    bool   use_patchwork_ground{true};  // false 回退到原 range-image 列传播 + z 带
    PatchworkppConfig patchwork;
    // XY 栅格地面地图
    double ground_z_min{-3.0};        // 传感器系地面点 z 下界（建图/配准共用）
    double ground_z_max{-1.5};        // 传感器系地面点 z 上界（建图/配准共用）
    double ground_cell_size{6.0};     // 每格 XY 边长（米）
    int    ground_cell_min_pts{80};   // 格内点数达此值才拟合曲面
    int    ground_cell_num_cp{5};     // BSpline 每维控制点数
    int    ground_query_radius{1};    // 配准查询格半径（格数）
    int    ground_cell_max_pts{400};  // 每格最多保留点数（超出均匀下采样）
    int    ground_fit_max_pts{150};   // 每格 BSpline 拟合最多用点数
    double ground_cell_z_pct{0.0};    // 建图：格内 z 参考；(0,1]=分位，<=0 用均值
    double ground_cell_z_tol{0.2};    // 建图：相对参考高度上容差（米）；<=0 关闭
    // 建图格内相对高度过滤总开关；false 时忽略 z_pct/z_tol 直接用输入点建图。
    // 该过滤是单边的（只砍 z>z_ref+tol），而配准侧不做同样处理，两侧点集不一致会
    // 使拟合面系统性偏低，每帧把位姿往下压。数值保留不动，便于 A/B 来回切。
    bool   ground_cell_z_filter{false};
    int    ground_skip_points{40};    // 地面配准采样间隔（已被 XY 格子采样替代，仅作初筛备用）
    double ground_reg_cell_size{3.0}; // 地面配准 XY 格子采样边长（米）；0=退化回 skip 模式
    int    ground_reg_cell_max_pts{30}; // 后/中区：每 XY 格均匀初抽种子数
    int    ground_reg_cell_max_pts_front{90}; // 前区（lx>=fb_x0）：更密初抽；<=0 同 max_pts
    int    ground_reg_cell_target_pts{50}; // 过 thr 后每格目标匹配数；不足则用种子邻居填充；<=0 关闭扩容
    int    ground_reg_nbr_per_seed{40}; // 每个种子最多挂的同格邻居数（均匀取自未抽中点）；<=0 不截断
    double ground_reg_y_max{25.0};    // 配准：雷达系 |y| 上限（米）；<=0 不限制
    // 障碍阶段末轮匹配门（米），从 match_dist_thr(0.8) 几何收紧到此值；<=0 关闭恒为 0.8。
    double obs_thr_last{0.0};
    // 障碍 Ceres Huber 阈值（米）。原硬编码 0.5，大于末轮门 0.2，末轮几乎全体二次。
    // 拟合噪声中位 0.13 m，0.5 把植被级外点当内点。<=0 回退 0.5。
    double obs_huber{0.5};
    // 障碍 Ceres 内层迭代。原硬编码 1：外层 4 轮各走 1 步 Gauss-Newton 后重找对应。
    // 地面阶段内层是 10。<=0 回退 1。
    int    obs_ceres_iters{1};
    // 障碍 range-image BFS 的相邻像素法向夹角上限（度）。>0 时折边处拆 cluster。
    // 0=关闭（原行为）。45° 对应 cos=0.707，只切开明显折边，缓曲面仍连在一起。
    double obs_seg_normal_deg{0.0};
    // 建图：拟合残差超过此值（米）的障碍面不入库。<=0 关闭。
    // 用 obs_fit_weight_adj 决定样本内还是无偏化残差。只砍极端尾，不重加权。
    double obs_fit_max_dist{0.0};
    // 建图：无偏化残差超过此值且点数>=60 时沿最长 PCA 轴对半拆开再拟合。
    // 两片子片都成功且残差都低于母片才替换，否则保留母片。<=0 关闭。
    double obs_fit_split_dist{0.0};
    // 障碍匹配中足点被夹在参数域边界([0,1])的对应如何处理。牛顿被夹说明最近点落在该片
    // PCA 数据支撑框之外，此时 |p-足点| 由切向偏移主导，与残差主项（法向分量）不是一个量。
    //   0 = 照旧用三维足点距离过门
    //   1 = 直接丢弃（实测 7 序列水平 0.411→0.525，09/10 崩，已否定）
    //   2 = 改用法向距离过门，并把该匹配的切向残差归零、横向偏移限在 obs_edge_tan_max
    int    obs_edge_mode{0};
    // 模式 2 的横向偏移上限（米）：防止把点粘到远处共面的另一段结构上
    double obs_edge_tan_max{2.0};
    // 建图：新障碍面覆盖旧面占据体素达此比例时退役旧面（<=0 关闭）。不开时同一面墙每次
    // 建图各拟合一张，seq06 单帧参与候选的曲面数 91 -> 570，且这些碎片位姿逐渐漂移、
    // 彼此不一致。
    double obs_map_replace_frac{0.0};
    // 建图：新障碍面占据的体素里把旧障碍面全摘掉（按体素论归属）。与 replace_frac 相比
    // 直接对准"同一体素塞着多张漂移碎片"这个机制。
    bool   obs_map_voxel_owner{false};
    // 障碍面拟合的目标点密度（点/控制点）；>0 时取代原分档表反解控制网格边长。
    // 原分档在低端只有 5 点/控制点，欠定拟合的误差集中在参数域边缘。
    double obs_fit_pts_per_cp{0.0};
    // 障碍面拟合的收敛判据。原值 10 / 0.05：相对下降连续两次低于 5% 即停，而实测停下时
    // 平均点到面距离中位仍有 0.13 m（传感器噪声 0.02 m，末轮匹配门限 0.2 m），
    // 即残差主导项是拟合误差而非位姿误差。
    int    obs_fit_max_iter{10};
    double obs_fit_eps{0.05};
    // 障碍残差按所匹配曲面自身的拟合距离做反方差加权：w ∝ 1/sqrt(sigma^2 + fit^2)，
    // sigma 为传感器测距噪声（米）。权重整体归一到均值 1，以保持 Huber 尺度不变。
    // <=0 关闭。等权时一张 0.4 m 的植被片和一张 0.02 m 的墙面片对位姿的影响相同。
    double obs_fit_weight_sigma{0.0};
    // 加权用样本内残差还是乘过 sqrt(n/(n-p)) 的无偏化值。样本内残差在 48 自由度拟合
    // 80~200 点时会把最过拟合（最不可信）的片评为最优。
    bool   obs_fit_weight_adj{false};
    // 障碍残差数塌陷时把 xy 平面的步长往匀速外推值收缩（方向与 yaw 保留求解结果）。
    // 实测 seq02 按末轮残差数分箱：res<202 的 227 帧（4.9%）步长相对误差 21~28%，
    // res>395 的 2343 帧只有 1.7%；seq00 残差数最小 215、全程无此效应，各分箱平的
    // （1.8~3.1%），故 hi<=210 时 seq00 不受影响。
    // 权重 w=clamp((res-lo)/(hi-lo),0,1)，步长取 w*解 + (1-w)*外推。lo<=0 或 hi<=lo 关闭。
    int    obs_starve_res_lo{0};
    int    obs_starve_res_hi{0};
    // 建图预热期地图近空、残差数天然低（实测 seq02 前 25 帧全部触发），而此时匀速外推还
    // 挂在 bootstrap_step2_tx 上（0.6 m，真值约 1.05 m），回退会把步长钉死在引导值。
    // 此帧数之前不启用回退。
    int    obs_starve_warmup{30};
    // 障碍阶段的运动模型先验（Tikhonov 正则）：把 x/y/yaw 往匀速外推值拉，残差
    // (x-x_pred)/sigma_xy 等，无 Huber。依据：seq02 第4432~4540帧 rcond 掉到 0.15、
    // dom 0.50（法向半数挤在一个 22.5° 方位桶），沿轨方向接近秩亏，解塌向"没动"；
    // 而匀速外推对该段真值步长的预测误差只有 0.007 m。先验信息只填零空间：约束强的
    // 方向上 n*lambda_max≈137 远压过先验，退化方向上 n*lambda_min≈20 与先验同量级。
    // <=0 关闭。sigma_yaw 单位度。
    double obs_prior_sigma_xy{0.0};
    double obs_prior_sigma_yaw{0.0};
    // true：先验只约束沿轨分量（残差 = 增量在预测行进方向上的投影 - 预测步长），横向与
    // yaw 不约束。各向同性版实测 seq02 2.780->0.740 但 seq10 0.679->0.855（偏航是 10 的
    // 最大误差轴，被 1° 偏航先验压住），放松到 0.6/5° 则 seq02 退回 1.397。
    bool   obs_prior_along_only{false};
    // 障碍匹配左右平衡：任一侧（雷达系 y>=0 为左）不得超过该比例，超出则按 scan_idx
    // 等间隔下采样。平行街道里一侧墙垄断时 yaw 被单边点到面残差拽偏，08 航向全程同号
    // 累积 4.5°。<=0 或 >=1 关闭。
    double obs_lr_max_frac{0.0};
    // 障碍残差按水平距离加权：w *= min(1, ref/r)。yaw 的 J ∝ r、JtJ ∝ r²，远距 13 cm
    // 拟合噪声的角误差更大却对 yaw 贡献更大。<=0 关闭。
    double obs_range_ref{0.0};
    // 地面阶段的姿态零阶保持先验强度（度/帧）；<=0 关闭。真值 roll 一步预测 std 0.152 °/帧，
    // 而本系统 roll 帧间抖动全序列超额（窄街段达 1.3°/帧），roll 误差 std ~1.0° 正好等于
    // 超额抖动 0.103 °/帧走 100 帧的随机游走幅度。pitch 无超额，默认关闭。
    double gnd_roll_prior_sigma{0.0};
    double gnd_pitch_prior_sigma{0.0};
    // 障碍(x/y/yaw) 与 地面(roll/pitch/z) 两组自由度的交替求解轮数；1=原单遍。
    int    reg_alternations{1};
    // 匀速外推只作用于 x/y/yaw；roll/pitch/z 沿用上一帧值，不做速率延拓。
    bool   predict_horizontal_only{false};
    // HDL-64 竖直角标定修正（度）。0=关闭。KITTI 公认值 0.205，见 pyLiDAR-SLAM
    // correct_scan。帧内运动去畸变已实测无效（两个符号都试过，均值 1.408→1.633，
    // 且高速的 01 恶化最多，与机制预期相反），KISS-ICP 在 KITTI 上同样 deskew=False。
    double scan_correct_deg{0.0};
    // 只修正 Patchwork++ 标出的地面点。全量修正会挪动 range image 的行分箱，
    // 在特征稀少处打散障碍聚类（seq01 失锁段障碍匹配 256→178，整条 2.1→17.7）。
    bool   scan_correct_ground_only{true};
    // 地面阶段末轮的匹配硬门（米）。门越紧，存活点越是"已经和当前估计吻合"的那批，
    // 估计量趋于自身预测的不动点。原硬编码 0.03。
    double ground_thr_last{0.03};
    // 配准：雷达系 |x| 上限（米）；<=0 不限制。pitch 绕传感器原点转，残差对 pitch 的
    // 灵敏度正比于 x，远前方点单独主导 pitch；而那里扫描最稀、地图格最不成熟。
    double ground_reg_x_max{0.0};
    // 只作用于前方（lx>0）的纵向上限（米）；<=0 不限制。实测远前方桶(x>35m)的竖直
    // 残差与其余各桶反号且大 6 倍，是 pitch 偏置的唯一来源；远后方桶残差正常且要
    // 留着撑力臂，所以门必须是单边的。
    double ground_reg_x_max_front{0.0};
    // 建图：只收雷达系水平距离 <= 此值的地面点（米）；<=0 不限制。远距离观测的高度
    // 误差正比于距离，且格内旧的远观测被均匀抽样永久保留，会把拟合面钉在偏差上。
    double ground_map_r_max{0.0};
    int    ground_reg_total_max{4000}; // 每帧地面种子总预算；超出按占用格数等比缩每格配额；<=0 不限
    // 每帧地面「最终匹配」总预算。total_max 只约束种子，扩容会把总量涨回去，
    // 这里是唯一能真正封顶的地方。裁剪在前/中/后三桶内各自等间隔进行，
    // 桶比例与力臂分布不变。<=0 不限；需 ground_reg_fb_front>0 才生效。
    int    ground_reg_match_max{2500};
    // 前后配额：thr 后按雷达系 x 分前/中/后；不足用成功种子邻居补，超额再裁；<=0 关闭
    double ground_reg_fb_x0{5.0};    // |x|<x0 为中桶；x>=x0 前，x<=-x0 后
    double ground_reg_fb_front{0.34}; // 前桶目标比例；<=0 关闭前后配额
    double ground_reg_fb_mid{0.33};  // 中桶目标比例
    double ground_reg_fb_back{0.33}; // 后桶目标比例
    // 粗层（多分辨率第二层）
    double ground_coarse_cell_size{20.0};  // 粗格 XY 边长（米）
    int    ground_coarse_min_pts{30};      // 粗层拟合门槛
    int    ground_coarse_num_cp{7};        // 粗层 BSpline 每维控制点数（7x7）
    int    ground_coarse_query_radius{1};  // 粗层查询格半径
    int    ground_coarse_cell_max_pts{600}; // 粗格最多保留点数
    int    ground_coarse_fit_max_pts{250}; // 粗层拟合最多用点数
    // 地面 range image 列向提取
    double gnd_ri_z_min{-3.0};      // 地面 RI / 建图 z 下界（传感器系）
    double gnd_ri_z_max{0.0};       // 远区 / 近区外 z 上界
    // 近区（雷达系 |x|<=near_x 且 |y|<=near_y）：更严 z 上界，挡车身抬高假地面
    double gnd_near_x_max{20.0};    // 近区前后半长（米）；<=0 关闭近区分层
    double gnd_near_y_max{10.0};    // 近区左右半宽（米）
    double gnd_near_z_max{-1.5};    // 近区 z 上界（比 gnd_ri_z_max 更严）
    double gnd_col_max_step{0.3};   // 列向传播最大 Δz（米）；超过则断链不断列
    double gnd_col_seed_h_up{0.4};  // 相对本列 seed_z 最大上抬（米）；挡住车身抬高假地面
    double gnd_normal_z_min{0.5};   // 拟合曲面法向 |z| 最小值；更小则视为墙面丢弃

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
        // 足点被夹在参数域边界：q 是片边缘点而非真最近点，切向残差编码的是"沿面滑动"
        // 的虚假约束，只保留法向分量。
        bool drop_tangent = false;
        // 所匹配曲面自身的平均拟合距离（米），<0 表示未拟合过
        double fit_dist = -1.0;
        // 残差整体缩放系数（反方差加权用），1.0 为原等权行为
        double weight = 1.0;
    };

    // 第 1 帧：初始化位姿并建图
    void processFirstFrame(Transf& T_world);

    // pw_ground_mask 非空时取代 z 带判据作为地面点来源；为空则沿用原逻辑
    Transf registerScanToMap(const pcl::PointCloud<pcl::PointXYZ>& scan_local,
                             const std::vector<uint8_t>& pw_ground_mask,
                             Transf T_guess,
                             BSplineMap& bspline_map,
                             MultiResGroundMap& mr_ground,
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
                     const std::vector<uint8_t>& pw_ground_mask,
                     const Transf& T_world,
                     RangeImageProcessor& range_proc,
                     RangeImageProcessor& range_proc_far,
                     RangeImageProcessor& range_proc_gnd,
                     BSplineMap& bspline_map,
                     MultiResGroundMap& mr_ground,
                     double match_dist_thr,
                     double ground_z_min,
                     double ground_z_max);

    PatchworkppGround patchwork_;

    void printMapSummary(const BSplineMap& bspline_map) const;
    void saveGroundGridZ(const MultiResGroundMap& mr_ground) const;
    // out_path 空：写 build/gnd_surfaces.txt；否则写指定路径（地面采样点 xyz）
    void saveGndSurfacesToTxt(const MultiResGroundMap& mr_ground,
                              const std::string& out_path = "") const;
    // dump_gnd_scan 区间内：每次建完地面，累计快照到 all_surfaces/all_surfaces_<step>.txt
    void dumpGndAllSurfacesAtBuild(const MultiResGroundMap& mr_ground) const;
    void saveControlPointsToTxt(const BSplineMap& bspline_map,
                                bool save_surface_samples,
                                int step_begin = 0,
                                int step_end = 0) const;

    // 3 阶 B-spline：u/v 同尺寸，范围 [4, 15]
    static int chooseControlGridSize(int num_fitting_points, double pts_per_cp = 0.0);
};
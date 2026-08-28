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
#include "ConsoleLogTee.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <iterator>
#include <omp.h>
Parameter param;//parameters
Log g_data;//global variables
static ConsoleLogTee g_console_log_tee;

namespace {
// dump 产物按工作区分目录。多个 catkin workspace 的二进制共用一个 dump 目录时，
// all_surfaces / frame1_apply_profile / 轨迹 txt 会被后跑的那轮覆盖，A/B 无法归因。
// 工作区名取自 /proc/self/exe 中 devel 的上一级，因此归属的是产出该二进制的工作区。
std::string bsplineDumpDir()
{
    constexpr const char* kBase = "/home/albus/slam-math/Bspline/build";
    std::string dir = std::string(kBase) + "/unknown_ws";

    std::error_code ec;
    const std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (!ec) {
        for (std::filesystem::path p = exe; p != p.parent_path(); p = p.parent_path()) {
            if (p.filename() == "devel" && p.has_parent_path()) {
                dir = std::string(kBase) + "/" + p.parent_path().filename().string();
                break;
            }
        }
    }
    std::filesystem::create_directories(dir, ec);
    return dir;
}
const std::string kBsplineBuildDir = bsplineDumpDir();
constexpr double kMatchDropRatioThr = 0.70;
constexpr int kMatchDropAbsThr = 150;

inline double farLayerZFloor(double ground_z_min)
{
    return ground_z_min - param.range_image_far_z_floor_offset;
}

inline void matrixToRPY(const Eigen::Matrix3d& R, double& roll, double& pitch, double& yaw)
{
    Eigen::Quaterniond q(R);
    tf::Matrix3x3(tf::Quaternion(q.x(), q.y(), q.z(), q.w())).getRPY(roll, pitch, yaw);
}

inline Eigen::Matrix3d rpyToMatrix(double roll, double pitch, double yaw)
{
    tf::Matrix3x3 m;
    m.setRPY(roll, pitch, yaw);
    Eigen::Matrix3d R;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R(i, j) = m[i][j];
    return R;
}

inline Transf makePoseFromXYZRPY(double x, double y, double z,
                                 double roll, double pitch, double yaw)
{
    Transf T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = rpyToMatrix(roll, pitch, yaw);
    T(0, 3) = x;
    T(1, 3) = y;
    T(2, 3) = z;
    return T;
}

// KITTI 的 HDL-64 有约 0.205° 的竖直角标定偏差：同一块地面在不同距离上被测出
// 不同高度，误差正比于水平距离（40 m 处 0.14 m）。地图里存的是这块地面更远时的
// 观测，当前帧在更近处看它，两者之差正比于这期间的距离变化，也就是正比于行驶
// 距离，最后被位姿吸收成同号的 pitch 偏置。
// 修正为绕 a = normalize(p × ẑ) 转 theta；a ⊥ p，Rodrigues 只剩两项。
// 取自 pyLiDAR-SLAM correct_scan，KISS-ICP / SAGE-ICP 在 KITTI 上同样先做这一步。
// mask 非空时只修正标记点：该缺陷表现为与距离成正比的高度误差，只对地面链的
// 竖直点到面残差有意义；障碍链匹配完整三维面，地图与扫描同转一个小角近乎无
// 操作，却会把 range image 的行分箱整体挪 0.155°（HDL-64 行距约 0.4°），在
// 特征稀少处打散聚类。seq01 实测全量修正时失锁段障碍匹配 256→178。
void correctScanCalib(pcl::PointCloud<pcl::PointXYZ>& scan, double theta,
                      const std::vector<uint8_t>* mask)
{
    const double c = std::cos(theta), s = std::sin(theta);
    const int n = static_cast<int>(scan.size());
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        if (mask && (i >= static_cast<int>(mask->size()) || (*mask)[i] == 0)) continue;
        auto& p = scan.points[static_cast<size_t>(i)];
        const double x = p.x, y = p.y, z = p.z;
        const double rho = std::hypot(x, y);
        if (rho < 1e-6) continue;
        p.x = static_cast<float>(x * c - (x * z / rho) * s);
        p.y = static_cast<float>(y * c - (y * z / rho) * s);
        p.z = static_cast<float>(z * c + rho * s);
    }
}

// 姿态零阶保持先验：把 roll 或 pitch 拉回上一帧的值，残差 = (角 - 上帧角)/sigma。
// 只有 roll 需要：点到面残差对绕轴旋转的 Hessian 是 H_roll=Σy²、H_pitch=Σx²，KITTI 实测
// rmsY 9.6 m vs rmsX 25.6 m，roll 弱 7 倍；窄街区间 rmsY 掉到 4 m，弱 8 倍以上，roll 帧间
// 摆幅达 1.3°。11 条序列的 roll 帧间抖动全部超出真值（超额 0.007~0.117 °/帧），pitch 全部
// 不超额，与 Hessian 差距一致。真值 roll 零阶保持一步预测 std 只有 0.152 °/帧；线性外推对
// 真值更准（0.128）但把量测噪声放大 2.24 倍，故取零阶保持。
struct AttitudeHoldPrior {
    AttitudeHoldPrior(int idx, double target, double w)
        : idx_(idx), t_(target), w_(w) {}

    template <typename T>
    bool operator()(const T* const rpy_z, T* residual) const
    {
        residual[0] = T(w_) * (rpy_z[idx_] - T(t_));
        return true;
    }

    static ceres::CostFunction* Create(int idx, double target, double w)
    {
        return new ceres::AutoDiffCostFunction<AttitudeHoldPrior, 1, 3>(
            new AttitudeHoldPrior(idx, target, w));
    }

    int idx_;
    double t_, w_;
};

// 地面：点到平面 r = scale * n·(R p + t - q)；只优化 [roll, pitch, z]
struct GroundPointToPlaneZRP {
    GroundPointToPlaneZRP(const Eigen::Vector3d& p_local,
                          const Eigen::Vector3d& surf_pt,
                          const Eigen::Vector3d& normal,
                          double x_fixed, double y_fixed, double yaw_fixed,
                          double residual_scale)
        : p_local_(p_local), q_(surf_pt), n_(normal.normalized()),
          x_(x_fixed), y_(y_fixed), yaw_(yaw_fixed), scale_(residual_scale) {}

    template <typename T>
    bool operator()(const T* const rpy_z, T* residual) const
    {
        const T roll = rpy_z[0];
        const T pitch = rpy_z[1];
        const T z = rpy_z[2];
        const T yaw = T(yaw_);
        const T cr = ceres::cos(roll), sr = ceres::sin(roll);
        const T cp = ceres::cos(pitch), sp = ceres::sin(pitch);
        const T cy = ceres::cos(yaw), sy = ceres::sin(yaw);

        Eigen::Matrix<T, 3, 3> R;
        R(0, 0) = cy * cp;
        R(0, 1) = cy * sp * sr - sy * cr;
        R(0, 2) = cy * sp * cr + sy * sr;
        R(1, 0) = sy * cp;
        R(1, 1) = sy * sp * sr + cy * cr;
        R(1, 2) = sy * sp * cr - cy * sr;
        R(2, 0) = -sp;
        R(2, 1) = cp * sr;
        R(2, 2) = cp * cr;

        Eigen::Matrix<T, 3, 1> p(T(p_local_.x()), T(p_local_.y()), T(p_local_.z()));
        Eigen::Matrix<T, 3, 1> t(T(x_), T(y_), z);
        Eigen::Matrix<T, 3, 1> pw = R * p + t;
        Eigen::Matrix<T, 3, 1> q(T(q_.x()), T(q_.y()), T(q_.z()));
        Eigen::Matrix<T, 3, 1> n(T(n_.x()), T(n_.y()), T(n_.z()));
        residual[0] = T(scale_) * n.dot(pw - q);
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& p_local,
                                       const Eigen::Vector3d& surf_pt,
                                       const Eigen::Vector3d& normal,
                                       double x_fixed, double y_fixed, double yaw_fixed,
                                       double residual_scale = 1.0)
    {
        return new ceres::AutoDiffCostFunction<GroundPointToPlaneZRP, 1, 3>(
            new GroundPointToPlaneZRP(
                p_local, surf_pt, normal, x_fixed, y_fixed, yaw_fixed, residual_scale));
    }

    Eigen::Vector3d p_local_, q_, n_;
    double x_, y_, yaw_, scale_;
};

// 障碍：SDM 残差；只优化 [x, y, yaw]，z/roll/pitch 固定
// 只约束沿轨分量的运动模型先验：残差 = (增量在预测行进方向上的投影 - 预测步长)/sigma。
// 匀速外推的精度是各向异性的：对步长的预测误差 0.007 m（seq02 失速段实测），而横向随
// 转向变化大得多。秩亏也只出现在沿轨方向。故横向与 yaw 完全不约束。
struct MotionPriorAlongTrack {
    MotionPriorAlongTrack(double x_prev, double y_prev,
                          double ux, double uy, double len_pred, double w)
        : x0_(x_prev), y0_(y_prev), ux_(ux), uy_(uy), len_(len_pred), w_(w) {}

    template <typename T>
    bool operator()(const T* const xy_yaw, T* residual) const
    {
        const T along = (xy_yaw[0] - T(x0_)) * T(ux_) + (xy_yaw[1] - T(y0_)) * T(uy_);
        residual[0] = T(w_) * (along - T(len_));
        return true;
    }

    static ceres::CostFunction* Create(double x_prev, double y_prev,
                                       double ux, double uy, double len_pred, double w)
    {
        return new ceres::AutoDiffCostFunction<MotionPriorAlongTrack, 1, 3>(
            new MotionPriorAlongTrack(x_prev, y_prev, ux, uy, len_pred, w));
    }

    double x0_, y0_, ux_, uy_, len_, w_;
};

// 只约束 yaw：残差 = wrap(yaw - yaw_pred)/sigma。along_only 把横向放开之后，
// 平行街道里 yaw 仍可能接近零空间；各向同性版伤 seq10 是因为连横向一起锁死。
struct MotionPriorYaw {
    MotionPriorYaw(double yaw_pred, double w) : yaw_(yaw_pred), w_(w) {}

    template <typename T>
    bool operator()(const T* const xy_yaw, T* residual) const
    {
        T dyaw = xy_yaw[2] - T(yaw_);
        while (dyaw > T(M_PI))  dyaw -= T(2.0 * M_PI);
        while (dyaw < T(-M_PI)) dyaw += T(2.0 * M_PI);
        residual[0] = T(w_) * dyaw;
        return true;
    }

    static ceres::CostFunction* Create(double yaw_pred, double w)
    {
        return new ceres::AutoDiffCostFunction<MotionPriorYaw, 1, 3>(
            new MotionPriorYaw(yaw_pred, w));
    }

    double yaw_, w_;
};

// 障碍阶段的运动模型先验：把 [x, y, yaw] 往匀速外推值拉。信息量只在数据的零空间里
// 起作用，故不设触发判据，也不加 Huber（先验不是观测，没有外点一说）。
struct MotionPriorXYYaw {
    MotionPriorXYYaw(double x_pred, double y_pred, double yaw_pred,
                     double w_xy, double w_yaw)
        : x_(x_pred), y_(y_pred), yaw_(yaw_pred), w_xy_(w_xy), w_yaw_(w_yaw) {}

    template <typename T>
    bool operator()(const T* const xy_yaw, T* residual) const
    {
        residual[0] = T(w_xy_) * (xy_yaw[0] - T(x_));
        residual[1] = T(w_xy_) * (xy_yaw[1] - T(y_));
        T dyaw = xy_yaw[2] - T(yaw_);
        while (dyaw > T(M_PI))  dyaw -= T(2.0 * M_PI);
        while (dyaw < T(-M_PI)) dyaw += T(2.0 * M_PI);
        residual[2] = T(w_yaw_) * dyaw;
        return true;
    }

    static ceres::CostFunction* Create(double x_pred, double y_pred, double yaw_pred,
                                       double w_xy, double w_yaw)
    {
        return new ceres::AutoDiffCostFunction<MotionPriorXYYaw, 3, 3>(
            new MotionPriorXYYaw(x_pred, y_pred, yaw_pred, w_xy, w_yaw));
    }

    double x_, y_, yaw_, w_xy_, w_yaw_;
};

struct ObstaclePointToSurfaceXYYaw {
    ObstaclePointToSurfaceXYYaw(const Eigen::Vector3d& p_local,
                                const Eigen::Vector3d& p_world,
                                const SurfaceCurvature& frame,
                                double roll_fixed,
                                double pitch_fixed,
                                double z_fixed,
                                bool drop_tangent = false,
                                double weight = 1.0)
        : weight_(weight), p_local_(p_local), q_(frame.point),
          tangent1_(frame.tangent1), tangent2_(frame.tangent2),
          normal_(frame.normal.normalized()),
          roll_(roll_fixed), pitch_(pitch_fixed), z_(z_fixed)
    {
        if (drop_tangent) return;  // 系数留 0：足点在片边缘时切向约束是假的
        const double dist = (p_world - frame.point).dot(normal_);
        const auto coefficient = [dist](double curvature) {
            constexpr double kEps = 1e-4;
            if (std::abs(curvature) < kEps) return 0.0;
            const double radius = 1.0 / curvature;
            const double denominator = dist - radius;
            if (std::abs(denominator) < kEps) return 0.0;
            const double value = dist / denominator;
            if (value > 100.0) return 10.0;
            return value > 0.0 ? std::sqrt(value) : 0.0;
        };
        coeff_t1_ = coefficient(frame.k1);
        coeff_t2_ = coefficient(frame.k2);
    }

    template <typename T>
    bool operator()(const T* const xy_yaw, T* residual) const
    {
        const T x = xy_yaw[0];
        const T y = xy_yaw[1];
        const T yaw = xy_yaw[2];
        const T roll = T(roll_);
        const T pitch = T(pitch_);
        const T cr = ceres::cos(roll), sr = ceres::sin(roll);
        const T cp = ceres::cos(pitch), sp = ceres::sin(pitch);
        const T cy = ceres::cos(yaw), sy = ceres::sin(yaw);

        Eigen::Matrix<T, 3, 3> R;
        R(0, 0) = cy * cp;
        R(0, 1) = cy * sp * sr - sy * cr;
        R(0, 2) = cy * sp * cr + sy * sr;
        R(1, 0) = sy * cp;
        R(1, 1) = sy * sp * sr + cy * cr;
        R(1, 2) = sy * sp * cr - cy * sr;
        R(2, 0) = -sp;
        R(2, 1) = cp * sr;
        R(2, 2) = cp * cr;

        Eigen::Matrix<T, 3, 1> p(T(p_local_.x()), T(p_local_.y()), T(p_local_.z()));
        Eigen::Matrix<T, 3, 1> t(x, y, T(z_));
        const Eigen::Matrix<T, 3, 1> diff = R * p + t - q_.template cast<T>();
        const T w = T(weight_);
        residual[0] = w * T(coeff_t1_) * diff.dot(tangent1_.template cast<T>());
        residual[1] = w * T(coeff_t2_) * diff.dot(tangent2_.template cast<T>());
        residual[2] = w * diff.dot(normal_.template cast<T>());
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& p_local,
                                       const Eigen::Vector3d& p_world,
                                       const SurfaceCurvature& frame,
                                       double roll_fixed,
                                       double pitch_fixed,
                                       double z_fixed,
                                       bool drop_tangent = false,
                                       double weight = 1.0)
    {
        return new ceres::AutoDiffCostFunction<ObstaclePointToSurfaceXYYaw, 3, 3>(
            new ObstaclePointToSurfaceXYYaw(
                p_local, p_world, frame, roll_fixed, pitch_fixed, z_fixed,
                drop_tangent, weight));
    }

    double weight_{1.0};
    Eigen::Vector3d p_local_, q_, tangent1_, tangent2_, normal_;
    double roll_, pitch_, z_, coeff_t1_{0.0}, coeff_t2_{0.0};
};

std::string seqIdFromParam(const std::string& seq)
{
    if (seq.size() >= 3 && seq[0] == '/') return seq.substr(1, 2);
    if (seq.size() >= 2) return seq.substr(seq.size() - 2);
    return seq.empty() ? "00" : seq;
}

std::string makeSeqSpecificLogPath(const std::string& raw_path, const std::string& seq)
{
    const std::filesystem::path p(raw_path);
    const std::string seq_id = seqIdFromParam(seq);
    const std::string stem = p.stem().string().empty() ? "debug" : p.stem().string();
    const std::string ext = p.extension().string().empty() ? ".txt" : p.extension().string();
    return (p.parent_path() / (stem + "-" + seq_id + ext)).string();
}

std::string matchDropLogPath(const std::string& report_dir, const std::string& seq)
{
    std::string dir = report_dir;
    while (!dir.empty() && dir.back() == '/') dir.pop_back();
    if (dir.empty()) dir = ".";
    return dir + "/seq" + seqIdFromParam(seq) + "_match_drop.txt";
}

void appendMatchDropEvent(int step,
                          int iter,
                          int prev_matches,
                          int curr_matches,
                          int n_gnd,
                          int n_obs)
{
    const int drop_abs = prev_matches - curr_matches;
    if (prev_matches <= 0 || drop_abs < kMatchDropAbsThr) return;
    const double ratio = static_cast<double>(curr_matches) / static_cast<double>(prev_matches);
    if (ratio > kMatchDropRatioThr) return;

    static std::string cached_path;
    static std::ofstream fout;
    const std::string path = matchDropLogPath(param.file_loc_report, param.seq);
    if (cached_path != path) {
        if (fout.is_open()) fout.close();
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
        fout.open(path, std::ios::out | std::ios::trunc);
        cached_path = path;
        if (fout.is_open()) {
            fout << "# step iter prev_matches curr_matches drop_abs curr_over_prev gnd obs\n";
            fout << std::fixed << std::setprecision(4);
        }
    }
    if (!fout.is_open()) return;

    fout << step << " "
         << iter << " "
         << prev_matches << " "
         << curr_matches << " "
         << drop_abs << " "
         << ratio << " "
         << n_gnd << " "
         << n_obs << "\n";
    fout.flush();

    std::cout << "  [MatchDrop] step=" << step
              << " iter=" << iter
              << " prev=" << prev_matches
              << " curr=" << curr_matches
              << " ratio=" << ratio
              << " gnd=" << n_gnd
              << " obs=" << n_obs << std::endl;
}

void saveFrame1GroundPoints(const std::vector<Eigen::Vector3d>& pts_lidar)
{
    if (pts_lidar.empty()) return;
    std::vector<Eigen::Vector3d> sorted = pts_lidar;
    std::sort(sorted.begin(), sorted.end(),
              [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) { return a.z() < b.z(); });

    const std::string path = std::string(kBsplineBuildDir) + "/frame1_ground_points.txt";
    std::ofstream fout(path);
    if (!fout.is_open()) return;
    fout << std::fixed << std::setprecision(5);
    for (const auto& p : sorted)
        fout << p.x() << " " << p.y() << " " << p.z() << "\n";
}

void saveOccludedPointsTxt(const std::vector<Eigen::Vector3d>& pts_lidar,
                           const Transf& T_world,
                           const std::string& base_dir)
{
    if (pts_lidar.empty()) return;
    const Eigen::Matrix3d R = T_world.block<3, 3>(0, 0);
    const Eigen::Vector3d t = T_world.block<3, 1>(0, 3);

    std::ofstream f_lidar(base_dir + "/occluded_lidar.txt", std::ios::out | std::ios::trunc);
    std::ofstream f_world(base_dir + "/occluded_world.txt", std::ios::out | std::ios::trunc);
    if (!f_lidar.is_open() || !f_world.is_open()) return;

    f_lidar << std::fixed << std::setprecision(6);
    f_world << std::fixed << std::setprecision(6);
    for (const auto& p : pts_lidar) {
        f_lidar << p.x() << " " << p.y() << " " << p.z() << "\n";
        const Eigen::Vector3d pw = R * p + t;
        f_world << pw.x() << " " << pw.y() << " " << pw.z() << "\n";
    }
}

void saveClusterFilterStatsTxt(const pcl::PointCloud<pcl::PointXYZ>& scan_local,
                               int step,
                               double ground_z_min,
                               double ground_z_max,
                               double split_dist,
                               const RangeImageProcessor& range_proc,
                               const RangeImageProcessor& range_proc_far)
{
    const bool use_far_layer = (split_dist > 0.0);
    const double near_min = std::max((double)range_proc.MIN_RANGE, 0.0);
    const double near_max = std::min((double)range_proc.MAX_RANGE, use_far_layer ? split_dist : 1e9);
    const double far_min  = std::max((double)range_proc_far.MIN_RANGE, split_dist);
    const double far_max  = std::min((double)range_proc_far.MAX_RANGE, 1e9);

    struct FilterStats {
        int total = 0;
        int nonfinite = 0;
        int ground_band = 0;
        int below_min_z = 0;
        int range_too_near = 0;
        int range_too_far = 0;
        int valid = 0;
    };

    FilterStats near_stats, far_stats;
    int scan_world_kept = 0;

    for (const auto& pt : scan_local.points) {
        ++near_stats.total;
        ++far_stats.total;

        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
            ++near_stats.nonfinite;
            ++far_stats.nonfinite;
            continue;
        }

        const bool in_ground_band = (pt.z >= ground_z_min && pt.z <= ground_z_max);
        if (!in_ground_band) ++scan_world_kept;
        if (in_ground_band) {
            ++near_stats.ground_band;
            ++far_stats.ground_band;
            continue;
        }

        const double range = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
        if (pt.z < range_proc.MIN_Z) {
            ++near_stats.below_min_z;
            ++far_stats.below_min_z;
            continue;
        }

        if (range < near_min) ++near_stats.range_too_near;
        else if (range > near_max) ++near_stats.range_too_far;
        else ++near_stats.valid;

        if (use_far_layer) {
            if (range < far_min) ++far_stats.range_too_near;
            else if (range > far_max) ++far_stats.range_too_far;
            else ++far_stats.valid;
        }
    }

    const std::string path = std::string(kBsplineBuildDir) + "/cluster_filter_stats_step" + std::to_string(step) + ".txt";
    std::ofstream fout(path, std::ios::out | std::ios::trunc);
    if (!fout.is_open()) return;

    fout << std::fixed << std::setprecision(6);
    fout << "step: " << step << "\n";
    fout << "ground_band_z: [" << ground_z_min << ", " << ground_z_max << "]\n";
    fout << "min_z: " << range_proc.MIN_Z << "\n";
    fout << "split_dist: " << split_dist << "\n";
    fout << "scan_world_kept_non_ground: " << scan_world_kept << "\n";
    fout << "\n";

    auto dump_one = [&](const char* name, const FilterStats& s, double rmin, double rmax) {
        fout << "[" << name << "]\n";
        fout << "range_window: [" << rmin << ", " << rmax << "]\n";
        fout << "total_raw: " << s.total << "\n";
        fout << "nonfinite: " << s.nonfinite << "\n";
        fout << "ground_band_excluded: " << s.ground_band << "\n";
        fout << "below_min_z_excluded: " << s.below_min_z << "\n";
        fout << "range_too_near: " << s.range_too_near << "\n";
        fout << "range_too_far: " << s.range_too_far << "\n";
        fout << "valid_for_range_image: " << s.valid << "\n";
        fout << "\n";
    };

    dump_one("near", near_stats, near_min, near_max);
    if (use_far_layer)
        dump_one("far", far_stats, far_min, far_max);
}

struct FarClusterVerdict {
    std::string rimg_status;
    std::string cluster_pre_ds;
    std::string cluster_post_ds;
    int pixel_u = -1;
    int pixel_v = -1;
    int pixel_idx = -1;
    int cluster_id = -1;
    int winner_idx = -1;
};

FarClusterVerdict classifyFarPointPipeline(
    const pcl::PointXYZ& pt,
    int cloud_idx,
    const RangeImageProcessor& rp,
    double split_dist,
    double z_floor,
    const SegmentationResult& seg_pre_ds,
    const SegmentationResult& seg_post_ds,
    const std::unordered_set<int>& pixels_in_cluster_pre,
    const std::unordered_set<int>& pixels_in_cluster_post,
    const std::unordered_map<int, int>& pixel_to_cid_pre,
    const std::unordered_map<int, int>& pixel_to_cid_post,
    const std::unordered_set<int>& kept_labels_pre)
{
    FarClusterVerdict v;
    const double eff_range_min = std::max(rp.MIN_RANGE, split_dist);
    const double eff_range_max = rp.MAX_RANGE;

    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
        v.rimg_status = "excluded_nonfinite";
        v.cluster_pre_ds = "n/a";
        v.cluster_post_ds = "n/a";
        return v;
    }
    if (pt.z < z_floor) {
        v.rimg_status = "excluded_below_z_floor";
        v.cluster_pre_ds = "n/a";
        v.cluster_post_ds = "n/a";
        return v;
    }

    const double range = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
    if (range < eff_range_min) {
        v.rimg_status = "excluded_range_too_near";
        v.cluster_pre_ds = "n/a";
        v.cluster_post_ds = "n/a";
        return v;
    }
    if (range > eff_range_max) {
        v.rimg_status = "excluded_range_too_far";
        v.cluster_pre_ds = "n/a";
        v.cluster_post_ds = "n/a";
        return v;
    }

    const float fov_up_rad = rp.FOV_UP * static_cast<float>(M_PI) / 180.0f;
    const float fov_down_rad = rp.FOV_DOWN * static_cast<float>(M_PI) / 180.0f;
    const float fov_total_rad = std::abs(fov_up_rad - fov_down_rad);
    const double angle_vert = std::asin(pt.z / range);
    const double row_ratio = (angle_vert - fov_down_rad) / fov_total_rad;
    const int row = static_cast<int>(std::round(row_ratio * (rp.H_SCANS - 1)));
    const double angle_horiz = std::atan2(pt.y, pt.x);
    int col = static_cast<int>(std::round((angle_horiz + M_PI) / (2.0 * M_PI) * rp.W_COLS));
    if (col >= rp.W_COLS) col -= rp.W_COLS;
    if (col < 0) col += rp.W_COLS;

    if (row < 0 || row >= rp.H_SCANS || col < 0 || col >= rp.W_COLS) {
        v.rimg_status = "excluded_fov_out";
        v.cluster_pre_ds = "n/a";
        v.cluster_post_ds = "n/a";
        return v;
    }

    v.pixel_u = row;
    v.pixel_v = col;
    v.pixel_idx = row * rp.W_COLS + col;
    v.winner_idx = (v.pixel_idx >= 0 && v.pixel_idx < (int)rp.pixel_to_cloud_idx_.size())
        ? rp.pixel_to_cloud_idx_[v.pixel_idx] : -1;

    if (v.winner_idx != cloud_idx) {
        v.rimg_status = "excluded_occluded";
        v.cluster_pre_ds = "n/a";
        v.cluster_post_ds = "n/a";
        return v;
    }

    v.rimg_status = "entered_range_image";
    const int px = v.pixel_idx;
    if (pixels_in_cluster_pre.count(px)) {
        v.cluster_pre_ds = "in_cluster";
        auto it = pixel_to_cid_pre.find(px);
        v.cluster_id = (it != pixel_to_cid_pre.end()) ? it->second : -1;
    } else {
        const int label = (px >= 0 && px < (int)seg_pre_ds.label_map.size())
            ? seg_pre_ds.label_map[px] : 0;
        if (label > 0 && !kept_labels_pre.count(label))
            v.cluster_pre_ds = "cluster_too_small";
        else
            v.cluster_pre_ds = "not_in_cluster";
    }

    if (pixels_in_cluster_post.count(px)) {
        v.cluster_post_ds = "in_cluster";
        if (v.cluster_id < 0) {
            auto it = pixel_to_cid_post.find(px);
            v.cluster_id = (it != pixel_to_cid_post.end()) ? it->second : -1;
        }
    } else if (v.cluster_pre_ds == "in_cluster")
        v.cluster_post_ds = "removed_by_downsample";
    else if (v.cluster_pre_ds == "cluster_too_small")
        v.cluster_post_ds = "cluster_too_small";
    else
        v.cluster_post_ds = "not_in_cluster";

    return v;
}

std::unordered_set<int> buildClusterPixelSet(const SegmentationResult& seg)
{
    std::unordered_set<int> s;
    for (const auto& cluster : seg.clusters)
        for (int px : cluster) s.insert(px);
    return s;
}

std::unordered_map<int, int> buildPixelToClusterId(const SegmentationResult& seg)
{
    std::unordered_map<int, int> m;
    for (int cid = 0; cid < (int)seg.clusters.size(); ++cid)
        for (int px : seg.clusters[cid]) m[px] = cid;
    return m;
}

std::unordered_set<int> buildKeptLabels(const SegmentationResult& seg)
{
    std::unordered_set<int> labels;
    for (const auto& cluster : seg.clusters) {
        if (cluster.empty()) continue;
        const int px = cluster.front();
        if (px >= 0 && px < (int)seg.label_map.size())
            labels.insert(seg.label_map[px]);
    }
    return labels;
}

void bumpCount(std::unordered_map<std::string, int>& m, const std::string& k) { ++m[k]; }

void saveFarClusterPipelineAuditTxt(
    const pcl::PointCloud<pcl::PointXYZ>& scan_local,
    int step,
    const Transf& T_world,
    double ground_z_min,
    double ground_z_max,
    double z_floor,
    double split_dist,
    int far_min_cluster,
    const RangeImageProcessor& rp_far,
    const SegmentationResult& seg_pre_ds,
    const SegmentationResult& seg_post_ds)
{
    const Eigen::Matrix3d R_w = T_world.block<3, 3>(0, 0);
    const Eigen::Vector3d t_w = T_world.block<3, 1>(0, 3);

    const auto pixels_pre  = buildClusterPixelSet(seg_pre_ds);
    const auto pixels_post = buildClusterPixelSet(seg_post_ds);
    const auto px2cid_pre  = buildPixelToClusterId(seg_pre_ds);
    const auto px2cid_post = buildPixelToClusterId(seg_post_ds);
    const auto kept_labels = buildKeptLabels(seg_pre_ds);

    const std::string path = std::string(kBsplineBuildDir) + "/far_cluster_funnel_step"
                           + std::to_string(step) + ".txt";
    std::error_code ec;
    std::filesystem::create_directories(kBsplineBuildDir, ec);
    std::ofstream fout(path, std::ios::out | std::ios::trunc);
    if (!fout.is_open()) {
        std::cerr << "  [FarClusterFunnel] failed to open: " << path << std::endl;
        return;
    }

    fout << std::fixed << std::setprecision(6);
    fout << "# Far-distance scan_world points: non-ground, range in [split_dist, MAX_RANGE]\n";
    fout << "step: " << step << "\n";
    fout << "ground_band_z: [" << ground_z_min << ", " << ground_z_max << "]\n";
    fout << "z_floor: " << z_floor << "\n";
    fout << "split_dist: " << split_dist << "\n";
    fout << "far_range: [" << split_dist << ", " << rp_far.MAX_RANGE << "]\n";
    fout << "far_min_cluster: " << far_min_cluster << "\n";
    fout << "far_clusters_after_seg: " << seg_pre_ds.clusters.size() << "\n";
    fout << "far_clusters_after_downsample: " << seg_post_ds.clusters.size() << "\n";
    fout << "\n";
    fout << "# idx wx wy wz lx ly lz range gnd_band"
         << " rimg_status cluster_pre_ds cluster_post_ds"
         << " pix_u pix_v cluster_id winner_idx\n";

    int far_scope_total = 0;
    int ground_skipped = 0;
    int near_skipped = 0;
    std::unordered_map<std::string, int> rimg_summary, pre_summary, post_summary;

    for (int i = 0; i < (int)scan_local.size(); ++i) {
        const auto& pt = scan_local.points[i];
        const Eigen::Vector3d pw = R_w * Eigen::Vector3d(pt.x, pt.y, pt.z) + t_w;
        const bool in_gnd = (pt.z >= ground_z_min && pt.z <= ground_z_max);
        const double range = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);

        if (in_gnd) {
            ++ground_skipped;
            continue;
        }
        if (range < split_dist) {
            ++near_skipped;
            continue;
        }

        ++far_scope_total;
        const FarClusterVerdict verdict = classifyFarPointPipeline(
            pt, i, rp_far, split_dist, z_floor,
            seg_pre_ds, seg_post_ds,
            pixels_pre, pixels_post,
            px2cid_pre, px2cid_post, kept_labels);

        bumpCount(rimg_summary, verdict.rimg_status);
        bumpCount(pre_summary, verdict.cluster_pre_ds);
        bumpCount(post_summary, verdict.cluster_post_ds);

        fout << i << " "
             << pw.x() << " " << pw.y() << " " << pw.z() << " "
             << pt.x << " " << pt.y << " " << pt.z << " "
             << range << " " << (in_gnd ? 1 : 0) << " "
             << verdict.rimg_status << " "
             << verdict.cluster_pre_ds << " "
             << verdict.cluster_post_ds << " "
             << verdict.pixel_u << " " << verdict.pixel_v << " "
             << verdict.cluster_id << " " << verdict.winner_idx << "\n";
    }

    fout << "\n========== SUMMARY ==========\n";
    fout << "scan_total: " << scan_local.size() << "\n";
    fout << "skipped_ground_band: " << ground_skipped << "\n";
    fout << "skipped_near_range_lt_split: " << near_skipped << "\n";
    fout << "far_scope_total: " << far_scope_total << "\n";
    fout << "\n";
    fout << "[far_layer_range_image]\n";
    fout << "# Why point did not enter far range image\n";
    for (const auto& kv : rimg_summary)
        fout << "  " << kv.first << ": " << kv.second << "\n";
    const int entered_rimg = rimg_summary.count("entered_range_image")
        ? rimg_summary.at("entered_range_image") : 0;
    fout << "  entered_ratio: "
         << (far_scope_total > 0 ? 100.0 * entered_rimg / far_scope_total : 0.0) << "%\n";
    fout << "\n";
    fout << "[cluster_after_segmentation]\n";
    fout << "# Among entered_range_image points only meaningful; n/a = did not enter range image\n";
    for (const auto& kv : pre_summary)
        fout << "  " << kv.first << ": " << kv.second << "\n";
    const int in_cluster_pre = pre_summary.count("in_cluster") ? pre_summary.at("in_cluster") : 0;
    fout << "  in_cluster_ratio_of_far_scope: "
         << (far_scope_total > 0 ? 100.0 * in_cluster_pre / far_scope_total : 0.0) << "%\n";
    fout << "  in_cluster_ratio_of_entered_rimg: "
         << (entered_rimg > 0 ? 100.0 * in_cluster_pre / entered_rimg : 0.0) << "%\n";
    fout << "\n";
    fout << "[cluster_after_downsample]\n";
    for (const auto& kv : post_summary)
        fout << "  " << kv.first << ": " << kv.second << "\n";
    const int in_cluster_post = post_summary.count("in_cluster") ? post_summary.at("in_cluster") : 0;
    fout << "  in_cluster_ratio_of_far_scope: "
         << (far_scope_total > 0 ? 100.0 * in_cluster_post / far_scope_total : 0.0) << "%\n";
    fout << "\n";
    fout << "[legend]\n";
    fout << "  excluded_occluded: passed z/range but same pixel has closer point\n";
    fout << "  cluster_too_small: in range image, BFS label exists but cluster size <= far_min_cluster\n";
    fout << "  removed_by_downsample: was in cluster, dropped by cluster_ds_target_max grid thinning\n";

    fout.close();
    std::cout << "  [FarClusterFunnel] step=" << step
              << " far_scope=" << far_scope_total
              << " entered_rimg=" << entered_rimg
              << " in_cluster_pre=" << in_cluster_pre
              << " in_cluster_post=" << in_cluster_post
              << " -> " << path << std::endl;
}

int saveRangeImagePointsTxt(const RangeImageProcessor& range_proc,
                            const Transf& T_world,
                            const std::string& base_dir)
{
    const Eigen::Matrix3d R = T_world.block<3, 3>(0, 0);
    const Eigen::Vector3d t = T_world.block<3, 1>(0, 3);

    std::ofstream f_lidar(base_dir + "/rangeimage_lidar.txt", std::ios::out | std::ios::trunc);
    std::ofstream f_world(base_dir + "/rangeimage_world.txt", std::ios::out | std::ios::trunc);
    if (!f_lidar.is_open() || !f_world.is_open()) return 0;

    f_lidar << std::fixed << std::setprecision(6);
    f_world << std::fixed << std::setprecision(6);
    int n = 0;
    for (const auto& px : range_proc.range_image_) {
        if (!px.valid) continue;
        f_lidar << px.x << " " << px.y << " " << px.z << "\n";
        const Eigen::Vector3d pw = R * Eigen::Vector3d(px.x, px.y, px.z) + t;
        f_world << pw.x() << " " << pw.y() << " " << pw.z() << "\n";
        ++n;
    }
    return n;
}
}  // namespace

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
        for(int i = 1; i< step; i++){
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
    nh.param("slamesher/ri_h_scans", ri_h_scans, 64);
    nh.param("slamesher/ri_w_cols",  ri_w_cols,  1500);
    nh.param("slamesher/ri_fov_up",  ri_fov_up,  2.0);
    nh.param("slamesher/ri_fov_down",ri_fov_down,-24.8);
    nh.param("slamesher/lidar_flip_yz", lidar_flip_yz, false);
    std::cout << "range_image: H=" << ri_h_scans << " W=" << ri_w_cols
              << " FOV=[" << ri_fov_down << "," << ri_fov_up << "]"
              << "  lidar_flip_yz=" << (lidar_flip_yz ? "on" : "off") << std::endl;
    nh.param("slamesher/max_steps", max_steps, 1);
    nh.param("slamesher/max_frames", max_frames, 0);
    nh.param("slamesher/dump_frame", dump_frame, 0);
    nh.param("slamesher/dump_cluster_step", dump_cluster_step, 0);
    nh.param("slamesher/dump_occluded_step", dump_occluded_step, 0);
    nh.param("slamesher/dump_gnd_scan_begin", dump_gnd_scan_begin, 0);
    nh.param("slamesher/dump_gnd_scan_end", dump_gnd_scan_end, 0);
    nh.param("slamesher/all_surfaces_max_step", all_surfaces_max_step, 0);
    std::cout<<"max_steps: "<<max_steps<<std::endl;
    std::cout<<"max_frames: "<<max_frames<<(max_frames > 0 ? " (debug stop)" : " (run full sequence)")<<std::endl;
    std::cout<<"dump_frame: "<<dump_frame
             <<(dump_frame > 0 ? " (save matched/unmatched + far_cluster_funnel -> " + std::string(kBsplineBuildDir) + ")" : " (off)")
             <<std::endl;
    std::cout<<"dump_cluster_step: "<<dump_cluster_step
             <<(dump_cluster_step > 0 ? " (save clusters -> " + std::string(kBsplineBuildDir) + "/output_clusters)" : " (off)")
             <<std::endl;
    std::cout<<"dump_occluded_step: "<<dump_occluded_step
             <<(dump_occluded_step > 0 ? " (save rangeimage+occluded -> " + std::string(kBsplineBuildDir) + "/rangeimage_*,occluded_*.txt)" : " (off)")
             <<std::endl;
    {
        const bool gnd_scan_on = (dump_gnd_scan_begin > 0 && dump_gnd_scan_end >= dump_gnd_scan_begin);
        std::cout << "dump_gnd_scan: "
                  << (gnd_scan_on
                      ? (std::to_string(dump_gnd_scan_begin) + ".." + std::to_string(dump_gnd_scan_end)
                         + " -> " + std::string(kBsplineBuildDir)
                         + "/gnd_scan/ + whole_gnd/ + scan_world/ + all_surfaces/")
                      : "off")
                  << std::endl;
        if (gnd_scan_on) {
            for (const char* sub : {"gnd_scan", "whole_gnd", "scan_world", "all_surfaces"}) {
                const std::filesystem::path gnd_dir =
                    std::filesystem::path(kBsplineBuildDir) / sub;
                std::error_code ec;
                std::filesystem::remove_all(gnd_dir, ec);
                if (ec) {
                    std::cerr << "  [DumpGndScan] clear failed: " << gnd_dir
                              << " (" << ec.message() << ")\n";
                }
                ec.clear();
                std::filesystem::create_directories(gnd_dir, ec);
                if (ec) {
                    std::cerr << "  [DumpGndScan] mkdir failed: " << gnd_dir
                              << " (" << ec.message() << ")\n";
                } else {
                    std::cout << "  [DumpGndScan] cleared " << gnd_dir << std::endl;
                }
            }
        }
    }
    std::cout<<"all_surfaces_max_step: "<<all_surfaces_max_step
             <<(all_surfaces_max_step > 0 ? " (export surfaces with created_step <= N)" : " (off)")
             <<std::endl;
    nh.param("slamesher/file_loc_report", file_loc_report, std::string("not_set"));
    nh.param("slamesher/console_log_path", console_log_path, std::string(""));
    nh.param("slamesher/save_mesh_map", save_mesh_map, false);
    nh.param("slamesher/save_surface_samples", save_surface_samples, false);
    nh.param("slamesher/map_save_step_begin", map_save_step_begin, 0);
    nh.param("slamesher/map_save_step_end",   map_save_step_end,   0);
    std::cout << "map_save_step: "
              << (map_save_step_begin > 0 || map_save_step_end > 0
                  ? std::to_string(map_save_step_begin) + ".." + std::to_string(map_save_step_end)
                  : "off (save all at end)")
              << std::endl;

    //<!--  register param  -->-
    nh.param("slamesher/range_max",  range_max,  100.0);
    nh.param("slamesher/range_min",  range_min,  1.0);
    nh.param("slamesher/range_unit", range_unit, 1.0);
    nh.param("slamesher/register_times", register_times, 5);
    nh.param("slamesher/map_update_interval", map_update_interval, 20);
    nh.param("slamesher/ground_build_interval", ground_build_interval, 20);
    nh.param("slamesher/map_unmatched_ratio_min", map_unmatched_ratio_min, 0.30);
    nh.param("slamesher/cluster_ds_min_pts",    cluster_ds_min_pts,    100);
    nh.param("slamesher/cluster_ds_target_max", cluster_ds_target_max, 200);
    nh.param("slamesher/range_image_split",          range_image_split,          30.0);
    nh.param("slamesher/range_image_far_z_floor_offset", range_image_far_z_floor_offset, 20.0);
    nh.param("slamesher/range_image_far_min_cluster", range_image_far_min_cluster, 20);
    nh.param("slamesher/obs_rimg_col_step",           obs_rimg_col_step,           3);
    nh.param("slamesher/obs_match_per_surf_max",      obs_match_per_surf_max,      50);
    nh.param("slamesher/obs_cand_target",             obs_cand_target,             1500);
    nh.param("slamesher/obs_reg_use_pw_mask",         obs_reg_use_pw_mask,         true);
    nh.param("slamesher/num_thread_reg",              num_thread_reg,              0);
    nh.param("slamesher/reg_time_breakdown",          reg_time_breakdown,          false);
    nh.param("slamesher/obs_reuse_cand_delta",        obs_reuse_cand_delta,        0.0);
    nh.param("slamesher/gnd_expand_last_iter_only",   gnd_expand_last_iter_only,   false);
    nh.param("slamesher/use_patchwork_ground", use_patchwork_ground, true);
    {
        const PatchworkppConfig d;
        nh.param("slamesher/pw_sensor_height",   patchwork.sensor_height,   d.sensor_height);
        nh.param("slamesher/pw_num_iter",        patchwork.num_iter,        d.num_iter);
        nh.param("slamesher/pw_num_lpr",         patchwork.num_lpr,         d.num_lpr);
        nh.param("slamesher/pw_num_min_pts",     patchwork.num_min_pts,     d.num_min_pts);
        nh.param("slamesher/pw_th_seeds",        patchwork.th_seeds,        d.th_seeds);
        nh.param("slamesher/pw_th_dist",         patchwork.th_dist,         d.th_dist);
        nh.param("slamesher/pw_th_seeds_v",      patchwork.th_seeds_v,      d.th_seeds_v);
        nh.param("slamesher/pw_th_dist_v",       patchwork.th_dist_v,       d.th_dist_v);
        nh.param("slamesher/pw_max_r",           patchwork.max_range,       d.max_range);
        nh.param("slamesher/pw_min_r",           patchwork.min_range,       d.min_range);
        nh.param("slamesher/pw_uprightness_thr", patchwork.uprightness_thr, d.uprightness_thr);
        nh.param("slamesher/pw_adaptive_seed_selection_margin",
                 patchwork.adaptive_seed_selection_margin,
                 d.adaptive_seed_selection_margin);
        nh.param("slamesher/pw_enable_RVPF",     patchwork.enable_RVPF,     d.enable_RVPF);
        nh.param("slamesher/pw_enable_TGR",      patchwork.enable_TGR,      d.enable_TGR);
        std::cout << "patchwork++: " << (use_patchwork_ground ? "ON" : "OFF")
                  << " sensor_h=" << patchwork.sensor_height
                  << " r=[" << patchwork.min_range << "," << patchwork.max_range << "]"
                  << " th_dist=" << patchwork.th_dist
                  << " th_seeds=" << patchwork.th_seeds
                  << " upright=" << patchwork.uprightness_thr
                  << " RVPF=" << patchwork.enable_RVPF
                  << " TGR=" << patchwork.enable_TGR
                  << " (RNR 需 intensity，当前管线未提供，自动跳过)" << std::endl;
    }
    nh.param("slamesher/ground_z_min",        ground_z_min,        -3.0);
    nh.param("slamesher/ground_z_max",        ground_z_max,        -1.5);
    nh.param("slamesher/ground_cell_size",    ground_cell_size,    6.0);
    nh.param("slamesher/ground_cell_min_pts", ground_cell_min_pts, 80);
    nh.param("slamesher/ground_cell_num_cp",  ground_cell_num_cp,  5);
    nh.param("slamesher/ground_query_radius", ground_query_radius, 1);
    nh.param("slamesher/ground_cell_max_pts",    ground_cell_max_pts,    400);
    nh.param("slamesher/ground_fit_max_pts",     ground_fit_max_pts,     150);
    nh.param("slamesher/ground_cell_z_pct",      ground_cell_z_pct,      0.0);
    nh.param("slamesher/ground_cell_z_tol",      ground_cell_z_tol,      0.2);
    nh.param("slamesher/ground_cell_z_filter",   ground_cell_z_filter,   false);
    nh.param("slamesher/ground_skip_points",  ground_skip_points,  40);
    nh.param("slamesher/ground_reg_cell_size",        ground_reg_cell_size,        3.0);
    nh.param("slamesher/ground_reg_cell_max_pts",       ground_reg_cell_max_pts,       30);
    nh.param("slamesher/ground_reg_cell_max_pts_front", ground_reg_cell_max_pts_front, 90);
    nh.param("slamesher/ground_reg_cell_target_pts",    ground_reg_cell_target_pts,    50);
    nh.param("slamesher/ground_reg_nbr_per_seed",       ground_reg_nbr_per_seed,       40);
    nh.param("slamesher/ground_reg_y_max",              ground_reg_y_max,              25.0);
    nh.param("slamesher/ground_reg_x_max",              ground_reg_x_max,              0.0);
    nh.param("slamesher/ground_reg_x_max_front",        ground_reg_x_max_front,        0.0);
    nh.param("slamesher/ground_thr_last",               ground_thr_last,               0.03);
    nh.param("slamesher/predict_horizontal_only",       predict_horizontal_only,       false);
    nh.param("slamesher/scan_correct_deg",              scan_correct_deg,              0.0);
    nh.param("slamesher/scan_correct_ground_only",      scan_correct_ground_only,      true);
    nh.param("slamesher/reg_alternations",              reg_alternations,              1);
    nh.param("slamesher/obs_thr_last",                  obs_thr_last,                  0.0);
    nh.param("slamesher/obs_huber",                     obs_huber,                     0.5);
    nh.param("slamesher/obs_ceres_iters",               obs_ceres_iters,               1);
    nh.param("slamesher/obs_seg_normal_deg",            obs_seg_normal_deg,            0.0);
    nh.param("slamesher/obs_fit_max_dist",              obs_fit_max_dist,              0.0);
    nh.param("slamesher/obs_fit_split_dist",            obs_fit_split_dist,            0.0);
    nh.param("slamesher/obs_edge_mode",                 obs_edge_mode,                 0);
    nh.param("slamesher/obs_edge_tan_max",              obs_edge_tan_max,              2.0);
    nh.param("slamesher/obs_map_replace_frac",          obs_map_replace_frac,          0.0);
    nh.param("slamesher/obs_map_voxel_owner",           obs_map_voxel_owner,           false);
    nh.param("slamesher/obs_fit_pts_per_cp",            obs_fit_pts_per_cp,            0.0);
    nh.param("slamesher/obs_fit_max_iter",              obs_fit_max_iter,              10);
    nh.param("slamesher/obs_fit_eps",                   obs_fit_eps,                   0.05);
    nh.param("slamesher/obs_fit_weight_sigma",          obs_fit_weight_sigma,          0.0);
    nh.param("slamesher/obs_fit_weight_adj",            obs_fit_weight_adj,            false);
    nh.param("slamesher/ground_map_r_max",              ground_map_r_max,              0.0);
    nh.param("slamesher/ground_reg_total_max",          ground_reg_total_max,          4000);
    nh.param("slamesher/ground_reg_match_max",          ground_reg_match_max,          2500);
    nh.param("slamesher/ground_reg_fb_x0",              ground_reg_fb_x0,              5.0);
    nh.param("slamesher/ground_reg_fb_front",           ground_reg_fb_front,           0.34);
    nh.param("slamesher/ground_reg_fb_mid",             ground_reg_fb_mid,             0.33);
    nh.param("slamesher/ground_reg_fb_back",            ground_reg_fb_back,            0.33);
    nh.param("slamesher/ground_coarse_cell_size",     ground_coarse_cell_size,     20.0);
    nh.param("slamesher/ground_coarse_min_pts",       ground_coarse_min_pts,       30);
    nh.param("slamesher/ground_coarse_num_cp",        ground_coarse_num_cp,        7);
    nh.param("slamesher/ground_coarse_query_radius",  ground_coarse_query_radius,  1);
    nh.param("slamesher/ground_coarse_cell_max_pts",  ground_coarse_cell_max_pts,  600);
    nh.param("slamesher/ground_coarse_fit_max_pts",   ground_coarse_fit_max_pts,   250);
    nh.param("slamesher/gnd_ri_z_min",       gnd_ri_z_min,       -3.0);
    nh.param("slamesher/gnd_ri_z_max",       gnd_ri_z_max,        0.0);
    nh.param("slamesher/gnd_near_x_max",     gnd_near_x_max,     20.0);
    nh.param("slamesher/gnd_near_y_max",     gnd_near_y_max,     10.0);
    nh.param("slamesher/gnd_near_z_max",     gnd_near_z_max,     -1.5);
    nh.param("slamesher/gnd_col_max_step",   gnd_col_max_step,    0.3);
    nh.param("slamesher/gnd_col_seed_h_up",  gnd_col_seed_h_up,   0.4);
    nh.param("slamesher/gnd_normal_z_min",   gnd_normal_z_min,    0.5);
    nh.param("slamesher/obs_starve_res_lo",  obs_starve_res_lo,   0);
    nh.param("slamesher/obs_starve_res_hi",  obs_starve_res_hi,   0);
    nh.param("slamesher/obs_starve_warmup",  obs_starve_warmup,  30);
    nh.param("slamesher/obs_prior_sigma_xy", obs_prior_sigma_xy,  0.0);
    nh.param("slamesher/obs_prior_sigma_yaw",obs_prior_sigma_yaw, 0.0);
    nh.param("slamesher/obs_prior_along_only", obs_prior_along_only, false);
    nh.param("slamesher/obs_lr_max_frac",      obs_lr_max_frac,      0.0);
    nh.param("slamesher/obs_range_ref",        obs_range_ref,        0.0);
    nh.param("slamesher/gnd_roll_prior_sigma", gnd_roll_prior_sigma, 0.0);
    nh.param("slamesher/gnd_pitch_prior_sigma", gnd_pitch_prior_sigma, 0.0);
    nh.param("slamesher/bootstrap_step2_tx", bootstrap_step2_tx, 0.5);
    nh.param("slamesher/bootstrap_step2_ty", bootstrap_step2_ty, 0.0);
    nh.param("slamesher/bootstrap_step2_tz", bootstrap_step2_tz, 0.0);
    std::cout << "cluster_ds: min_pts=" << cluster_ds_min_pts
              << " target_max=" << cluster_ds_target_max
              << (cluster_ds_target_max > 0 ? " (range-image 2D downsample on)" : " (off)") << std::endl;
    std::cout << "range_image_split: " << range_image_split
              << (range_image_split > 0 ? " m (two-layer range image on)" : " (off, single layer)")
              << "  far_z_floor_offset=" << range_image_far_z_floor_offset
              << "  far_min_cluster=" << range_image_far_min_cluster << std::endl;
    std::cout << "obs_rimg_col_step=" << obs_rimg_col_step
              << "  obs_match_per_surf_max=" << obs_match_per_surf_max
              << "  obs_cand_target=" << obs_cand_target
              << "  obs_reg_use_pw_mask=" << (obs_reg_use_pw_mask ? "on" : "off")
              << "  obs_edge_mode=" << obs_edge_mode
              << "  obs_edge_tan_max=" << obs_edge_tan_max
              << "  obs_map_replace_frac=" << obs_map_replace_frac
              << "  obs_map_voxel_owner=" << (obs_map_voxel_owner ? "on" : "off")
              << "  obs_fit_pts_per_cp=" << obs_fit_pts_per_cp
              << "  obs_fit_max_iter=" << obs_fit_max_iter
              << "  obs_fit_eps=" << obs_fit_eps
              << "  obs_lr_max_frac=" << obs_lr_max_frac
              << "  obs_range_ref=" << obs_range_ref
              << "  obs_seg_normal_deg=" << obs_seg_normal_deg
              << "  obs_fit_max_dist=" << obs_fit_max_dist
              << "  obs_fit_split_dist=" << obs_fit_split_dist
              << "  obs_ceres_iters=" << obs_ceres_iters << std::endl;
    std::cout << "scan_correct_deg: " << scan_correct_deg
              << (scan_correct_deg != 0.0 ? " (HDL-64 vertical angle fix on)" : " (off)")
              << "  ground_only=" << (scan_correct_ground_only ? "yes" : "no") << std::endl;
    std::cout << "ground_grid(fine): z_band=[" << ground_z_min << "," << ground_z_max << "]"
              << " cell=" << ground_cell_size
              << "m min_pts=" << ground_cell_min_pts
              << " num_cp=" << ground_cell_num_cp
              << " query_r=" << ground_query_radius
              << " cell_max=" << ground_cell_max_pts
              << " (map: XY-first, dense-only downsample)"
              << " fit_max=" << ground_fit_max_pts
              << " z_filter=" << (ground_cell_z_filter ? "on" : "off")
              << " z_pct=" << ground_cell_z_pct
              << " z_tol=" << ground_cell_z_tol << "m"
              << " reg_cell=" << ground_reg_cell_size << "m"
              << " reg_cell_max=" << ground_reg_cell_max_pts
              << " front_max=" << ground_reg_cell_max_pts_front
              << " reg_target=" << ground_reg_cell_target_pts
              << " reg_nbr/seed=" << ground_reg_nbr_per_seed
              << " reg_y_max=" << ground_reg_y_max << "m"
              << " reg_x_max=" << ground_reg_x_max << "m"
              << " reg_x_max_f=" << ground_reg_x_max_front << "m"
              << " map_r_max=" << ground_map_r_max << "m"
              << " reg_total_max=" << ground_reg_total_max
              << " reg_match_max=" << ground_reg_match_max
              << " fb_x0=" << ground_reg_fb_x0 << "m"
              << " fb(F/M/B)=" << ground_reg_fb_front << "/" << ground_reg_fb_mid << "/" << ground_reg_fb_back
              << std::endl;
    std::cout << "ground_grid(coarse): cell=" << ground_coarse_cell_size
              << "m min_pts=" << ground_coarse_min_pts
              << " num_cp=" << ground_coarse_num_cp
              << " query_r=" << ground_coarse_query_radius
              << " cell_max=" << ground_coarse_cell_max_pts
              << " fit_max=" << ground_coarse_fit_max_pts << std::endl;
    std::cout << "gnd_ri: z=[" << gnd_ri_z_min << "," << gnd_ri_z_max << "]"
              << " near(|x|<=" << gnd_near_x_max << ",|y|<=" << gnd_near_y_max
              << ")->z_max=" << gnd_near_z_max
              << " col_dz_max=" << gnd_col_max_step
              << " seed_h_up=" << gnd_col_seed_h_up
              << " normal_z_min=" << gnd_normal_z_min << std::endl;
    std::cout << "bootstrap_step2 (velo m): [" << bootstrap_step2_tx << ", "
              << bootstrap_step2_ty << ", " << bootstrap_step2_tz << "]" << std::endl;
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
    if (num_thread_reg <= 0) num_thread_reg = num_thread;
    std::cout << "num_thread=" << num_thread
              << "  num_thread_reg=" << num_thread_reg
              << "  reg_time_breakdown=" << (reg_time_breakdown ? "on" : "off")
              << "  obs_reuse_cand_delta=" << obs_reuse_cand_delta
              << "  gnd_expand_last_iter_only="
              << (gnd_expand_last_iter_only ? "on" : "off") << std::endl;
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
        else if (g_data.step == 2) {
            // 第二帧无运动历史：用固定前进平移作初值（Velodyne +X）
            Transf dT = Eigen::Matrix4d::Identity();
            dT(0, 3) = param.bootstrap_step2_tx;
            dT(1, 3) = param.bootstrap_step2_ty;
            dT(2, 3) = param.bootstrap_step2_tz;
            odom = g_data.T_seq[g_data.step - 1] * dT;
        }
        else if (g_data.step > 2) {
            // 匀速外推：T_{t-1} * (T_{t-2}^{-1} T_{t-1})
            odom = g_data.T_seq[g_data.step - 1] * g_data.T_seq[g_data.step - 2].inverse() * g_data.T_seq[g_data.step - 1];
            if (param.predict_horizontal_only) {
                // roll/pitch/z 由地面阶段求解，而地面阶段的修正量恒等于「本帧速率减
                // 上帧速率」，对速率本身没有回复力。再对这三个自由度做匀速外推，等于
                // 给它们各接一个自由积分器：任何速率都是不动点，早期噪声取到的漂移率
                // 会被永久保持下去。只外推 x/y/yaw（每帧真有米级运动，障碍阶段需要）。
                const Transf& Tp = g_data.T_seq[g_data.step - 1];
                double r_e, p_e, y_e, r_p, p_p, y_p;
                matrixToRPY(odom.block<3, 3>(0, 0), r_e, p_e, y_e);
                matrixToRPY(Tp.block<3, 3>(0, 0),   r_p, p_p, y_p);
                odom = makePoseFromXYZRPY(odom(0, 3), odom(1, 3), Tp(2, 3),
                                          r_p, p_p, y_e);
            }
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

int SLAMesher::chooseControlGridSize(int num_fitting_points, double pts_per_cp)
{
    constexpr int kMinCp = 4;
    constexpr int kMaxCp = 15;

    // 每个控制点 3 个自由度，下面这套分档在低端只有 5 点/控制点（80 点拟合 48 个自由度），
    // 属欠定：曲面会去插值噪声，且欠定在参数域边缘最严重——而 23%~56% 的配准匹配足点
    // 正落在边缘。>0 时改按目标点密度反解控制网格边长。
    if (pts_per_cp > 0.0) {
        const int m = (int)std::lround(
            std::sqrt((double)num_fitting_points / pts_per_cp));
        return std::max(kMinCp, std::min(m, kMaxCp));
    }

    int n;
    if (num_fitting_points < 80)
        n = 4;
    else if (num_fitting_points < 150)
        n = 5;
    else if (num_fitting_points < 300)
        n = 6;
    else if (num_fitting_points < 500)
        n = 8;
    else if (num_fitting_points < 900)
        n = 9;
    else if (num_fitting_points < 1000)
        n = 10;
    else
        n = 12;

    return std::max(kMinCp, std::min(n, kMaxCp));
}

void SLAMesher::processFirstFrame(Transf& T_world)
{
    g_data.updatePose(T_world);
    pubTf();
    path_pub.publish(g_data.path);
}

// 障碍匹配：对 surf_to_pts 中所有 (sid, indices) 做 footprint 并写入 matches
void SLAMesher::buildObsMatches(
        const pcl::PointCloud<pcl::PointXYZ>& scan_local,
        const Transf& T_curr,
        double cur_match_thr,
        bool want_ground,
        BSplineMap& bspline_map,
        const std::unordered_map<int, std::vector<int>>& surf_to_pts,
        std::unordered_map<int, std::vector<int>>& prev_indices,
        std::unordered_map<int, std::vector<std::pair<BSplineSurface::Parameter, BSplineSurface::Parameter>>>& prev_uv,
        int& stat_edge_kept, int& stat_edge_rej, int& stat_gate_rej,
        std::vector<RegMatch>& out)
{
    // 任务表按 surf_to_pts 的自然迭代序建立；并行段只写各自槽位，归并再按同一序
    // 追加到 out。out 的顺序决定 Ceres 残差块的加入顺序，进而决定浮点求和顺序，
    // 所以这里必须与串行版逐位一致（Ceres 本身仍是 num_threads=1）。
    struct ObsTask {
        int sid = -1;
        const std::vector<int>* indices = nullptr;
        BSplineSurface* surf = nullptr;
        std::vector<std::pair<BSplineSurface::Parameter, BSplineSurface::Parameter>> footprints;
        std::vector<RegMatch> matches;
        int edge_kept = 0, edge_rej = 0, gate_rej = 0;
    };

    std::vector<ObsTask> tasks;
    tasks.reserve(surf_to_pts.size());
    for (const auto& [sid, indices] : surf_to_pts) {
        const BSplineMapEntry* entry = bspline_map.getEntry(sid);
        if (!entry || !entry->surface) continue;
        if (entry->is_ground != want_ground) continue;
        if (indices.empty()) continue;

        ObsTask t;
        t.sid     = sid;
        t.indices = &indices;
        t.surf    = entry->surface.get();
        // warm start：同一批下标才能复用上次收敛的 uv。命中后 findFootPrintWarm
        // 内部的收敛判据会提前退出，Newton 步数远少于 6。
        auto pi = prev_indices.find(sid);
        auto pu = prev_uv.find(sid);
        if (pi != prev_indices.end() && pu != prev_uv.end()
            && pi->second == indices && pu->second.size() == indices.size()) {
            t.footprints = pu->second;
        }
        tasks.push_back(std::move(t));
    }

    const int n_task = static_cast<int>(tasks.size());
    const int n_thr  = std::max(1, param.num_thread_reg);
    const Eigen::Matrix3d R_curr = T_curr.block<3,3>(0,0);
    const Eigen::Vector3d t_curr = T_curr.block<3,1>(0,3);

    // findFootPrintWarm / getCurvature 只读曲面的 knots/controls，写的都是出参；
    // 每个任务对应唯一 sid，曲面之间不重叠，故可并发。
#pragma omp parallel for schedule(dynamic) num_threads(n_thr)
    for (int ti = 0; ti < n_task; ++ti) {
        ObsTask& t = tasks[ti];
        const std::vector<int>& indices = *t.indices;
        BSplineSurface* surf = t.surf;
        const auto& knU = surf->getKnotsU();
        const auto& knV = surf->getKnotsV();
        const auto& cps = surf->getControls();
        const int num_cpv = surf->getNumCpV();

        std::vector<Eigen::Vector3d> pts_world;
        pts_world.reserve(indices.size());
        for (int idx : indices)
            pts_world.push_back(R_curr *
                Eigen::Vector3d(scan_local[idx].x, scan_local[idx].y, scan_local[idx].z)
                + t_curr);

        std::vector<double> dists;
        surf->findFootPrintWarm(pts_world, t.footprints, dists, /*newton_steps=*/6);

        t.matches.reserve(indices.size());
        for (int k = 0; k < (int)indices.size(); k++) {
            double d = std::sqrt(std::abs(dists[k]));
            constexpr double kEdgeEps = 1e-4;
            const bool on_edge =
                t.footprints[k].first.second  < kEdgeEps ||
                t.footprints[k].first.second  > 1.0 - kEdgeEps ||
                t.footprints[k].second.second < kEdgeEps ||
                t.footprints[k].second.second > 1.0 - kEdgeEps;
            if (d < 1e-6) { ++t.gate_rej; if (on_edge) ++t.edge_rej; continue; }

            // 地面片是瓦片式铺开的，瓦片边缘的足点由邻接瓦片正常覆盖，不算外点；
            // 边界处理只对障碍片生效。
            const bool edge_obs = on_edge && !want_ground;
            auto [paraU, paraV] = t.footprints[k];

            SurfaceCurvature curv;
            bool curv_ready = false;
            bool drop_tangent = false;

            if (edge_obs && param.obs_edge_mode == 2) {
                curv = surf->getCurvature(paraU, paraV, knU, knV, cps, num_cpv);
                if (curv.normal.norm() < 1e-9) continue;
                curv.normal.normalize();
                curv_ready = true;
                const Eigen::Vector3d r = pts_world[k] - curv.point;
                const double dn  = std::abs(curv.normal.dot(r));
                const double dtan = std::sqrt(std::max(0.0, r.squaredNorm() - dn * dn));
                if (dn > cur_match_thr || dtan > param.obs_edge_tan_max) {
                    ++t.gate_rej;
                    ++t.edge_rej;
                    continue;
                }
                drop_tangent = true;
            } else {
                if (d > cur_match_thr) {
                    ++t.gate_rej;
                    if (on_edge) ++t.edge_rej;
                    continue;
                }
                if (edge_obs && param.obs_edge_mode == 1) {
                    ++t.edge_kept;
                    continue;
                }
            }
            if (on_edge) ++t.edge_kept;

            if (!curv_ready) {
                curv = surf->getCurvature(paraU, paraV, knU, knV, cps, num_cpv);
                if (curv.normal.norm() < 1e-9) continue;
                curv.normal.normalize();
            }

            RegMatch m;
            m.p_local   = Eigen::Vector3d(scan_local[indices[k]].x,
                                          scan_local[indices[k]].y,
                                          scan_local[indices[k]].z);
            m.p_world   = pts_world[k];
            m.curvature = curv;
            m.scan_idx  = indices[k];
            m.is_ground = want_ground;
            m.drop_tangent = drop_tangent;
            m.fit_dist  = param.obs_fit_weight_adj ? surf->getFitMeanDistAdj()
                                                   : surf->getFitMeanDist();
            t.matches.push_back(m);
        }
    }

    for (ObsTask& t : tasks) {
        prev_indices[t.sid] = *t.indices;
        prev_uv[t.sid]      = std::move(t.footprints);
        stat_edge_kept += t.edge_kept;
        stat_edge_rej  += t.edge_rej;
        stat_gate_rej  += t.gate_rej;
        out.insert(out.end(), std::make_move_iterator(t.matches.begin()),
                              std::make_move_iterator(t.matches.end()));
    }
}

// 地面匹配：使用 GroundGridMap；门限由调用方在收集后过滤
void SLAMesher::buildGroundMatches(
        const pcl::PointCloud<pcl::PointXYZ>& scan_local,
        const Transf& T_curr,
        double thr,
        const std::unordered_map<const BSplineSurface*, std::vector<int>>& surf_to_pts,
        std::unordered_map<const BSplineSurface*, std::vector<int>>& prev_indices,
        std::unordered_map<const BSplineSurface*, std::vector<std::pair<BSplineSurface::Parameter, BSplineSurface::Parameter>>>& prev_uv,
        std::vector<RegMatch>& out)
{
    // 按点集最小扫描下标定序后再遍历。原实现直接迭代以曲面指针为 key 的
    // unordered_map，顺序随堆地址（ASLR）变化；而下游按 out 的顺序做每格截顶
    // （每格只留前 reg_tgt 个），于是同配置重跑保留的匹配子集不同，实测 seq06
    // 第 2 帧 z 就差 1.5 mm。每个扫描点只归一张最近面，点集不相交，最小下标严格可比。
    struct GndTask {
        const BSplineSurface* surf = nullptr;
        const std::vector<int>* indices = nullptr;
        std::vector<std::pair<BSplineSurface::Parameter, BSplineSurface::Parameter>> footprints;
        std::vector<RegMatch> matches;
    };

    std::vector<std::pair<int, const BSplineSurface*>> order;
    order.reserve(surf_to_pts.size());
    for (const auto& [sp, idxs] : surf_to_pts) {
        if (!sp || idxs.empty()) continue;
        order.emplace_back(*std::min_element(idxs.begin(), idxs.end()), sp);
    }
    std::sort(order.begin(), order.end());

    std::vector<GndTask> tasks;
    tasks.reserve(order.size());
    for (const auto& [order_key, surf_ptr] : order) {
        (void)order_key;
        GndTask t;
        t.surf    = surf_ptr;
        t.indices = &surf_to_pts.at(surf_ptr);
        auto pi = prev_indices.find(surf_ptr);
        auto pu = prev_uv.find(surf_ptr);
        if (pi != prev_indices.end() && pu != prev_uv.end()
            && pi->second == *t.indices && pu->second.size() == t.indices->size()) {
            t.footprints = pu->second;
        }
        tasks.push_back(std::move(t));
    }

    const int n_task = static_cast<int>(tasks.size());
    const int n_thr  = std::max(1, param.num_thread_reg);
    const Eigen::Matrix3d R_curr = T_curr.block<3,3>(0,0);
    const Eigen::Vector3d t_curr = T_curr.block<3,1>(0,3);

    // 每个扫描点只归一张最近面，任务间曲面不重叠；归并按 order 序，与串行版一致。
#pragma omp parallel for schedule(dynamic) num_threads(n_thr)
    for (int ti = 0; ti < n_task; ++ti) {
        GndTask& t = tasks[ti];
        const std::vector<int>& indices = *t.indices;
        BSplineSurface* surf = const_cast<BSplineSurface*>(t.surf);
        const auto& knU  = surf->getKnotsU();
        const auto& knV  = surf->getKnotsV();
        const auto& cps  = surf->getControls();
        const int num_cpv = surf->getNumCpV();

        std::vector<Eigen::Vector3d> pts_world;
        pts_world.reserve(indices.size());
        for (int idx : indices)
            pts_world.push_back(R_curr *
                Eigen::Vector3d(scan_local[idx].x, scan_local[idx].y, scan_local[idx].z)
                + t_curr);

        std::vector<double> dists;
        surf->findFootPrintWarm(pts_world, t.footprints, dists, /*newton_steps=*/6);

        t.matches.reserve(indices.size());
        for (int k = 0; k < (int)indices.size(); k++) {
            double d = std::sqrt(std::abs(dists[k]));
            if (d > thr || d < 1e-6) continue;
            auto [paraU, paraV] = t.footprints[k];
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
            m.is_ground = true;
            t.matches.push_back(m);
        }
    }

    for (GndTask& t : tasks) {
        prev_indices[t.surf] = *t.indices;
        prev_uv[t.surf]      = std::move(t.footprints);
        out.insert(out.end(), std::make_move_iterator(t.matches.begin()),
                              std::make_move_iterator(t.matches.end()));
    }
}

Transf SLAMesher::registerScanToMap(const pcl::PointCloud<pcl::PointXYZ>& scan_local,
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
                                    bool use_far)
{
    Transf T_curr = T_guess;
    double delta_scale = 100.0;
    int prev_match_count = -1;
    std::vector<RegMatch> last_matches;
    std::unordered_map<int, std::vector<int>> obs_prev_indices;
    std::unordered_map<int, std::vector<std::pair<BSplineSurface::Parameter, BSplineSurface::Parameter>>> obs_prev_uv;
    std::unordered_map<const BSplineSurface*, std::vector<int>> gnd_prev_indices;
    std::unordered_map<const BSplineSurface*, std::vector<std::pair<BSplineSurface::Parameter, BSplineSurface::Parameter>>> gnd_prev_uv;

    // 障碍阶段逐轮收紧的匹配门；地面阶段在自己的循环里复位成 match_dist_thr
    double cur_match_thr = match_dist_thr;

    // 观测计数：足点被夹到参数域边界说明最近点落在 PCA 数据支撑框之外，此时 d 由切向
    // 偏移主导而残差主项是法向分量，两个量不是一回事。分开统计过门与被门毙掉的边界命中。
    int stat_edge_kept = 0, stat_edge_rej = 0, stat_gate_rej = 0;

    // 阶段 2 的产物，外层交替时每轮覆盖，循环结束后给 dump 用
    std::vector<int> gnd_enter_indices;      // 最后一轮进入地面匹配的点索引
    std::vector<RegMatch> last_gnd_matches;  // 最后一轮过 thr、进 Ceres 的地面匹配

    // 分段耗时累计（毫秒），param.reg_time_breakdown 打开时每帧打印
    const bool prof = param.reg_time_breakdown;
    double ms_obs_collect = 0, ms_obs_match = 0, ms_obs_solve = 0;
    double ms_gnd_sample = 0, ms_gnd_match = 0, ms_gnd_solve = 0;
    TicToc t_seg;

    // 障碍解 x/y/yaw 时 roll/pitch/z 取自外推，地面解完 roll/pitch/z 后不再回头，
    // 于是用错误姿态算出的 x/y/yaw 永久留在结果里。多跑几轮让两组自由度互相收敛。
    const int n_alt = std::max(1, param.reg_alternations);
    for (int alt = 0; alt < n_alt; ++alt) {
    delta_scale = 100.0;

    // ══════════════════════════════════════════════════════════════════════
    // 阶段 1：障碍匹配 — 只优化世界系 x / y / yaw；z / roll / pitch 固定
    // ══════════════════════════════════════════════════════════════════════
    int obs_last_res = -1;
    // 跨迭代持有：obs_reuse_cand_delta 开启时上一轮的候选集可以被下一轮沿用
    std::unordered_map<int, std::vector<int>> obs_surf_to_pts;
    for (int iter = 0; iter < max_iters && delta_scale > converge_thr; iter++) {
        // 首轮用大门吃掉外推残差，之后几何收紧到 obs_thr_last 剔外点（动态车辆、
        // 植被、鬼面）。原实现四轮恒定 0.8 m，末轮仍在吸收明显不该要的对应。
        cur_match_thr = match_dist_thr;
        if (param.obs_thr_last > 0.0 && param.obs_thr_last < match_dist_thr
            && max_iters > 1) {
            const double t = (double)std::min(iter, max_iters - 1) / (max_iters - 1);
            cur_match_thr = match_dist_thr
                          * std::pow(param.obs_thr_last / match_dist_thr, t);
        }
        const int col_step = std::max(1, param.obs_rimg_col_step);
        const int surf_cap = param.obs_match_per_surf_max;
        const double far_match_min_z = farLayerZFloor(ground_z_min);

        // 每个有效像素要做 27 次体素哈希查询，queryCandidates 内部还各带一次
        // unordered_set + vector 的堆分配；这段每帧要跑 max_iters 轮且全串行。
        // 改成按行并行，命中先落进本行的桶，再按行序归并到 obs_surf_to_pts ——
        // 各 sid 下的 cloud_idx 顺序与串行版完全相同（下游截顶按该顺序等间隔取）。
        std::vector<std::vector<std::pair<int,int>>> row_hits;
        auto collectFromRangeImage = [&](const RangeImageProcessor& rp, double obs_min_z, int v0) {
            const int W = rp.W_COLS;
            const int H = rp.H_SCANS;
            const int n_thr = std::max(1, param.num_thread_reg);
            const Eigen::Matrix3d R_reg = T_curr.block<3,3>(0,0);
            const Eigen::Vector3d t_reg = T_curr.block<3,1>(0,3);
            row_hits.resize(H);            // 跨调用复用各行缓冲，只清内容不还容量
            for (auto& h : row_hits) h.clear();
            // 各行有效像素数差异大（近地面的行密、朝天的行几乎全空），用 dynamic
#pragma omp parallel for schedule(dynamic) num_threads(n_thr)
            for (int u = 0; u < H; ++u) {
                std::vector<std::pair<int,int>>& hits = row_hits[u];
                for (int v = v0; v < W; v += col_step) {
                    const int px_idx = u * W + v;
                    const auto& px = rp.range_image_[px_idx];
                    if (!px.valid) continue;
                    if (px.z < obs_min_z) continue;
                    const int cloud_idx = rp.pixel_to_cloud_idx_[px_idx];
                    if (cloud_idx < 0 || cloud_idx >= (int)scan_local.size()) continue;
                    Eigen::Vector3d p_w = R_reg * Eigen::Vector3d(px.x, px.y, px.z) + t_reg;
                    for (int sid : bspline_map.queryCandidates(p_w, 1))
                        hits.emplace_back(sid, cloud_idx);
                }
            }
            for (const auto& hits : row_hits)
                for (const auto& [sid, cloud_idx] : hits)
                    obs_surf_to_pts[sid].push_back(cloud_idx);
        };
        // 逐曲面截顶后的候选量：已满额的大曲面不会因补采而增加，
        // 补采的增量只落在未满额的小曲面上，故用截顶后的量做触发判据
        auto cappedTotal = [&]() {
            int n = 0;
            for (const auto& [sid, idxs] : obs_surf_to_pts)
                n += (surf_cap > 0) ? std::min((int)idxs.size(), surf_cap) : (int)idxs.size();
            return n;
        };

        // delta_scale 此刻还是上一轮解出的位姿增量。增量足够小时候选集实质不变，
        // 跳过整张 range image 的重扫 + queryCandidates。附带效果更值钱：下标列表
        // 不变才能命中 buildObsMatches 的 warm-start，命中后 Newton 靠内部收敛判据
        // 提前退出。会改变结果（候选略有滞后），故默认关闭。
        const bool reuse_cand = param.obs_reuse_cand_delta > 0.0
                             && iter > 0
                             && !obs_surf_to_pts.empty()
                             && delta_scale <= param.obs_reuse_cand_delta;
        if (prof) t_seg.tic();
        if (!reuse_cand) {
            obs_surf_to_pts.clear();
            collectFromRangeImage(rp_near, match_min_z, 0);
            if (use_far) collectFromRangeImage(rp_far, far_match_min_z, 0);

            if (param.obs_cand_target > 0) {
                for (int off = 1; off < col_step && cappedTotal() < param.obs_cand_target; ++off) {
                    collectFromRangeImage(rp_near, match_min_z, off);
                    if (use_far) collectFromRangeImage(rp_far, far_match_min_z, off);
                }
            }

            if (surf_cap > 0) {
                for (auto& [sid, idxs] : obs_surf_to_pts) {
                    if ((int)idxs.size() > surf_cap) {
                        const int step = (int)idxs.size() / surf_cap;
                        std::vector<int> kept;
                        kept.reserve(surf_cap);
                        for (int i = 0; i < (int)idxs.size() && (int)kept.size() < surf_cap; i += step)
                            kept.push_back(idxs[i]);
                        idxs = std::move(kept);
                    }
                }
            }
        }

        if (prof) { ms_obs_collect += t_seg.toc(); t_seg.tic(); }

        std::vector<RegMatch> obs_matches;
        obs_matches.reserve(bspline_map.size() * param.obs_match_per_surf_max);
        stat_edge_kept = stat_edge_rej = stat_gate_rej = 0;
        buildObsMatches(scan_local, T_curr, cur_match_thr, /*want_ground=*/false,
                       bspline_map, obs_surf_to_pts, obs_prev_indices, obs_prev_uv,
                       stat_edge_kept, stat_edge_rej, stat_gate_rej, obs_matches);
        obs_last_res = (int)obs_matches.size();
        if (prof) { ms_obs_match += t_seg.toc(); t_seg.tic(); }

        // 左右平衡：一侧墙垄断时点到面残差把 yaw 往单边拽。雷达系 y>=0 为左。
        // 任一侧超过 max_frac 时，该侧保留 keep = frac/(1-frac)*对侧 个，按 scan_idx 等间隔取。
        if (param.obs_lr_max_frac > 0.0 && param.obs_lr_max_frac < 1.0
            && obs_matches.size() >= 10) {
            int nL = 0, nR = 0;
            for (const auto& m : obs_matches)
                (m.p_local.y() >= 0.0 ? nL : nR)++;
            const double f = param.obs_lr_max_frac;
            const int capL = nR > 0 ? (int)std::floor(f / (1.0 - f) * nR) : 0;
            const int capR = nL > 0 ? (int)std::floor(f / (1.0 - f) * nL) : 0;
            auto downsample = [&](bool left, int keep) {
                std::vector<int> idx;
                for (int i = 0; i < (int)obs_matches.size(); ++i)
                    if ((obs_matches[i].p_local.y() >= 0.0) == left) idx.push_back(i);
                if ((int)idx.size() <= keep) return;
                std::sort(idx.begin(), idx.end(), [&](int a, int b) {
                    return obs_matches[a].scan_idx < obs_matches[b].scan_idx;
                });
                const int n = (int)idx.size();
                std::vector<char> keep_flag(obs_matches.size(), 1);
                for (int i : idx) keep_flag[i] = 0;
                for (int k = 0; k < keep; ++k) keep_flag[idx[k * n / keep]] = 1;
                std::vector<RegMatch> out;
                out.reserve(obs_matches.size() - (n - keep));
                for (int i = 0; i < (int)obs_matches.size(); ++i)
                    if (keep_flag[i]) out.push_back(obs_matches[i]);
                obs_matches.swap(out);
            };
            if (nL > 0 && nR > 0) {
                if (nL > capL) downsample(true,  std::max(10, capL));
                if (nR > capR) downsample(false, std::max(10, capR));
            }
        }

        appendMatchDropEvent(g_data.step, iter, prev_match_count,
                             static_cast<int>(obs_matches.size()), 0,
                             static_cast<int>(obs_matches.size()));
        prev_match_count = static_cast<int>(obs_matches.size());

        if ((int)obs_matches.size() < 10) {
            ROS_WARN("Registration obs iter %d: only %d matches, skip",
                     iter, (int)obs_matches.size());
            break;
        }
        last_matches = obs_matches;

        double roll_fix = 0.0, pitch_fix = 0.0, yaw_init = 0.0;
        matrixToRPY(T_curr.block<3, 3>(0, 0), roll_fix, pitch_fix, yaw_init);
        const double z_fix = T_curr(2, 3);
        double xy_yaw[3] = {T_curr(0, 3), T_curr(1, 3), yaw_init};

        // 反方差加权：残差噪声 = 传感器测距噪声与该片自身拟合误差的正交和。等权时一张
        // 0.4 m 的植被片和一张 0.02 m 的墙面片对位姿影响相同，而实测障碍面拟合距离中位
        // 0.13 m、33% 超 0.15 m。归一到均值 1 是为了不改变 Huber(0.5) 的实际稳健尺度。
        if (param.obs_fit_weight_sigma > 0.0) {
            const double s2 = param.obs_fit_weight_sigma * param.obs_fit_weight_sigma;
            double sum_w = 0.0;
            for (auto& m : obs_matches) {
                const double fit = m.fit_dist >= 0.0 ? m.fit_dist : 0.0;
                m.weight = 1.0 / std::sqrt(s2 + fit * fit);
                sum_w += m.weight;
            }
            const double scale = obs_matches.size() / std::max(sum_w, 1e-12);
            for (auto& m : obs_matches) m.weight *= scale;
        } else {
            for (auto& m : obs_matches) m.weight = 1.0;
        }
        // yaw 的 J ∝ r，等权下远距匹配（拟合噪声 ~0.13 m）角误差大却主导航向。
        // 乘 min(1, ref/r) 后近处匹配主导；再归一到均值 1，Huber(0.5) 尺度不变。
        if (param.obs_range_ref > 0.0 && !obs_matches.empty()) {
            const double ref = param.obs_range_ref;
            double sum_w = 0.0;
            for (auto& m : obs_matches) {
                const double r = std::hypot(m.p_local.x(), m.p_local.y());
                m.weight *= ref / std::max(r, ref);
                sum_w += m.weight;
            }
            const double scale = obs_matches.size() / std::max(sum_w, 1e-12);
            for (auto& m : obs_matches) m.weight *= scale;
        }

        ceres::LossFunction* loss = new ceres::HuberLoss(
            param.obs_huber > 0.0 ? param.obs_huber : 0.5);
        ceres::Problem problem;
        for (auto& m : obs_matches) {
            problem.AddResidualBlock(
                ObstaclePointToSurfaceXYYaw::Create(
                    m.p_local, m.p_world, m.curvature,
                    roll_fix, pitch_fix, z_fix, m.drop_tangent, m.weight),
                loss, xy_yaw);
        }

        if (param.obs_prior_sigma_xy > 0.0 && param.obs_prior_along_only
            && g_data.step >= 2) {
            const Transf& T_p = g_data.T_seq[g_data.step - 1];
            const double dx = T_guess(0, 3) - T_p(0, 3);
            const double dy = T_guess(1, 3) - T_p(1, 3);
            const double len = std::hypot(dx, dy);
            if (len > 1e-3) {
                problem.AddResidualBlock(
                    MotionPriorAlongTrack::Create(
                        T_p(0, 3), T_p(1, 3), dx / len, dy / len, len,
                        1.0 / param.obs_prior_sigma_xy),
                    nullptr, xy_yaw);
            }
            if (param.obs_prior_sigma_yaw > 0.0) {
                double r_g, p_g, yaw_g;
                matrixToRPY(T_guess.block<3, 3>(0, 0), r_g, p_g, yaw_g);
                problem.AddResidualBlock(
                    MotionPriorYaw::Create(
                        yaw_g, 1.0 / (param.obs_prior_sigma_yaw * M_PI / 180.0)),
                    nullptr, xy_yaw);
            }
        } else if (param.obs_prior_sigma_xy > 0.0 && param.obs_prior_sigma_yaw > 0.0) {
            double r_g, p_g, yaw_g;
            matrixToRPY(T_guess.block<3, 3>(0, 0), r_g, p_g, yaw_g);
            problem.AddResidualBlock(
                MotionPriorXYYaw::Create(
                    T_guess(0, 3), T_guess(1, 3), yaw_g,
                    1.0 / param.obs_prior_sigma_xy,
                    1.0 / (param.obs_prior_sigma_yaw * M_PI / 180.0)),
                nullptr, xy_yaw);
        }

        ceres::Solver::Options opts;
        opts.linear_solver_type = ceres::DENSE_QR;
        opts.max_num_iterations = param.obs_ceres_iters > 0 ? param.obs_ceres_iters : 1;
        opts.minimizer_progress_to_stdout = false;
        // 多线程下残差块到线程的划分每次运行都变，每线程 scratch 的分部分和分组随之
        // 变化，总 cost 在 bit 级不同，trust-region 的接受判定就可能分叉。实测同配置
        // 重跑 seq02 摆幅达 0.8/2.6，A/B 无法测量。3 个参数 + DENSE_QR，单线程代价可忽略。
        opts.num_threads = 1;
        ceres::Solver::Summary summary;
        ceres::Solve(opts, &problem, &summary);
        if (prof) ms_obs_solve += t_seg.toc();

        const Transf T_new = makePoseFromXYZRPY(
            xy_yaw[0], xy_yaw[1], z_fix, roll_fix, pitch_fix, xy_yaw[2]);
        delta_scale = (T_new.block<3,1>(0,3) - T_curr.block<3,1>(0,3)).norm()
                    + 5.0 * (T_new.block<3,3>(0,0) - T_curr.block<3,3>(0,0)).norm();
        T_curr = T_new;
    }

    // 残差数塌陷时沿轨平移不可观测：存活对应方向高度集中（seq02 失效段 dom 0.42~0.54 对
    // 正常 0.27），解被地图粘在滞后位置上，步长逐帧衰减约 7%，二十帧后估计步长 0.055 m 而
    // 真值 1.41 m。衰减是正反馈：位姿滞后 -> 匀速外推继承已缩小的增量 -> 下一帧更滞后。
    // 只回退步长、保留解出的方向与 yaw：实测饿死帧上求解的转角误差 0.135°/帧优于外推的
    // 0.174°/帧（该段真值累计转角 41.65°，全靠外推会欠 17°），而步长用真值历史做匀速外推
    // 的预测误差仅 0.007 m。z/roll/pitch 归地面阶段，这里只动 xy。
    if (param.obs_starve_res_lo > 0
        && param.obs_starve_res_hi > param.obs_starve_res_lo
        && obs_last_res >= 0 && obs_last_res < param.obs_starve_res_hi
        && g_data.step > param.obs_starve_warmup) {
        const double span = param.obs_starve_res_hi - param.obs_starve_res_lo;
        double w = (obs_last_res - param.obs_starve_res_lo) / span;
        w = std::max(0.0, std::min(1.0, w));
        const Transf& T_prev = g_data.T_seq[g_data.step - 1];
        const Eigen::Vector2d d_solve(T_curr(0, 3) - T_prev(0, 3),
                                      T_curr(1, 3) - T_prev(1, 3));
        const Eigen::Vector2d d_pred(T_guess(0, 3) - T_prev(0, 3),
                                     T_guess(1, 3) - T_prev(1, 3));
        const double l_solve = d_solve.norm();
        const double l_target = w * l_solve + (1.0 - w) * d_pred.norm();
        if (l_solve > 1e-6) {
            const Eigen::Vector2d t_new =
                Eigen::Vector2d(T_prev(0, 3), T_prev(1, 3))
                + d_solve * (l_target / l_solve);
            T_curr(0, 3) = t_new.x();
            T_curr(1, 3) = t_new.y();
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // 阶段 2：地面匹配 — 点到平面；只优化世界系 roll / pitch / z
    // ══════════════════════════════════════════════════════════════════════
    {
        gnd_enter_indices.clear();
        last_gnd_matches.clear();
        cur_match_thr = match_dist_thr;  // 地面阶段自带 gnd_thr 调度，这里复位
        const std::vector<RegMatch> obs_saved = last_matches;
        const double x_fix = T_curr(0, 3);
        const double y_fix = T_curr(1, 3);
        double roll_i = 0.0, pitch_i = 0.0, yaw_fix = 0.0;
        matrixToRPY(T_curr.block<3, 3>(0, 0), roll_i, pitch_i, yaw_fix);

        constexpr int kGndIters = 3;
        const double gnd_thr_last = param.ground_thr_last;
        auto gndMatchDistForIter = [&](int it) -> double {
            const double kSched[3] = {1, 0.3, gnd_thr_last};
            return kSched[std::min(std::max(it, 0), kGndIters - 1)];
        };

        // XY 格子采样：分层 z + |y|；格内按 lx 分前/中/后，前向更密初抽并挂邻居；
        // 过 thr 后格扩容，再 FB 邻居补短板并裁超额。
        // 这批参数与地面判据在三轮迭代间恒定，提到循环外只算一次。
        const double reg_cs   = param.ground_reg_cell_size;
        const int    reg_cmax = param.ground_reg_cell_max_pts > 0
                                ? param.ground_reg_cell_max_pts : 30;
        const int    reg_cmax_f = (param.ground_reg_cell_max_pts_front > 0)
                                ? param.ground_reg_cell_max_pts_front : reg_cmax;
        const int    reg_tgt  = param.ground_reg_cell_target_pts;
        const int    nbr_cap  = param.ground_reg_nbr_per_seed;
        const int    seed_bud = param.ground_reg_total_max;
        const double y_max    = param.ground_reg_y_max;
        const double x_max    = param.ground_reg_x_max;
        const double x_max_f  = param.ground_reg_x_max_front;
        const double near_x   = param.gnd_near_x_max;
        const double near_y   = param.gnd_near_y_max;
        const double near_z   = param.gnd_near_z_max;
        const bool   use_near = (near_x > 0.0 && near_y > 0.0);
        const double fb_x0    = param.ground_reg_fb_x0;
        const double fb_front = param.ground_reg_fb_front;
        const double fb_mid   = param.ground_reg_fb_mid;
        auto zCeilAt = [&](double lx, double ly) -> double {
            if (use_near && std::abs(lx) <= near_x && std::abs(ly) <= near_y)
                return std::min(near_z, ground_z_max);  // 近区只许更严
            return ground_z_max;
        };
        // 地面判据：有 Patchwork++ mask 时以其为准，否则回退分层 z 带。
        // |y| 上限属于采样密度控制，两条路径都保留。
        const bool use_pw = !pw_ground_mask.empty();
        auto passGndLocal = [&](int idx, double lx, double ly, double lz) -> bool {
            if (y_max > 0.0 && std::abs(ly) > y_max) return false;
            if (x_max > 0.0 && std::abs(lx) > x_max) return false;
            if (x_max_f > 0.0 && lx > x_max_f) return false;
            if (use_pw) return pw_ground_mask[idx] != 0;
            if (lz < ground_z_min || lz > zCeilAt(lx, ly)) return false;
            return true;
        };

        // 判据只看雷达系坐标与 mask，三轮迭代结果完全相同；原实现每轮重扫全云
        // （HDL-64 约 12 万点）。下标仍是升序，分桶的插入顺序不变。
        std::vector<int> gnd_cand_idx;
        if (reg_cs > 0.0) {
            gnd_cand_idx.reserve(scan_local.size() / 4);
            for (int i = 0; i < (int)scan_local.size(); ++i)
                if (passGndLocal(i, scan_local[i].x, scan_local[i].y, scan_local[i].z))
                    gnd_cand_idx.push_back(i);
        }

        for (int giter = 0; giter < kGndIters; ++giter) {
            if (prof) t_seg.tic();
            const double gnd_thr = gndMatchDistForIter(giter);
            std::unordered_map<const BSplineSurface*, std::vector<int>> gnd_surf_to_pts;

            // seed -> 同格未抽中邻居；scan_idx -> 所属世界 XY 格 key（扩容统计用）
            std::unordered_map<int, std::vector<int>> seed_neighbors;
            std::unordered_map<int, std::int64_t>     pt_cell_key;
            const Eigen::Matrix3d R_curr = T_curr.block<3,3>(0,0);
            const Eigen::Vector3d t_curr = T_curr.block<3,1>(0,3);

            if (reg_cs > 0.0) {
                std::unordered_map<std::int64_t, std::vector<int>> xy_bucket;
                for (int i : gnd_cand_idx) {
                    const Eigen::Vector3d p_w = R_curr *
                        Eigen::Vector3d(scan_local[i].x, scan_local[i].y, scan_local[i].z)
                        + t_curr;
                    const std::int64_t cx = static_cast<std::int64_t>(std::floor(p_w.x() / reg_cs));
                    const std::int64_t cy = static_cast<std::int64_t>(std::floor(p_w.y() / reg_cs));
                    const std::int64_t key = cx * 1000003LL + cy;
                    xy_bucket[key].push_back(i);
                    pt_cell_key[i] = key;
                }
                // 放宽横向门后占用格数成倍增长，逐格配额不变会让查询量随之膨胀。
                // 按占用格数把每格配额等比压到全局预算内：覆盖范围照常扩大，单帧代价有界。
                int cmax_eff = reg_cmax, cmax_f_eff = reg_cmax_f;
                if (seed_bud > 0 && !xy_bucket.empty()) {
                    const double per_cell_cfg = reg_cmax_f + 2.0 * reg_cmax;
                    const double per_cell_bud =
                        static_cast<double>(seed_bud) / static_cast<double>(xy_bucket.size());
                    if (per_cell_bud < per_cell_cfg) {
                        const double s = per_cell_bud / per_cell_cfg;
                        cmax_eff   = std::max(1, (int)std::lround(reg_cmax   * s));
                        cmax_f_eff = std::max(1, (int)std::lround(reg_cmax_f * s));
                    }
                }

                // 每个种子都要走一次 queryNearest，而 queryNearest 会对邻域内每张
                // 候选面做一次冷启动 findFootPrint（PCA 投影 + 6 步 Newton），是地面
                // 采样的主要开销。按格并行：每格产物先落自己的桶，再按 xy_bucket 的
                // 自然迭代序归并，各曲面下的种子顺序与串行版相同。
                // findFootPrint/coldInitUVFromPCA 都只读曲面数据，同一张面被多格
                // 并发查询也安全。
                std::vector<const std::vector<int>*> cells;
                cells.reserve(xy_bucket.size());
                for (auto& [key, indices] : xy_bucket) {
                    (void)key;
                    cells.push_back(&indices);
                }

                struct CellOut {
                    std::vector<std::pair<const BSplineSurface*, int>> hits;  // (最近面, 种子)
                    std::vector<std::pair<int, std::vector<int>>>      nbrs;  // (种子, 同格邻居)
                };
                std::vector<CellOut> cell_out(cells.size());
                const int n_cell = (int)cells.size();
                const int n_thr  = std::max(1, param.num_thread_reg);

                // 每格内按雷达系 lx 分前/中/后，前向用更密 max_pts，再挂邻居
                auto sample_group = [&](const std::vector<int>& indices, int cmax, CellOut& co) {
                    const int n = (int)indices.size();
                    if (n <= 0 || cmax <= 0) return;
                    const int step = std::max(1, n / cmax);
                    for (int k = 0; k < n; k += step) {
                        const int seed = indices[k];
                        const Eigen::Vector3d p_w = R_curr *
                            Eigen::Vector3d(scan_local[seed].x, scan_local[seed].y,
                                           scan_local[seed].z) + t_curr;
                        const BSplineSurface* sp = mr_ground.queryNearest(p_w);
                        if (sp) co.hits.emplace_back(sp, seed);

                        const int k_end = std::min(n, k + step);
                        std::vector<int> nbrs;
                        nbrs.reserve(static_cast<size_t>(k_end - k));
                        for (int j = k + 1; j < k_end; ++j)
                            nbrs.push_back(indices[j]);
                        if (nbr_cap > 0 && (int)nbrs.size() > nbr_cap) {
                            std::vector<int> kept;
                            kept.reserve(static_cast<size_t>(nbr_cap));
                            for (int t = 0; t < nbr_cap; ++t) {
                                const int pos = (t * (int)nbrs.size()) / nbr_cap;
                                kept.push_back(nbrs[pos]);
                            }
                            nbrs.swap(kept);
                        }
                        if (!nbrs.empty())
                            co.nbrs.emplace_back(seed, std::move(nbrs));
                    }
                };

#pragma omp parallel for schedule(dynamic) num_threads(n_thr)
                for (int ci = 0; ci < n_cell; ++ci) {
                    const std::vector<int>& indices = *cells[ci];
                    CellOut& co = cell_out[ci];
                    std::vector<int> idx_f, idx_m, idx_b;
                    idx_f.reserve(indices.size());
                    idx_m.reserve(indices.size());
                    idx_b.reserve(indices.size());
                    for (int i : indices) {
                        const double lx = scan_local[i].x;
                        if (lx >= fb_x0)       idx_f.push_back(i);
                        else if (lx <= -fb_x0) idx_b.push_back(i);
                        else                   idx_m.push_back(i);
                    }
                    sample_group(idx_f, cmax_f_eff, co);
                    sample_group(idx_m, cmax_eff,   co);
                    sample_group(idx_b, cmax_eff,   co);
                }

                for (CellOut& co : cell_out) {
                    for (const auto& [sp, seed] : co.hits)
                        gnd_surf_to_pts[sp].push_back(seed);
                    for (auto& [seed, nb] : co.nbrs)
                        seed_neighbors[seed] = std::move(nb);
                }
            } else {
                // 退化：旧 skip 模式（无邻居扩容）
                const int gnd_skip = param.ground_skip_points > 0 ? param.ground_skip_points : skip_points;
                for (int i = 0; i < (int)scan_local.size(); i += gnd_skip) {
                    const double lx = scan_local[i].x, ly = scan_local[i].y, lz = scan_local[i].z;
                    if (!passGndLocal(i, lx, ly, lz)) continue;
                    Eigen::Vector3d p_w = R_curr * Eigen::Vector3d(lx, ly, lz) + t_curr;
                    const BSplineSurface* sp = mr_ground.queryNearest(p_w);
                    if (sp) gnd_surf_to_pts[sp].push_back(i);
                }
            }

            // 记录本轮进入地面匹配（已找到最近曲面）的点
            gnd_enter_indices.clear();
            for (const auto& [sp, indices] : gnd_surf_to_pts) {
                (void)sp;
                gnd_enter_indices.insert(gnd_enter_indices.end(), indices.begin(), indices.end());
            }

            if (prof) { ms_gnd_sample += t_seg.toc(); t_seg.tic(); }

            std::vector<RegMatch> gnd_matches;
            buildGroundMatches(scan_local, T_curr, gnd_thr, gnd_surf_to_pts,
                              gnd_prev_indices, gnd_prev_uv, gnd_matches);

            // 补点扩容只影响本轮解的加密程度，而前两轮的解都会被后续轮次覆盖；
            // 这一段每点要走 6 步 Newton，是地面链最贵的部分。默认仍逐轮扩容。
            const bool do_expand = !param.gnd_expand_last_iter_only
                                || giter == kGndIters - 1;

            // 过 thr 后：每格不足 target 时，用该格内成功种子挂的邻居填充，再裁到 target
            if (do_expand && reg_cs > 0.0 && reg_tgt > 0 && !seed_neighbors.empty()) {
                std::unordered_map<std::int64_t, std::vector<int>> cell_ok_seeds;
                std::unordered_map<std::int64_t, int> cell_match_cnt;
                std::unordered_set<int> matched_idx;
                matched_idx.reserve(gnd_matches.size() * 2);
                for (const auto& m : gnd_matches) {
                    matched_idx.insert(m.scan_idx);
                    auto it = pt_cell_key.find(m.scan_idx);
                    if (it == pt_cell_key.end()) continue;
                    ++cell_match_cnt[it->second];
                    // 仅成功匹配且本身是种子（有邻居表）的点可作扩容锚点
                    if (seed_neighbors.count(m.scan_idx))
                        cell_ok_seeds[it->second].push_back(m.scan_idx);
                }

                std::unordered_map<const BSplineSurface*, std::vector<int>> expand_surf_to_pts;
                for (auto& [ckey, seeds] : cell_ok_seeds) {
                    const int have = cell_match_cnt[ckey];
                    if (have >= reg_tgt) continue;
                    const int need = reg_tgt - have;

                    // 汇集该格成功种子的邻居，均匀穿插取 need 个未用点
                    std::vector<int> pool;
                    for (int s : seeds) {
                        auto nit = seed_neighbors.find(s);
                        if (nit == seed_neighbors.end()) continue;
                        for (int ni : nit->second) {
                            if (!matched_idx.count(ni))
                                pool.push_back(ni);
                        }
                    }
                    if (pool.empty()) continue;

                    const int take = std::min(need, (int)pool.size());
                    const int step = std::max(1, (int)pool.size() / take);
                    int added = 0;
                    for (int j = 0; j < (int)pool.size() && added < take; j += step) {
                        const int ni = pool[j];
                        if (matched_idx.count(ni)) continue;
                        const Eigen::Vector3d p_w = R_curr *
                            Eigen::Vector3d(scan_local[ni].x, scan_local[ni].y,
                                           scan_local[ni].z) + t_curr;
                        const BSplineSurface* sp = mr_ground.queryNearest(p_w);
                        if (!sp) continue;
                        expand_surf_to_pts[sp].push_back(ni);
                        matched_idx.insert(ni);
                        ++added;
                    }
                }

                if (!expand_surf_to_pts.empty()) {
                    buildGroundMatches(scan_local, T_curr, gnd_thr, expand_surf_to_pts,
                                      gnd_prev_indices, gnd_prev_uv, gnd_matches);
                    // 每格最多保留 reg_tgt（种子优先，因其排在前面）
                    std::unordered_map<std::int64_t, int> kept;
                    std::vector<RegMatch> trimmed;
                    trimmed.reserve(gnd_matches.size());
                    for (const auto& m : gnd_matches) {
                        auto it = pt_cell_key.find(m.scan_idx);
                        const std::int64_t key = (it != pt_cell_key.end()) ? it->second : 0;
                        if (kept[key] >= reg_tgt) continue;
                        ++kept[key];
                        trimmed.push_back(m);
                    }
                    gnd_matches.swap(trimmed);
                }
            }

            // 前/后/中配额：不足用该桶成功种子的邻居补（不限格子 target），再裁超额
            if (fb_front > 0.0 && !gnd_matches.empty()) {
                auto bucket_of_lx = [&](double lx) -> int {
                    if (lx >= fb_x0) return 0;       // front
                    if (lx <= -fb_x0) return 2;      // back
                    return 1;                        // mid
                };

                const int total0 = (int)gnd_matches.size();
                // 桶配额以封顶后的总量为基准：超预算时三桶按同比例缩，
                // 下面 trim_bucket 在桶内等间隔抽，前后力臂分布不受影响
                const int tgt_base = (param.ground_reg_match_max > 0)
                    ? std::min(total0, param.ground_reg_match_max)
                    : total0;
                int tgt[3] = {
                    (int)std::round(tgt_base * fb_front),
                    (int)std::round(tgt_base * fb_mid),
                    0
                };
                tgt[2] = tgt_base - tgt[0] - tgt[1];

                std::unordered_set<int> matched_idx;
                matched_idx.reserve(gnd_matches.size() * 2);
                std::vector<int> ok_seeds[3];
                int have[3] = {0, 0, 0};
                for (const auto& m : gnd_matches) {
                    matched_idx.insert(m.scan_idx);
                    const int b = bucket_of_lx(m.p_local.x());
                    ++have[b];
                    if (seed_neighbors.count(m.scan_idx))
                        ok_seeds[b].push_back(m.scan_idx);
                }

                // 短板桶：从成功种子邻居补点（邻居 lx 仍须落在同桶）
                std::unordered_map<const BSplineSurface*, std::vector<int>> fb_expand;
                for (int b = 0; do_expand && b < 3; ++b) {
                    if (have[b] >= tgt[b] || ok_seeds[b].empty()) continue;
                    const int need = tgt[b] - have[b];
                    std::vector<int> pool;
                    for (int s : ok_seeds[b]) {
                        auto nit = seed_neighbors.find(s);
                        if (nit == seed_neighbors.end()) continue;
                        for (int ni : nit->second) {
                            if (matched_idx.count(ni)) continue;
                            if (bucket_of_lx(scan_local[ni].x) != b) continue;
                            pool.push_back(ni);
                        }
                    }
                    if (pool.empty()) continue;
                    const int take = std::min(need, (int)pool.size());
                    const int step = std::max(1, (int)pool.size() / take);
                    int added = 0;
                    for (int j = 0; j < (int)pool.size() && added < take; j += step) {
                        const int ni = pool[j];
                        if (matched_idx.count(ni)) continue;
                        const Eigen::Vector3d p_w = R_curr *
                            Eigen::Vector3d(scan_local[ni].x, scan_local[ni].y,
                                           scan_local[ni].z) + t_curr;
                        const BSplineSurface* sp = mr_ground.queryNearest(p_w);
                        if (!sp) continue;
                        fb_expand[sp].push_back(ni);
                        matched_idx.insert(ni);
                        ++added;
                    }
                }
                if (!fb_expand.empty())
                    buildGroundMatches(scan_local, T_curr, gnd_thr, fb_expand,
                                      gnd_prev_indices, gnd_prev_uv, gnd_matches);

                // 重新分桶并裁到 tgt（超额裁，不足全留）
                const int total = (int)gnd_matches.size();
                std::vector<int> idx_bkt[3];
                for (int i = 0; i < total; ++i)
                    idx_bkt[bucket_of_lx(gnd_matches[i].p_local.x())].push_back(i);

                auto trim_bucket = [](std::vector<int>& idx, int t) {
                    if (t <= 0 || (int)idx.size() <= t) return;
                    const int n = (int)idx.size();
                    const int step = std::max(1, n / t);
                    std::vector<int> kept;
                    kept.reserve(t);
                    for (int j = 0; j < n && (int)kept.size() < t; j += step)
                        kept.push_back(idx[j]);
                    idx.swap(kept);
                };
                for (int b = 0; b < 3; ++b)
                    trim_bucket(idx_bkt[b], std::max(0, tgt[b]));

                std::unordered_set<int> keep_idx;
                keep_idx.reserve(idx_bkt[0].size() + idx_bkt[1].size() + idx_bkt[2].size());
                for (int b = 0; b < 3; ++b)
                    for (int i : idx_bkt[b]) keep_idx.insert(i);

                std::vector<RegMatch> balanced;
                balanced.reserve(keep_idx.size());
                for (int i = 0; i < total; ++i)
                    if (keep_idx.count(i)) balanced.push_back(gnd_matches[i]);
                gnd_matches.swap(balanced);
            }

            if (prof) { ms_gnd_match += t_seg.toc(); t_seg.tic(); }

            if ((int)gnd_matches.size() < 10) break;

            last_gnd_matches = gnd_matches;

            double rpy_z[3] = {roll_i, pitch_i, T_curr(2, 3)};
            ceres::Problem problem;
            ceres::LossFunction* loss = new ceres::HuberLoss(0.5);
            for (const auto& m : gnd_matches) {
                problem.AddResidualBlock(
                    GroundPointToPlaneZRP::Create(
                        m.p_local, m.curvature.point, m.curvature.normal,
                        x_fix, y_fix, yaw_fix, 1.0),
                    loss, rpy_z);
            }

            if (g_data.step >= 2
                && (param.gnd_roll_prior_sigma > 0.0
                    || param.gnd_pitch_prior_sigma > 0.0)) {
                double r_p, p_p, y_p;
                matrixToRPY(g_data.T_seq[g_data.step - 1].block<3, 3>(0, 0),
                            r_p, p_p, y_p);
                const double kDeg = M_PI / 180.0;
                if (param.gnd_roll_prior_sigma > 0.0)
                    problem.AddResidualBlock(
                        AttitudeHoldPrior::Create(
                            0, r_p, 1.0 / (param.gnd_roll_prior_sigma * kDeg)),
                        nullptr, rpy_z);
                if (param.gnd_pitch_prior_sigma > 0.0)
                    problem.AddResidualBlock(
                        AttitudeHoldPrior::Create(
                            1, p_p, 1.0 / (param.gnd_pitch_prior_sigma * kDeg)),
                        nullptr, rpy_z);
            }

            ceres::Solver::Options opts;
            opts.linear_solver_type = ceres::DENSE_QR;
            opts.max_num_iterations = 10;
            opts.minimizer_progress_to_stdout = false;
            opts.num_threads = 1;  // 同上：可复现优先
            ceres::Solver::Summary summary;
            ceres::Solve(opts, &problem, &summary);
            if (prof) ms_gnd_solve += t_seg.toc();

            T_curr = makePoseFromXYZRPY(
                x_fix, y_fix, rpy_z[2], rpy_z[0], rpy_z[1], yaw_fix);
            roll_i = rpy_z[0];
            pitch_i = rpy_z[1];

            last_matches = obs_saved;
            last_matches.insert(last_matches.end(), gnd_matches.begin(), gnd_matches.end());
        }
    }
    }  // 外层交替

    const Eigen::Matrix3d R_dump = T_curr.block<3, 3>(0, 0);
    const Eigen::Vector3d t_dump = T_curr.block<3, 1>(0, 3);

    // 指定区间：每帧写 gnd_scan/（过 thr 匹配）+ whole_gnd/（z 带初分类地面点）+ scan_world/（全量世界系点）
    if (param.dump_gnd_scan_begin > 0
        && param.dump_gnd_scan_end >= param.dump_gnd_scan_begin
        && g_data.step >= param.dump_gnd_scan_begin
        && g_data.step <= param.dump_gnd_scan_end) {
        const double y_max_dump = param.ground_reg_y_max;
        auto dumpWorldPts = [&](const std::filesystem::path& dir,
                                const std::string& prefix,
                                const auto& write_pts,
                                const char* tag) {
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            if (ec) {
                std::cerr << "  [" << tag << "] mkdir failed: " << dir
                          << " (" << ec.message() << ")\n";
                return;
            }
            const std::string out =
                (dir / (prefix + std::to_string(g_data.step) + ".txt")).string();
            std::ofstream f(out, std::ios::out | std::ios::trunc);
            f << std::fixed << std::setprecision(6);
            write_pts(f);
            f.close();
        };

        dumpWorldPts(
            std::filesystem::path(kBsplineBuildDir) / "gnd_scan",
            "gnd_scan_",
            [&](std::ofstream& f) {
                for (const auto& m : last_gnd_matches) {
                    const Eigen::Vector3d p_w = R_dump * m.p_local + t_dump;
                    f << p_w.x() << " " << p_w.y() << " " << p_w.z() << "\n";
                }
                return (int)last_gnd_matches.size();
            },
            "DumpGndScan");

        // whole_gnd：与配准相同的地面判据 + |y|<=ground_reg_y_max
        dumpWorldPts(
            std::filesystem::path(kBsplineBuildDir) / "whole_gnd",
            "whole_gnd_",
            [&](std::ofstream& f) {
                const double near_x = param.gnd_near_x_max;
                const double near_y = param.gnd_near_y_max;
                const double near_z = param.gnd_near_z_max;
                const bool   use_near = (near_x > 0.0 && near_y > 0.0);
                const bool   use_pw_dump = !pw_ground_mask.empty();
                auto zCeilAt = [&](double lx, double ly) -> double {
                    if (use_near && std::abs(lx) <= near_x && std::abs(ly) <= near_y)
                        return std::min(near_z, ground_z_max);
                    return ground_z_max;
                };
                int n = 0;
                for (int i = 0; i < (int)scan_local.size(); ++i) {
                    const double lx = scan_local[i].x, ly = scan_local[i].y, lz = scan_local[i].z;
                    if (use_pw_dump) { if (!pw_ground_mask[i]) continue; }
                    else if (lz < ground_z_min || lz > zCeilAt(lx, ly)) continue;
                    if (y_max_dump > 0.0 && std::abs(ly) > y_max_dump) continue;
                    const Eigen::Vector3d p_w = R_dump *
                        Eigen::Vector3d(lx, ly, lz) + t_dump;
                    f << p_w.x() << " " << p_w.y() << " " << p_w.z() << "\n";
                    ++n;
                }
                return n;
            },
            "DumpWholeGnd");

        dumpWorldPts(
            std::filesystem::path(kBsplineBuildDir) / "scan_world",
            "scan_world_",
            [&](std::ofstream& f) {
                for (int i = 0; i < (int)scan_local.size(); ++i) {
                    const Eigen::Vector3d p_w = R_dump *
                        Eigen::Vector3d(scan_local[i].x, scan_local[i].y, scan_local[i].z) + t_dump;
                    f << p_w.x() << " " << p_w.y() << " " << p_w.z() << "\n";
                }
                return (int)scan_local.size();
            },
            "DumpScanWorld");
    }

    if (param.dump_frame > 0 && g_data.step == param.dump_frame) {
        const std::string base = kBsplineBuildDir;
        std::error_code ec;
        std::filesystem::create_directories(base, ec);
        if (ec) {
            std::cerr << "  [Dump] failed to create dir: " << base << " (" << ec.message() << ")\n";
        } else {
            const Eigen::Matrix3d& R = R_dump;
            const Eigen::Vector3d& t = t_dump;

            std::ofstream f_origin(base + "/origin_scan.txt", std::ios::out | std::ios::trunc);
            std::ofstream f_scan(base + "/scan_world.txt", std::ios::out | std::ios::trunc);
            std::ofstream f_scan_gnd(base + "/scan_gnd.txt", std::ios::out | std::ios::trunc);
            std::ofstream f_self_gnd(base + "/self_gnd.txt", std::ios::out | std::ios::trunc);
            std::ofstream f_self_obs(base + "/self_obs.txt", std::ios::out | std::ios::trunc);
            f_origin << std::fixed << std::setprecision(6);
            f_scan << std::fixed << std::setprecision(6);
            f_scan_gnd << std::fixed << std::setprecision(6);
            f_self_gnd << std::fixed << std::setprecision(6);
            f_self_obs << std::fixed << std::setprecision(6);

            // scan_gnd / self_gnd：最后一轮进入地面匹配（送入曲面查询）的点
            std::unordered_set<int> gnd_enter_set(
                gnd_enter_indices.begin(), gnd_enter_indices.end());
            for (int idx : gnd_enter_indices) {
                const auto& pt = scan_local.points[idx];
                const Eigen::Vector3d p_l(pt.x, pt.y, pt.z);
                const Eigen::Vector3d p_w = R * p_l + t;
                f_scan_gnd << p_w.x() << " " << p_w.y() << " " << p_w.z() << "\n";
                f_self_gnd << p_l.x() << " " << p_l.y() << " " << p_l.z() << "\n";
            }
            for (int i = 0; i < (int)scan_local.size(); ++i) {
                const auto& pt = scan_local.points[i];
                const Eigen::Vector3d p_l(pt.x, pt.y, pt.z);
                const Eigen::Vector3d p_w = R * p_l + t;
                f_origin << p_w.x() << " " << p_w.y() << " " << p_w.z() << "\n";
                if (gnd_enter_set.count(i)) continue;
                f_scan << p_w.x() << " " << p_w.y() << " " << p_w.z() << "\n";
                f_self_obs << p_l.x() << " " << p_l.y() << " " << p_l.z() << "\n";
            }
            f_origin.close();
            f_scan.close();
            f_scan_gnd.close();
            f_self_gnd.close();
            f_self_obs.close();

            std::ofstream f_matched(base + "/matched.txt", std::ios::out | std::ios::trunc);
            std::ofstream f_matched_gnd(base + "/matched_gnd.txt", std::ios::out | std::ios::trunc);
            f_matched << std::fixed << std::setprecision(6);
            f_matched_gnd << std::fixed << std::setprecision(6);
            for (const auto& m : last_matches) {
                f_matched << m.p_world.x() << " " << m.p_world.y() << " " << m.p_world.z() << "\n";
                if (m.is_ground)
                    f_matched_gnd << m.p_world.x() << " " << m.p_world.y() << " " << m.p_world.z() << "\n";
            }
            f_matched.close();
            f_matched_gnd.close();

            std::unordered_set<int> matched_set;
            for (const auto& m : last_matches) matched_set.insert(m.scan_idx);

            int n_unmatched = 0;
            std::ofstream f_unmatched(base + "/unmatched.txt", std::ios::out | std::ios::trunc);
            f_unmatched << std::fixed << std::setprecision(6);
            for (int i = 0; i < (int)scan_local.size(); i += skip_points) {
                const double z = scan_local[i].z;
                const bool is_gnd = (z >= ground_z_min && z <= ground_z_max);
                const bool is_obs = (z >= match_min_z);
                if (!is_gnd && !is_obs) continue;
                if (matched_set.count(i)) continue;
                const Eigen::Vector3d p_w = R * Eigen::Vector3d(scan_local[i].x, scan_local[i].y, scan_local[i].z) + t;
                f_unmatched << p_w.x() << " " << p_w.y() << " " << p_w.z() << "\n";
                ++n_unmatched;
            }
            f_unmatched.close();
            (void)n_unmatched;
        }
    }

    if (prof) {
        std::cout << "  [reg] obs collect/match/solve = "
                  << ms_obs_collect << "/" << ms_obs_match << "/" << ms_obs_solve
                  << " ms | gnd sample/match/solve = "
                  << ms_gnd_sample << "/" << ms_gnd_match << "/" << ms_gnd_solve
                  << " ms" << std::endl;
    }
    return T_curr;
}

void SLAMesher::runMapBuild(const pcl::PointCloud<pcl::PointXYZ>& scan_local,
                     const std::vector<uint8_t>& pw_ground_mask,
                     const Transf& T_world,
                     RangeImageProcessor& range_proc,
                     RangeImageProcessor& range_proc_far,
                     RangeImageProcessor& range_proc_gnd,
                     BSplineMap& bspline_map,
                     MultiResGroundMap& mr_ground,
                     double match_dist_thr,
                     double ground_z_min,
                     double ground_z_max)
{
    constexpr int    MIN_CLUSTER_PTS  = 30;
    constexpr int    MAX_NEW_SURFACES = 70;

    const bool dump_clusters = (param.dump_cluster_step > 0 && g_data.step == param.dump_cluster_step);
    const bool dump_occluded = (param.dump_occluded_step > 0 && g_data.step == param.dump_occluded_step);
    const bool dump_far_funnel = (param.dump_frame > 0 && g_data.step == param.dump_frame);
    const bool do_obstacle = (g_data.step == 1) || (g_data.step % param.map_update_interval  == 0);
    const bool do_ground   = (g_data.step == 1) || (g_data.step % param.ground_build_interval == 0);
    if (!do_obstacle && !do_ground && !dump_clusters && !dump_occluded && !dump_far_funnel) return;

    TicToc t_upd;
    const double split_dist = param.range_image_split;
    const bool use_far_layer = (split_dist > 0.0);
    const double far_z_floor = farLayerZFloor(ground_z_min);
    const Eigen::Matrix3d R_w = T_world.block<3,3>(0,0);
    const Eigen::Vector3d t_w = T_world.block<3,1>(0,3);

    int n_added_gnd = 0, n_added_obs = 0;
    const bool profile_frame1_apply = (g_data.step == 1);
    if (profile_frame1_apply) {
        BSplineSurface::setApplyProfileLogPath(
            std::string(kBsplineBuildDir) + "/frame1_apply_profile.txt");
        std::cout << "  [Frame1Apply] profiling frame-1 map build apply -> "
                  << kBsplineBuildDir << "/frame1_apply_profile.txt" << std::endl;
    }

    // ── 1) 地面 RI 列提取 → 多层 XY 建图 ──
    // 远区 z ∈ [gnd_ri_z_min, gnd_ri_z_max]；近区（|x|<=near_x 且 |y|<=near_y）
    // 用更严 gnd_near_z_max。列底→上：|Δz|>col_dz 断链不断列；相对 seed 上抬受限。
    // z>gnd_ri_z_max 整列硬停；近区超 near_z_max 只跳过该点，不硬停远区。
    const double gnd_z_floor  = param.gnd_ri_z_min;
    const double gnd_z_ceil   = param.gnd_ri_z_max;
    const double near_x_max   = param.gnd_near_x_max;
    const double near_y_max   = param.gnd_near_y_max;
    const double near_z_max   = param.gnd_near_z_max;
    const double col_dz_max   = param.gnd_col_max_step;
    const double seed_h_up    = param.gnd_col_seed_h_up;
    const int    N             = static_cast<int>(scan_local.size());
    const bool   use_near_band = (near_x_max > 0.0 && near_y_max > 0.0);

    auto zCeilAt = [&](double x, double y) -> double {
        if (use_near_band &&
            std::abs(x) <= near_x_max && std::abs(y) <= near_y_max)
            return std::min(near_z_max, gnd_z_ceil);  // 近区只许更严，不许更松
        return gnd_z_ceil;
    };

    // ground_mask[i]：scan_local[i] 是否为地面候选
    std::vector<bool> ground_mask(N, false);
    const bool use_pw_mask = !pw_ground_mask.empty();

    if (use_pw_mask) {
        for (int i = 0; i < N; ++i)
            ground_mask[i] = (i < (int)pw_ground_mask.size() && pw_ground_mask[i] != 0);
    } else {
        // 先建地面专用 RI（不排除任何 z 带，让列传播自己决定）
        range_proc_gnd.generateRangeImage(scan_local,
            -1e9, 1e9, /*exclude_ground_band=*/false,
            0.0, 1e9, gnd_z_floor);

        // 列底→上传播：H_SCANS 行 row=0 最低仰角（FOV_DOWN），row=H_SCANS-1 最高
        for (int v = 0; v < range_proc_gnd.W_COLS; ++v) {
            double last_z = std::numeric_limits<double>::quiet_NaN();
            double seed_z = std::numeric_limits<double>::quiet_NaN();
            for (int u = 0; u < range_proc_gnd.H_SCANS; ++u) {
                const int pidx = u * range_proc_gnd.W_COLS + v;
                const auto& px = range_proc_gnd.range_image_[pidx];
                if (!px.valid) continue;
                if (px.z > gnd_z_ceil) break;  // 远区硬顶，整列停止

                // 近区更严：超 near_z_max 不当地面，不跟爬，但继续向上找远区点
                if (px.z > zCeilAt(px.x, px.y)) {
                    last_z = std::numeric_limits<double>::quiet_NaN();
                    continue;
                }

                const bool have_seed = !std::isnan(seed_z);
                const bool within_seed =
                    !have_seed || (px.z <= seed_z + seed_h_up);

                if (std::isnan(last_z)) {
                    // 新种子（含断链后）：必须仍贴近本列 seed 高度，禁止抬高再开段
                    if (!within_seed) continue;
                    seed_z = have_seed ? seed_z : px.z;
                    last_z = px.z;
                    const int cidx = range_proc_gnd.pixel_to_cloud_idx_[pidx];
                    if (cidx >= 0 && cidx < N) ground_mask[cidx] = true;
                } else if (std::abs(px.z - last_z) <= col_dz_max && within_seed) {
                    last_z = px.z;
                    const int cidx = range_proc_gnd.pixel_to_cloud_idx_[pidx];
                    if (cidx >= 0 && cidx < N) ground_mask[cidx] = true;
                } else if (std::abs(px.z - last_z) > col_dz_max) {
                    // 台阶断链：清 last_z，保留 seed_z；抬高点不可再种子
                    last_z = std::numeric_limits<double>::quiet_NaN();
                }
                // else: 连续但高于 seed+h_up → 跳过，保持 last_z（不跟爬）
            }
        }
    }

    if (do_ground) {
        // 地面世界系点云（先投 XY 格，格内按密度降采样）；再按近/远 z 上界复核
        pcl::PointCloud<pcl::PointXYZ> cloud_gnd_world;
        cloud_gnd_world.reserve(static_cast<size_t>(N / 4));
        std::vector<Eigen::Vector3d> gnd_lidar_pts;
        gnd_lidar_pts.reserve(static_cast<size_t>(N / 4));
        const double map_r_max = param.ground_map_r_max;
        for (int i = 0; i < N; ++i) {
            if (!ground_mask[i]) continue;
            const auto& pt = scan_local.points[i];
            // 超距点不进图，但仍算地面：置 false 会让它漏进 scan_obs 污染障碍匹配
            if (map_r_max > 0.0 &&
                (double)pt.x * pt.x + (double)pt.y * pt.y > map_r_max * map_r_max)
                continue;
            // Patchwork++ 的判定已是最终结论，不再叠加 z 带复核
            if (!use_pw_mask && pt.z > zCeilAt(pt.x, pt.y)) {
                ground_mask[i] = false;  // 不进建图，也不从障碍里抠掉
                continue;
            }
            gnd_lidar_pts.emplace_back(pt.x, pt.y, pt.z);
            const Eigen::Vector3d pw = R_w * Eigen::Vector3d(pt.x, pt.y, pt.z) + t_w;
            cloud_gnd_world.push_back(pcl::PointXYZ(
                static_cast<float>(pw.x()), static_cast<float>(pw.y()), static_cast<float>(pw.z())));
        }

        if (g_data.step == 1)
            saveFrame1GroundPoints(gnd_lidar_pts);

        if (!cloud_gnd_world.empty()) {
            mr_ground.addPoints(cloud_gnd_world, g_data.step);
            if (profile_frame1_apply)
                BSplineSurface::setApplyProfileLabel("gnd");
            n_added_gnd = mr_ground.refitAll(param.num_thread);
        }
        // dump 区间内：每次建完地面，写累计 all_surfaces_<step>.txt（含此前建图）
        dumpGndAllSurfacesAtBuild(mr_ground);
    }

    if (dump_clusters) {
        saveClusterFilterStatsTxt(scan_local, g_data.step, ground_z_min, ground_z_max,
                                  split_dist, range_proc, range_proc_far);
    }

    // ── 2) 障碍 range image：用 ground_mask 排除地面点 ──
    // 构建去掉地面点的 scan_obs，避免地面点污染障碍 cluster
    pcl::PointCloud<pcl::PointXYZ> scan_obs;
    scan_obs.reserve(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) {
        if (!ground_mask[i]) scan_obs.push_back(scan_local[i]);
    }

    // 遮挡点只有 dump 要用，平时不收集（KITTI 单帧碰撞点数与点云同量级）
    range_proc.setCollectOccluded(dump_occluded);
    range_proc_far.setCollectOccluded(dump_occluded);

    range_proc.generateRangeImage(scan_obs,
        -1e9, 1e9, /*exclude_ground_band=*/false,
        0.0, use_far_layer ? split_dist : 1e9, ground_z_min);
    if (use_far_layer)
        range_proc_far.generateRangeImage(scan_obs,
            -1e9, 1e9, /*exclude_ground_band=*/false,
            split_dist, 1e9, far_z_floor);

    if (dump_occluded) {
        const std::string build_dir = std::string(kBsplineBuildDir);
        std::error_code ec;
        std::filesystem::create_directories(build_dir, ec);
        if (ec) {
            std::cerr << "  [DumpOccluded] failed to create dir: " << build_dir
                      << " (" << ec.message() << ")\n";
        } else {
            // 近层
            const auto& occ = range_proc.getOccludedPoints();
            saveOccludedPointsTxt(occ, T_world, build_dir);
            const int n_in = saveRangeImagePointsTxt(range_proc, T_world, build_dir);
            // 远层
            int n_in_far = 0;
            if (use_far_layer) {
                const auto& occ_far = range_proc_far.getOccludedPoints();
                // 保存到 occluded_far_* 和 rangeimage_far_*
                std::ofstream fl(build_dir + "/occluded_far_lidar.txt", std::ios::trunc);
                std::ofstream fw(build_dir + "/occluded_far_world.txt", std::ios::trunc);
                const Eigen::Matrix3d R = T_world.block<3,3>(0,0);
                const Eigen::Vector3d t = T_world.block<3,1>(0,3);
                fl << std::fixed << std::setprecision(6);
                fw << std::fixed << std::setprecision(6);
                for (const auto& p : occ_far) {
                    fl << p.x() << " " << p.y() << " " << p.z() << "\n";
                    Eigen::Vector3d pw = R * p + t;
                    fw << pw.x() << " " << pw.y() << " " << pw.z() << "\n";
                }
                // rangeimage_far_*
                std::ofstream frl(build_dir + "/rangeimage_far_lidar.txt", std::ios::trunc);
                std::ofstream frw(build_dir + "/rangeimage_far_world.txt", std::ios::trunc);
                frl << std::fixed << std::setprecision(6);
                frw << std::fixed << std::setprecision(6);
                for (const auto& px : range_proc_far.range_image_) {
                    if (!px.valid) continue;
                    frl << px.x << " " << px.y << " " << px.z << "\n";
                    Eigen::Vector3d pw = R * Eigen::Vector3d(px.x, px.y, px.z) + t;
                    frw << pw.x() << " " << pw.y() << " " << pw.z() << "\n";
                    ++n_in_far;
                }
            }
            std::cout << "  [DumpOccluded] step=" << g_data.step
                      << " near: occluded=" << occ.size() << " rangeimage=" << n_in
                      << (use_far_layer ? " far: occluded=" + std::to_string(range_proc_far.getOccludedPoints().size())
                                              + " rangeimage=" + std::to_string(n_in_far) : "")
                      << " -> " << build_dir << std::endl;
        }
    }

    // 地面点已在投影阶段排除；分割侧 exclude=true 作双保险
    SegmentationResult seg = range_proc.segmentRangeImage(
        5, 0.1, MIN_CLUSTER_PTS, ground_z_min, ground_z_max, true,
        param.obs_seg_normal_deg);
    range_proc.downsampleClusters(
        seg,
        param.cluster_ds_min_pts,
        param.cluster_ds_target_max,
        MIN_CLUSTER_PTS);

    // 远层分割（如果启用）
    SegmentationResult seg_far;
    SegmentationResult seg_far_pre_ds;
    if (use_far_layer) {
        const int far_min = param.range_image_far_min_cluster;
        seg_far = range_proc_far.segmentRangeImage(
            5, 0.1, far_min, ground_z_min, ground_z_max, true,
            param.obs_seg_normal_deg);
        if (dump_far_funnel)
            seg_far_pre_ds = seg_far;
        range_proc_far.downsampleClusters(
            seg_far,
            param.cluster_ds_min_pts,
            param.cluster_ds_target_max,
            far_min);
        if (dump_far_funnel) {
            saveFarClusterPipelineAuditTxt(scan_local, g_data.step, T_world,
                                           ground_z_min, ground_z_max, far_z_floor,
                                           split_dist, far_min,
                                           range_proc_far, seg_far_pre_ds, seg_far);
        }
    }

    // ── 障碍分支：range image 仅含非地面点 ──
    if (do_obstacle && profile_frame1_apply)
        BSplineSurface::setApplyProfileLabel("obs");

    // 障碍建图 task（用于 OMP 并行 apply）
    struct ObsBuildTask {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_world;
        std::vector<Eigen::Vector3d> init_cp;
        int num_cp = 0;
        std::shared_ptr<BSplineSurface> surf;
        bool ok = false;
    };

    std::vector<double> obs_fit_dists;
    // 障碍建图 lambda：A 收集 → B 并行 apply → C 串行入库
    auto buildObsFromSeg = [&](RangeImageProcessor& proc, SegmentationResult& s, int min_pts) {
        // ── 阶段 A：并行筛 cluster + footprint 匹配，按 cid 定位写入 ──
        // 每个 cluster 的判定只读 bspline_map / range_image，互不依赖；唯一的跨 cluster
        // 耦合是 MAX_NEW_SURFACES 截断只看已接受的个数，所以把截断挪到后面的串行收集，
        // 结果与串行版逐位一致（串行版也是按 cid 升序接受到额满为止）。
        // findFootPrint 只读曲面成员，写全落在调用方的 uv_state/point_dists 上，同一张
        // 曲面被多个 cluster 并发查询是安全的。
        std::vector<ObsBuildTask> tasks;
        // do_obstacle=false 时 n_loop=0，整段跳过：等价于串行版在首个够大的 cluster 处 break
        const int n_loop = do_obstacle ? (int)s.clusters.size() : 0;
        std::vector<ObsBuildTask> per_cid(n_loop);
        std::vector<char> cid_ready(n_loop, 0);
        const int n_omp_a = std::max(1, param.num_thread);
#pragma omp parallel for schedule(dynamic) num_threads(n_omp_a)
        for (int cid = 0; cid < n_loop; cid++) {
            const auto& pixels = s.clusters[cid];
            if ((int)pixels.size() < min_pts) continue;

            std::unordered_map<int, std::vector<int>> surf_to_kidx;
            std::vector<Eigen::Vector3d> pix_world(pixels.size(), Eigen::Vector3d::Zero());
            std::vector<bool> pix_valid(pixels.size(), false);

            // ── PCA 法向检测：平坦水平面（地面类）→ 跳过障碍建图 ──
            {
                Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
                int n_pca = 0;
                for (int k = 0; k < (int)pixels.size(); ++k) {
                    const auto& px = proc.range_image_[pixels[k]];
                    if (!px.valid) continue;
                    centroid += Eigen::Vector3d(px.x, px.y, px.z);
                    ++n_pca;
                }
                if (n_pca < min_pts) continue;
                centroid /= n_pca;
                Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
                for (int k = 0; k < (int)pixels.size(); ++k) {
                    const auto& px = proc.range_image_[pixels[k]];
                    if (!px.valid) continue;
                    Eigen::Vector3d d = Eigen::Vector3d(px.x, px.y, px.z) - centroid;
                    cov += d * d.transpose();
                }
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
                // 特征值升序，最小特征值对应的特征向量即法向
                const Eigen::Vector3d cluster_normal = solver.eigenvectors().col(0);
                if (std::abs(cluster_normal.z()) > 0.85) continue;
            }

            for (int k = 0; k < (int)pixels.size(); ++k) {
                const auto& px = proc.range_image_[pixels[k]];
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

            int n_valid = static_cast<int>(std::count(pix_valid.begin(), pix_valid.end(), true));
            if (n_valid < min_pts) continue;

            auto cloud_local = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
            cloud_local->reserve(pixels.size());
            for (int k = 0; k < (int)pixels.size(); ++k) {
                if (!pix_valid[k] || confirmed_matched[k]) continue;
                const auto& px = proc.range_image_[pixels[k]];
                cloud_local->push_back(pcl::PointXYZ(px.x, px.y, px.z));
            }
            const int n_unmatched = static_cast<int>(cloud_local->size());
            if (n_unmatched < min_pts) continue;
            if (static_cast<double>(n_unmatched) / n_valid < param.map_unmatched_ratio_min) continue;

            const int num_cp = chooseControlGridSize(n_unmatched, param.obs_fit_pts_per_cp);
            auto cloud_world = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
            pcl::transformPointCloud(*cloud_local, *cloud_world, T_world.cast<float>());

            auto init_cp = proc.computeInitControlPoints(s, cid, num_cp, num_cp, 2);
            if (init_cp.empty()) continue;
            for (auto& cp : init_cp) cp = R_w * cp + t_w;

            ObsBuildTask t;
            t.cloud_world = cloud_world;
            t.init_cp     = std::move(init_cp);
            t.num_cp      = num_cp;
            per_cid[cid]  = std::move(t);
            cid_ready[cid] = 1;
        }

        // 按 cid 升序收集，额满即停：与串行版的接受集合完全相同
        for (int cid = 0; cid < (int)cid_ready.size(); ++cid) {
            if (!cid_ready[cid]) continue;
            if (g_data.step != 1 && (n_added_obs + (int)tasks.size()) >= MAX_NEW_SURFACES) break;
            tasks.push_back(std::move(per_cid[cid]));
        }

        // ── 阶段 B：OMP 并行 apply（每个 task 独立，无共享写） ──
        const int n_omp = std::max(1, param.num_thread);
        // Frame1Apply profile 在并行区不可用（全局变量），先暂停
        const bool had_profile = profile_frame1_apply && do_obstacle;
        if (had_profile) BSplineSurface::setApplyProfileLogPath("");
#pragma omp parallel for schedule(dynamic) num_threads(n_omp)
        for (int i = 0; i < (int)tasks.size(); ++i) {
            auto& t = tasks[i];
            t.surf = std::make_shared<BSplineSurface>(3, 3, t.num_cp, t.num_cp);
            t.surf->setExternalInitControls(t.init_cp);
            t.ok = t.surf->apply(t.cloud_world, param.obs_fit_max_iter, 1, 1,
                                 param.obs_fit_eps);
        }

        // ── 阶段 C：串行入库 ──
        int n_fit_rej = 0, n_split_ok = 0, n_split_keep = 0;
        auto addObsSurf = [&](std::shared_ptr<BSplineSurface> surf,
                              pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
            bspline_map.addSurface(surf, cloud, /*is_ground=*/false, g_data.step,
                                   param.obs_map_replace_frac, param.obs_map_voxel_owner);
            obs_fit_dists.push_back(surf->getFitMeanDist());
            ++n_added_obs;
        };
        auto fitDistOf = [&](const std::shared_ptr<BSplineSurface>& surf) {
            return param.obs_fit_weight_adj ? surf->getFitMeanDistAdj()
                                            : surf->getFitMeanDist();
        };
        for (auto& t : tasks) {
            if (!t.ok) continue;
            const double fd = fitDistOf(t.surf);
            const int npts = (int)t.cloud_world->size();
            bool replaced = false;
            if (param.obs_fit_split_dist > 0.0 && fd > param.obs_fit_split_dist && npts >= 60) {
                Eigen::Vector3d c = Eigen::Vector3d::Zero();
                for (const auto& p : t.cloud_world->points)
                    c += Eigen::Vector3d(p.x, p.y, p.z);
                c /= npts;
                Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
                for (const auto& p : t.cloud_world->points) {
                    Eigen::Vector3d d = Eigen::Vector3d(p.x, p.y, p.z) - c;
                    cov += d * d.transpose();
                }
                const Eigen::Vector3d axis =
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(cov).eigenvectors().col(2);
                std::vector<double> proj(npts);
                for (int i = 0; i < npts; ++i) {
                    const auto& p = t.cloud_world->points[i];
                    proj[i] = (Eigen::Vector3d(p.x, p.y, p.z) - c).dot(axis);
                }
                std::vector<double> proj_s = proj;
                std::nth_element(proj_s.begin(), proj_s.begin() + npts / 2, proj_s.end());
                const double mid = proj_s[npts / 2];
                auto c0 = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
                auto c1 = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
                for (int i = 0; i < npts; ++i)
                    (proj[i] < mid ? c0 : c1)->push_back(t.cloud_world->points[i]);
                if ((int)c0->size() >= 30 && (int)c1->size() >= 30) {
                    auto fitChild = [&](pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
                        -> std::shared_ptr<BSplineSurface> {
                        const int ncp = chooseControlGridSize((int)cloud->size(),
                                                              param.obs_fit_pts_per_cp);
                        auto s = std::make_shared<BSplineSurface>(3, 3, ncp, ncp);
                        if (!s->apply(cloud, param.obs_fit_max_iter, 1, 1, param.obs_fit_eps))
                            return nullptr;
                        return s;
                    };
                    auto s0 = fitChild(c0);
                    auto s1 = fitChild(c1);
                    if (s0 && s1 && fitDistOf(s0) < fd && fitDistOf(s1) < fd) {
                        addObsSurf(s0, c0);
                        addObsSurf(s1, c1);
                        ++n_split_ok;
                        replaced = true;
                    } else {
                        ++n_split_keep;
                    }
                } else {
                    ++n_split_keep;
                }
            }
            if (replaced) continue;
            if (param.obs_fit_max_dist > 0.0 && fd > param.obs_fit_max_dist) {
                ++n_fit_rej;
                continue;
            }
            addObsSurf(t.surf, t.cloud_world);
        }
        if (n_fit_rej > 0)
            std::cout << "  [obs fit-rej] n=" << n_fit_rej
                      << " thr=" << param.obs_fit_max_dist << std::endl;
        if (n_split_ok + n_split_keep > 0)
            std::cout << "  [obs split] ok=" << n_split_ok
                      << " keep=" << n_split_keep
                      << " thr=" << param.obs_fit_split_dist << std::endl;
    };

    // 近层建图
    buildObsFromSeg(range_proc, seg, MIN_CLUSTER_PTS);
    // 远层建图（如启用）
    if (use_far_layer)
        buildObsFromSeg(range_proc_far, seg_far, param.range_image_far_min_cluster);

    if (profile_frame1_apply) {
        BSplineSurface::clearApplyProfileLog();
    }

    if (dump_clusters) {
        const std::string cluster_dir = std::string(kBsplineBuildDir) + "/output_clusters";
        std::error_code ec;
        std::filesystem::create_directories(cluster_dir, ec);
        if (ec) {
            std::cerr << "  [DumpCluster] failed to create dir: " << cluster_dir
                      << " (" << ec.message() << ")\n";
        } else {
            range_proc.saveClustersWorldToTxt(seg, cluster_dir, T_world);
            int n_far_clusters = 0;
            if (use_far_layer) {
                // 远层 cluster 文件名加 _far 前缀，写入同目录
                const std::string far_dir = std::string(kBsplineBuildDir) + "/output_clusters_far";
                std::error_code ec2;
                std::filesystem::create_directories(far_dir, ec2);
                if (!ec2) {
                    range_proc_far.saveClustersWorldToTxt(seg_far, far_dir, T_world);
                    n_far_clusters = (int)seg_far.clusters.size();
                }
            }
            std::cout << "  [DumpCluster] step=" << g_data.step
                      << " near_clusters=" << seg.clusters.size()
                      << (use_far_layer ? " far_clusters=" + std::to_string(n_far_clusters) : "")
                      << " -> " << cluster_dir << "/ (world frame)" << std::endl;
        }
    }

    std::cout << "  [MapBuild] step=" << g_data.step
              << " near_clusters=" << seg.clusters.size()
              << (use_far_layer ? " far_clusters=" + std::to_string(seg_far.clusters.size()) : "")
              << " +gnd_cells_refit=" << n_added_gnd << " +obs=" << n_added_obs
              << " obs_total=" << bspline_map.size()
              << " alive=" << bspline_map.aliveSurfaceCount()
              << " retired=" << bspline_map.retiredCount()
              << " (" << t_upd.toc() << " ms)" << std::endl;

    // 新入库障碍面自身的平均点到面距离分布（米）。传感器测距噪声约 0.02 m，
    // 超出部分是拟合/几何误差，直接进配准残差。
    if (!obs_fit_dists.empty()) {
        std::sort(obs_fit_dists.begin(), obs_fit_dists.end());
        const int n = (int)obs_fit_dists.size();
        const double sum = std::accumulate(obs_fit_dists.begin(), obs_fit_dists.end(), 0.0);
        int n_over_5cm = 0, n_over_15cm = 0;
        for (double d : obs_fit_dists) {
            if (d > 0.05) ++n_over_5cm;
            if (d > 0.15) ++n_over_15cm;
        }
        char fbuf[220];
        snprintf(fbuf, sizeof(fbuf),
                 "  [obs fit] n=%d mean=%.4f p50=%.4f p90=%.4f max=%.4f"
                 " over5cm=%.0f%% over15cm=%.0f%%",
                 n, sum / n, obs_fit_dists[n / 2], obs_fit_dists[(int)(n * 0.9)],
                 obs_fit_dists[n - 1], 100.0 * n_over_5cm / n, 100.0 * n_over_15cm / n);
        std::cout << fbuf << std::endl;
    }
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

void SLAMesher::saveGroundGridZ(const MultiResGroundMap& mr_ground) const
{
    const std::string out_path = std::string(kBsplineBuildDir) + "/ground_grid_z.txt";
    std::ofstream fout(out_path);
    if (!fout.is_open()) return;

    fout << std::fixed << std::setprecision(5);

    for (int li = 0; li < mr_ground.numLayers(); ++li) {
        const GroundGridMap& gm = mr_ground.layer(li);
        std::vector<GroundCellKey> keys;
        keys.reserve(gm.cells().size());
        for (const auto& [k, cell] : gm.cells()) {
            if (cell.surf) keys.push_back(k);
        }
        std::sort(keys.begin(), keys.end(), [](const GroundCellKey& a, const GroundCellKey& b) {
            return a.ix < b.ix || (a.ix == b.ix && a.iy < b.iy);
        });

        for (const auto& k : keys) {
            const auto& cell = gm.cells().at(k);
            const auto& cps = cell.surf->getControls();
            if (cps.empty()) continue;
            double z_sum = 0;
            for (const auto& cp : cps) z_sum += cp.z();
            // 格式：layer ix iy z_mean
            fout << li << " " << k.ix << " " << k.iy
                 << " " << (z_sum / static_cast<double>(cps.size())) << "\n";
        }
    }
}

void SLAMesher::saveGndSurfacesToTxt(const MultiResGroundMap& mr_ground,
                                     const std::string& out_path_in) const
{
    const std::string out_path = out_path_in.empty()
        ? (std::string(kBsplineBuildDir) + "/gnd_surfaces.txt")
        : out_path_in;
    {
        std::error_code ec;
        const auto parent = std::filesystem::path(out_path).parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent, ec);
            if (ec) {
                std::cerr << "Failed to create directory: " << parent
                          << " (" << ec.message() << ")\n";
                return;
            }
        }
    }
    std::ofstream fout(out_path, std::ios::out | std::ios::trunc);
    if (!fout.is_open()) {
        std::cerr << "Failed to open: " << out_path << std::endl;
        return;
    }
    fout << std::fixed << std::setprecision(6);

    constexpr int kSampleGridU = 8;
    constexpr int kSampleGridV = 8;
    auto findSpan = [](double t, const std::vector<double>& knots, int num_cp) {
        if (t >= 1.0 - 1e-9) return num_cp - 1;
        for (int k = 3; k < num_cp; ++k)
            if (knots[k] <= t && t < knots[k + 1]) return k;
        return 3;
    };

    int n_surf = 0, n_pts = 0;
    for (int li = 0; li < mr_ground.numLayers(); ++li) {
        for (const auto& [key, cell] : mr_ground.layer(li).cells()) {
            (void)key;
            if (!cell.surf) continue;
            const auto& surf = cell.surf;
            const auto& knU = surf->getKnotsU();
            const auto& knV = surf->getKnotsV();
            const auto& cps = surf->getControls();
            const int num_cpv = surf->getNumCpV();
            const int num_cpu = surf->getNumCpU();
            for (int iu = 0; iu < kSampleGridU; ++iu) {
                const double u = (kSampleGridU > 1) ? double(iu) / double(kSampleGridU - 1) : 0.0;
                const BSplineSurface::Parameter paraU(findSpan(u, knU, num_cpu), u);
                for (int iv = 0; iv < kSampleGridV; ++iv) {
                    const double v = (kSampleGridV > 1) ? double(iv) / double(kSampleGridV - 1) : 0.0;
                    const BSplineSurface::Parameter paraV(findSpan(v, knV, num_cpv), v);
                    const Eigen::Vector3d p = surf->getPos(paraU, paraV, knU, knV, cps, num_cpv);
                    fout << p.x() << " " << p.y() << " " << p.z() << "\n";
                    ++n_pts;
                }
            }
            ++n_surf;
        }
    }
    fout.close();
    std::cout << "Saved gnd surfaces: " << n_surf << " surfaces ("
              << mr_ground.numLayers() << " layers), "
              << n_pts << " sample points -> " << out_path << std::endl;
}

void SLAMesher::saveGroundControlsToTxt(const MultiResGroundMap& mr_ground) const
{
    const std::string out_dir = std::string(kBsplineBuildDir) + "/gnd_controls";
    std::error_code ec;
    std::filesystem::remove_all(out_dir, ec);
    std::filesystem::create_directories(out_dir, ec);
    if (ec) {
        std::cerr << "Failed to create directory: " << out_dir
                  << " (" << ec.message() << ")\n";
        return;
    }

    int n_saved = 0;
    for (int li = 0; li < mr_ground.numLayers(); ++li) {
        for (const auto& [key, cell] : mr_ground.layer(li).cells()) {
            if (!cell.surf) continue;
            const auto& cps = cell.surf->getControls();
            if (cps.empty()) continue;
            const std::string out_path = out_dir + "/"
                + std::to_string(li) + "_"
                + std::to_string(key.ix) + "_"
                + std::to_string(key.iy) + ".txt";
            std::ofstream out(out_path, std::ios::out | std::ios::trunc);
            if (!out) {
                std::cerr << "Failed to open: " << out_path << std::endl;
                continue;
            }
            out << std::fixed << std::setprecision(6);
            out << cell.surf->getNumCpU() << " " << cell.surf->getNumCpV() << "\n";
            for (const auto& cp : cps)
                out << cp.x() << " " << cp.y() << " " << cp.z() << "\n";
            ++n_saved;
        }
    }
    std::cout << "Ground control points saved: " << n_saved
              << " files in " << out_dir << std::endl;
}

void SLAMesher::dumpGndAllSurfacesAtBuild(const MultiResGroundMap& mr_ground) const
{
    if (param.dump_gnd_scan_begin <= 0
        || param.dump_gnd_scan_end < param.dump_gnd_scan_begin
        || g_data.step < param.dump_gnd_scan_begin
        || g_data.step > param.dump_gnd_scan_end) {
        return;
    }
    const std::filesystem::path dir =
        std::filesystem::path(kBsplineBuildDir) / "all_surfaces";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        std::cerr << "  [DumpGndAllSurfaces] mkdir failed: " << dir
                  << " (" << ec.message() << ")\n";
        return;
    }
    // 累计快照：当前地面地图全量采样（含此前各次建图结果）
    const std::string out =
        (dir / ("all_surfaces_" + std::to_string(g_data.step) + ".txt")).string();
    saveGndSurfacesToTxt(mr_ground, out);
}

void SLAMesher::saveControlPointsToTxt(const BSplineMap& bspline_map,
                                       bool save_surface_samples,
                                       int step_begin,
                                       int step_end) const
{
    const std::string build_dir = kBsplineBuildDir;
    constexpr int kSampleGridU = 8;
    constexpr int kSampleGridV = 8;

    const bool filter_by_step = (step_begin > 0 || step_end > 0);
    if (filter_by_step) save_surface_samples = true;

    auto inStepRange = [&](int created_step) {
        if (!filter_by_step) return true;
        if (step_begin > 0 && created_step < step_begin) return false;
        if (step_end   > 0 && created_step > step_end)   return false;
        return true;
    };

    std::string all_sample_path = build_dir + "/all_surfaces.txt";

    auto findSpan = [](double t, const std::vector<double>& knots, int num_cp) {
        if (t >= 1.0 - 1e-9) return num_cp - 1;
        for (int k = 3; k < num_cp; ++k)
            if (knots[k] <= t && t < knots[k + 1]) return k;
        return 3;
    };

    const std::string out_dir    = build_dir + "/controls";
    const std::string sample_dir = build_dir + "/global_map";

    std::error_code ec;
    if (!filter_by_step) {
        std::filesystem::create_directories(out_dir, ec);
        if (ec) {
            std::cerr << "Failed to create directory: " << out_dir << " (" << ec.message() << ")\n";
            return;
        }
        // 上一轮 dump 的 sid 文件不会被覆盖，渲染时会变成远离轨迹的幽灵面
        std::vector<std::filesystem::path> stale;
        for (const auto& ent : std::filesystem::directory_iterator(out_dir, ec)) {
            if (ec) break;
            if (ent.is_regular_file() && ent.path().extension() == ".txt")
                stale.push_back(ent.path());
        }
        for (const auto& p : stale)
            std::filesystem::remove(p, ec);
    }
    if (save_surface_samples && !filter_by_step) {
        std::filesystem::create_directories(sample_dir, ec);
        if (ec) {
            std::cerr << "Failed to create directory: " << sample_dir << " (" << ec.message() << ")\n";
            return;
        }
    }

    std::ofstream all_sample;
    bool write_all_samples = false;
    if (save_surface_samples) {
        all_sample.open(all_sample_path, std::ios::out | std::ios::trunc);
        if (!all_sample.is_open()) {
            std::cerr << "Failed to open: " << all_sample_path << std::endl;
            return;
        }
        all_sample << std::fixed << std::setprecision(6);
        write_all_samples = true;
    }

    int n_saved = 0;
    int n_samples_saved = 0;
    int n_all_sample_pts = 0;
    int n_filtered_surfaces = 0;
    for (int sid = 0; sid < bspline_map.size(); ++sid) {
        const BSplineMapEntry* e = bspline_map.getEntry(sid);
        if (!e || !e->surface) continue;
        if (!inStepRange(e->created_step)) continue;
        ++n_filtered_surfaces;

        if (!filter_by_step) {
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
        }

        if (!save_surface_samples) continue;

        const auto& surf = e->surface;
        const auto& knU = surf->getKnotsU();
        const auto& knV = surf->getKnotsV();
        const auto& cps = surf->getControls();
        const int num_cpv = surf->getNumCpV();
        const int num_cpu = surf->getNumCpU();

        if (!filter_by_step) {
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
                    if (write_all_samples) {
                        all_sample << p.x() << " " << p.y() << " " << p.z() << "\n";
                        ++n_all_sample_pts;
                    }
                }
            }
        } else {
            for (int iu = 0; iu < kSampleGridU; ++iu) {
                const double u = (kSampleGridU > 1) ? double(iu) / double(kSampleGridU - 1) : 0.0;
                const BSplineSurface::Parameter paraU(findSpan(u, knU, num_cpu), u);
                for (int iv = 0; iv < kSampleGridV; ++iv) {
                    const double v = (kSampleGridV > 1) ? double(iv) / double(kSampleGridV - 1) : 0.0;
                    const BSplineSurface::Parameter paraV(findSpan(v, knV, num_cpv), v);
                    const Eigen::Vector3d p = surf->getPos(paraU, paraV, knU, knV, cps, num_cpv);
                    all_sample << p.x() << " " << p.y() << " " << p.z() << "\n";
                    ++n_all_sample_pts;
                }
            }
        }
        ++n_samples_saved;
    }

    if (write_all_samples) all_sample.close();

    if (filter_by_step) {
        std::cout << "Map surfaces in step [" << step_begin << ", " << step_end << "]: "
                  << n_filtered_surfaces << " surfaces, "
                  << n_all_sample_pts << " sample points -> " << all_sample_path << std::endl;
        return;
    }

    std::cout << "B-spline control points saved: " << n_saved
              << " files in " << out_dir << std::endl;
    if (save_surface_samples) {
        std::cout << "B-spline surface samples saved: " << n_samples_saved
                  << " files in " << sample_dir << std::endl;
        std::cout << "All surface samples merged: " << n_all_sample_pts
                  << " -> " << all_sample_path << std::endl;
    }
}

void SLAMesher::process(){
    TicToc t_whole;

    // ========== 全局地图 & 工具初始化 ==========
    BSplineMap bspline_map(param.grid);
    MultiResGroundMap mr_ground({
        // 层 0：细格（精细，近处为主）
        {param.ground_cell_size,
         param.ground_cell_min_pts,
         param.ground_cell_num_cp,
         param.ground_query_radius,
         param.ground_cell_max_pts,
         param.ground_fit_max_pts,
         param.ground_cell_z_pct,
         param.ground_cell_z_tol,
         param.gnd_normal_z_min,
         param.ground_cell_z_filter},
        // 层 1：粗格（远处/稀疏区 fallback）
        {param.ground_coarse_cell_size,
         param.ground_coarse_min_pts,
         param.ground_coarse_num_cp,
         param.ground_coarse_query_radius,
         param.ground_coarse_cell_max_pts,
         param.ground_coarse_fit_max_pts,
         param.ground_cell_z_pct,
         param.ground_cell_z_tol,
         param.gnd_normal_z_min,
         param.ground_cell_z_filter}
    });
    RangeImageProcessor range_proc;
    RangeImageProcessor range_proc_far;  // 远层（range >= range_image_split）
    RangeImageProcessor range_proc_gnd;  // 地面专用 RI（宽 z 带，列底→上传播）
    range_proc.configure(param.ri_h_scans, param.ri_w_cols,
                         (float)param.ri_fov_up, (float)param.ri_fov_down);
    range_proc_far.configure(param.ri_h_scans, param.ri_w_cols,
                             (float)param.ri_fov_up, (float)param.ri_fov_down);
    range_proc_gnd.configure(param.ri_h_scans, param.ri_w_cols,
                             (float)param.ri_fov_up, (float)param.ri_fov_down);

    g_data.extendLog();
    Transf T_world = g_data.initFirstTransf();
    g_data.updatePose(T_world);

    // 配准参数
    const int    max_rg_iters   = param.register_times;  // 外层迭代次数
    const double converge_thr   = param.converge_thr;
    const double match_dist_thr = 0.8;  // 点到曲面最大容许距离 (m)
    const int    skip_points    = 20;    // 每隔几个点取一个用于配准 (降低计算量)
    static std::ofstream traj_file;
    if (!traj_file.is_open()) {
        traj_file.open(kBsplineBuildDir + "/bspline_traj_xyz.txt",
                       std::ios::out | std::ios::trunc);
        traj_file << std::fixed << std::setprecision(6);
    }

    // 地面点 z 带：传感器系，建图/配准/range-image 排除共用（见 yaml ground_z_min/max）
    const double GROUND_Z_MIN = param.ground_z_min;
    const double GROUND_Z_MAX = param.ground_z_max;

    patchwork_.reset(param.patchwork);
    std::vector<uint8_t> pw_ground_mask;  // 关闭 Patchwork++ 时保持为空，两条链回退原逻辑
    std::vector<bool>    obs_exclude_mask; // 配准侧障碍 range image 的地面排除位（复用缓冲）
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
        if (param.lidar_flip_yz) {
            for (auto& p : scan_local.points) {
                p.y = -p.y;
                p.z = -p.z;
            }
        }
        std::cout << "===STEP " << g_data.step << "=== points: " << scan_local.size() << std::endl;

        // 地面分割每帧只跑一次，建图链与配准链共用同一份 mask
        pw_ground_mask.clear();
        if (param.use_patchwork_ground) {
            TicToc t_pw;
            patchwork_.estimateGround(scan_local, pw_ground_mask);
            const int n_gnd = (int)std::count(pw_ground_mask.begin(), pw_ground_mask.end(),
                                              uint8_t(1));
            std::cout << "  [Patchwork++] ground=" << n_gnd << "/" << scan_local.size()
                      << "  sensor_h=" << patchwork_.sensorHeight()
                      << "  " << t_pw.toc() << " ms" << std::endl;
        }

        // 竖直角标定修正：必须在建 range image / 配准 / 建图之前。原地改写，
        // 下标不变，pw_ground_mask 仍然对齐。ground_only 时只动地面点，障碍链
        // 拿到的仍是原始几何（runMapBuild 按 !ground_mask 取障碍点）。
        if (param.scan_correct_deg != 0.0) {
            const bool gnd_only = param.scan_correct_ground_only
                               && !pw_ground_mask.empty();
            correctScanCalib(scan_local, param.scan_correct_deg * M_PI / 180.0,
                             gnd_only ? &pw_ground_mask : nullptr);
        }

        if(g_data.step == 1){
            processFirstFrame(T_world);
            if (traj_file.is_open()) {
                traj_file << T_world(0,3) << " "
                          << T_world(1,3) << " "
                          << T_world(2,3) << "\n";
                traj_file.flush();
            }
            runMapBuild(scan_local, pw_ground_mask, T_world, range_proc, range_proc_far, range_proc_gnd, bspline_map, mr_ground, match_dist_thr, GROUND_Z_MIN, GROUND_Z_MAX);
            continue;
        }

        TicToc t_register;
        Transf T_guess = getOdom();

        // 为配准生成 range image：排除地面点，仅剩余点进障碍链。
        // 有 Patchwork++ mask 时按 mask 排除，与 runMapBuild 建障碍面所用的点集
        // （!ground_mask）逐点一致；否则回退 z 带排除（原行为）。
        {
            const double split = param.range_image_split;
            const bool use_far = (split > 0.0);
            const double far_z_floor = farLayerZFloor(GROUND_Z_MIN);
            const bool use_mask = param.obs_reg_use_pw_mask && !pw_ground_mask.empty();
            if (use_mask) {
                obs_exclude_mask.assign(scan_local.size(), false);
                for (size_t i = 0; i < obs_exclude_mask.size(); ++i)
                    obs_exclude_mask[i] = (i < pw_ground_mask.size() && pw_ground_mask[i] != 0);
            }
            const std::vector<bool>* excl = use_mask ? &obs_exclude_mask : nullptr;
            range_proc.generateRangeImage(scan_local, GROUND_Z_MIN, GROUND_Z_MAX, true,
                                          0.0, use_far ? split : 1e9, GROUND_Z_MIN, excl);
            if (use_far)
                range_proc_far.generateRangeImage(scan_local, GROUND_Z_MIN, GROUND_Z_MAX, true,
                                                  split, 1e9, far_z_floor, excl);
        }

        T_world = registerScanToMap(scan_local, pw_ground_mask, T_guess, bspline_map, mr_ground,
                                    max_rg_iters, converge_thr, match_dist_thr, skip_points,
                                    GROUND_Z_MIN, GROUND_Z_MIN, GROUND_Z_MAX,
                                    range_proc, range_proc_far, (param.range_image_split > 0.0));

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

        runMapBuild(scan_local, pw_ground_mask, T_world, range_proc, range_proc_far, range_proc_gnd, bspline_map, mr_ground, match_dist_thr, GROUND_Z_MIN, GROUND_Z_MAX);

        path_pub.publish(g_data.path);
        std::cout << "===STEP " << g_data.step << "=== Total: " << t_step.toc() << " ms===" << std::endl;
        std::cout.flush();
    }

    std::cout << "Process finished. Total time: " << t_whole.toc() / 1000.0 << " s" << std::endl;

    printMapSummary(bspline_map);
    saveGroundGridZ(mr_ground);
    saveGroundControlsToTxt(mr_ground);
    if (param.save_surface_samples || param.all_surfaces_max_step > 0
        || param.map_save_step_begin > 0 || param.map_save_step_end > 0) {
        saveGndSurfacesToTxt(mr_ground);
    }
    if (param.all_surfaces_max_step > 0) {
        saveControlPointsToTxt(bspline_map, true, 0, param.all_surfaces_max_step);
    } else if (param.map_save_step_begin > 0 || param.map_save_step_end > 0) {
        saveControlPointsToTxt(bspline_map, true,
                               param.map_save_step_begin, param.map_save_step_end);
    } else {
        saveControlPointsToTxt(bspline_map, param.save_surface_samples);
    }
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
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);
    ros::init(argc, argv, "slamesher");

    ros::NodeHandle nh;
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);//Debug Info

    std::string file_loc_report, seq, console_log_path;
    nh.param("slamesher/file_loc_report", file_loc_report, std::string("not_set"));
    nh.param("slamesher/seq", seq, std::string("/00"));
    nh.param("slamesher/console_log_path", console_log_path, std::string(""));
    if (!console_log_path.empty()) {
        g_console_log_tee.enable(makeSeqSpecificLogPath(console_log_path, seq));
    } else {
        g_console_log_tee.enableAuto(file_loc_report, seq);
    }
    if (g_console_log_tee.isEnabled()) {
        std::cout << "[ConsoleLog] writing stdout/stderr to: "
                  << g_console_log_tee.path() << std::endl;
    }

    std::cout << std::setprecision(5) << setiosflags(std::ios::fixed);
    std::cout << "====START====" << std::endl;

    ros::Rate r(1);

    SLAMesher slamesher(nh, param, g_data);
    r.sleep();//waiting for the topic registration
    slamesher.process();

    return 0;
}

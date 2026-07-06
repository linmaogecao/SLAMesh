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
#include <sstream>
#include <unordered_map>
#include <omp.h>
Parameter param;//parameters
Log g_data;//global variables
static ConsoleLogTee g_console_log_tee;

constexpr const char* kBsplineBuildDir = "/home/albus/slam-math/Bspline/build";

namespace {
constexpr double kMatchDropRatioThr = 0.70;
constexpr int kMatchDropAbsThr = 150;

inline double farLayerZFloor(double ground_z_min)
{
    return ground_z_min - param.range_image_far_z_floor_offset;
}

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

Transf kittiVelo2CamExtrinsic()
{
    Transf T_velo2cam = Eigen::Matrix4d::Identity();
    if (param.seq == "/00" || param.seq == "/01" || param.seq == "/02" || param.seq == "/13" ||
        param.seq == "/14" || param.seq == "/15" || param.seq == "/16" || param.seq == "/17" ||
        param.seq == "/18" || param.seq == "/19" || param.seq == "/20" || param.seq == "/21") {
        T_velo2cam << 4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02,
                      -7.210626507497e-03, 8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02,
                      9.999738645903e-01, 4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01,
                      0.0, 0.0, 0.0, 1.0;
    } else if (param.seq == "/03") {
        T_velo2cam << 2.347736981471e-04, -9.999441545438e-01, -1.056347781105e-02, -2.796816941295e-03,
                      1.044940741659e-02, 1.056535364138e-02, -9.998895741176e-01, -7.510879138296e-02,
                      9.999453885620e-01, 1.243653783865e-04, 1.045130299567e-02, -2.721327964059e-01,
                      0.0, 0.0, 0.0, 1.0;
    } else if (param.seq == "/04" || param.seq == "/05" || param.seq == "/06" || param.seq == "/07" ||
               param.seq == "/08" || param.seq == "/09" || param.seq == "/10" || param.seq == "/11" ||
               param.seq == "/12") {
        T_velo2cam << -1.857739385241e-03, -9.999659513510e-01, -8.039975204516e-03, -4.784029760483e-03,
                      -6.481465826011e-03, 8.051860151134e-03, -9.999466081774e-01, -7.337429464231e-02,
                      9.999773098287e-01, -1.805528627661e-03, -6.496203536139e-03, -3.339968064433e-01,
                      0.0, 0.0, 0.0, 1.0;
    }
    return T_velo2cam;
}

double poseDeltaNorm(const Transf& T_from, const Transf& T_to)
{
    const Transf dT = T_from.inverse() * T_to;
    return dT.block<3, 1>(0, 3).norm() +
           5.0 * (dT.block<3, 3>(0, 0) - Eigen::Matrix3d::Identity()).norm();
}

void poseAlignedError(const Transf& T_gt, const Transf& T_pred,
                      Transf& T_gt_align_ref, Transf& T_pred_align_ref,
                      bool& align_init,
                      double& ape_t, double& ape_r,
                      double& err_tx, double& err_ty, double& err_tz)
{
    if (!align_init) {
        T_gt_align_ref = T_gt;
        T_pred_align_ref = T_pred;
        align_init = true;
    }
    const Transf T_gt_rel = T_gt_align_ref.inverse() * T_gt;
    const Transf T_pred_rel = T_pred_align_ref.inverse() * T_pred;
    const Transf T_err = T_gt_rel.inverse() * T_pred_rel;
    const Eigen::Vector3d et = T_err.block<3, 1>(0, 3);
    err_tx = et.x();
    err_ty = et.y();
    err_tz = et.z();
    ape_t = et.norm();
    const Eigen::Matrix3d R_err = T_err.block<3, 3>(0, 0);
    ape_r = Eigen::AngleAxisd(R_err).angle();
}

bool loadKittiGtPoseCam(int frame_idx, Transf& T_cam_out)
{
    if (param.dataset != 1) return false;
    const std::filesystem::path poses_dir =
        std::filesystem::path(param.file_loc_dataset).parent_path() / "poses";
    const std::string pose_path = (poses_dir / (seqIdFromParam(param.seq) + ".txt")).string();
    std::ifstream fin(pose_path);
    if (!fin) return false;
    std::string line;
    for (int i = 0; i <= frame_idx; ++i) {
        if (!std::getline(fin, line)) return false;
    }
    std::istringstream iss(line);
    T_cam_out = Eigen::Matrix4d::Identity();
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            if (!(iss >> T_cam_out(r, c))) return false;
    return true;
}

bool getGtPoseVelo(int step, Transf& T_gt_velo_out)
{
    const int frame_idx = step - 1;
    if (frame_idx < 0) return false;
    if (param.odom_available && param.read_offline_pcd &&
        frame_idx < static_cast<int>(g_data.odom_offline.size())) {
        T_gt_velo_out = state2quat2trans3(g_data.odom_offline[frame_idx]);
        return true;
    }
    Transf T_cam;
    if (!loadKittiGtPoseCam(frame_idx, T_cam)) return false;
    const Transf T_velo2cam = kittiVelo2CamExtrinsic();
    T_gt_velo_out = T_velo2cam.inverse() * T_cam * T_velo2cam;
    return true;
}

std::ofstream& diagLogFile(const char* filename, const char* header)
{
    static std::unordered_map<std::string, std::ofstream> files;
    static std::unordered_map<std::string, bool> header_written;
    const std::string key(filename);
    auto it = files.find(key);
    if (it == files.end()) {
        std::error_code ec;
        std::filesystem::create_directories(kBsplineBuildDir, ec);
        std::ofstream f(std::string(kBsplineBuildDir) + "/" + filename,
                        std::ios::out | std::ios::trunc);
        it = files.emplace(key, std::move(f)).first;
        header_written[key] = false;
    }
    if (it->second.is_open() && !header_written[key]) {
        it->second << header << "\n";
        it->second << std::fixed << std::setprecision(6);
        header_written[key] = true;
    }
    return it->second;
}

void logPerFrameDiagnostics(int step,
                            const Transf& T_world,
                            const Transf& T_guess,
                            const SLAMesher::RegFrameDiag* reg_diag,
                            double delta_guess_vs_last)
{
    static Transf T_gt_align_ref = Eigen::Matrix4d::Identity();
    static Transf T_pred_align_ref = Eigen::Matrix4d::Identity();
    static bool align_init = false;
    static double prev_ape_t = 0;
    static bool prev_ape_valid = false;

    const auto& gnd = reg_diag ? reg_diag->gnd : SLAMesher::MatchQualityStats{};
    const auto& obs = reg_diag ? reg_diag->obs : SLAMesher::MatchQualityStats{};
    const double delta_gnd = reg_diag ? reg_diag->delta_gnd : 0;
    const double delta_obs_iter0 = reg_diag ? reg_diag->delta_obs_iter0 : 0;
    const double delta_obs_final = reg_diag ? reg_diag->delta_obs_final : 0;

    auto& f_gnd = diagLogFile("per_frame_gnd_quality.txt",
        "# step n_gnd mean_dist std_dist max_dist mean_dz p25_dz p75_dz");
    f_gnd << step << " " << gnd.n << " " << gnd.mean_dist << " " << gnd.std_dist << " "
          << gnd.max_dist << " " << gnd.mean_dz << " " << gnd.p25_dz << " " << gnd.p75_dz << "\n";
    f_gnd.flush();

    auto& f_obs = diagLogFile("per_frame_obs_quality.txt",
        "# step n_obs mean_dist std_dist max_dist delta_obs_final");
    f_obs << step << " " << obs.n << " " << obs.mean_dist << " " << obs.std_dist << " "
          << obs.max_dist << " " << delta_obs_final << "\n";
    f_obs.flush();

    auto& f_stage = diagLogFile("per_frame_stage_delta.txt",
        "# step delta_gnd delta_obs_iter0 delta_obs_final");
    f_stage << step << " " << delta_gnd << " " << delta_obs_iter0 << " " << delta_obs_final << "\n";
    f_stage.flush();

    auto& f_zbias = diagLogFile("per_frame_gnd_z_bias.txt",
        "# step mean_dz p25_dz p75_dz");
    f_zbias << step << " " << gnd.mean_dz << " " << gnd.p25_dz << " " << gnd.p75_dz << "\n";
    f_zbias.flush();

    auto& f_guess = diagLogFile("per_frame_guess_quality.txt",
        "# step delta_guess_vs_last delta_gnd delta_obs_final");
    f_guess << step << " " << delta_guess_vs_last << " " << delta_gnd << " "
            << delta_obs_final << "\n";
    f_guess.flush();

    auto& f_debug = diagLogFile("per_frame_debug.txt",
        "# step n_gnd mean_gnd_dist mean_gnd_dz n_obs mean_obs_dist delta_gnd delta_obs_final");
    f_debug << step << " " << gnd.n << " " << gnd.mean_dist << " " << gnd.mean_dz << " "
            << obs.n << " " << obs.mean_dist << " " << delta_gnd << " " << delta_obs_final << "\n";
    f_debug.flush();

    Transf T_gt_velo;
    if (!getGtPoseVelo(step, T_gt_velo)) return;

    double ape_t = 0, ape_r = 0, err_tx = 0, err_ty = 0, err_tz = 0;
    poseAlignedError(T_gt_velo, T_world, T_gt_align_ref, T_pred_align_ref, align_init,
                     ape_t, ape_r, err_tx, err_ty, err_tz);
    const double delta_ape_t = prev_ape_valid ? (ape_t - prev_ape_t) : 0;
    prev_ape_t = ape_t;
    prev_ape_valid = true;

    Eigen::Quaterniond q_pred(T_world.block<3, 3>(0, 0));
    auto& f_ape = diagLogFile("per_frame_ape.txt",
        "# step tx ty tz qx qy qz qw ape_t ape_r err_tx err_ty err_tz delta_ape_t");
    f_ape << step << " "
          << T_world(0, 3) << " " << T_world(1, 3) << " " << T_world(2, 3) << " "
          << q_pred.x() << " " << q_pred.y() << " " << q_pred.z() << " " << q_pred.w() << " "
          << ape_t << " " << ape_r << " "
          << err_tx << " " << err_ty << " " << err_tz << " "
          << delta_ape_t << "\n";
    f_ape.flush();
}

}  // namespace

SLAMesher::MatchQualityStats SLAMesher::computeGndMatchQuality(const std::vector<RegMatch>& ms)
{
    MatchQualityStats q;
    std::vector<double> dists, dzs;
    dists.reserve(ms.size());
    dzs.reserve(ms.size());
    for (const auto& m : ms) {
        if (!m.is_ground) continue;
        const double d_n = std::abs((m.p_world - m.curvature.point).dot(m.curvature.normal));
        dists.push_back(d_n);
        dzs.push_back(m.p_world.z() - m.curvature.point.z());
    }
    q.n = static_cast<int>(dists.size());
    if (q.n == 0) return q;
    double sum = 0, sum2 = 0, sum_dz = 0;
    q.max_dist = 0;
    for (int i = 0; i < q.n; ++i) {
        sum += dists[i];
        sum2 += dists[i] * dists[i];
        q.max_dist = std::max(q.max_dist, dists[i]);
        sum_dz += dzs[i];
    }
    q.mean_dist = sum / q.n;
    q.std_dist = std::sqrt(std::max(0.0, sum2 / q.n - q.mean_dist * q.mean_dist));
    q.mean_dz = sum_dz / q.n;
    std::sort(dzs.begin(), dzs.end());
    q.p25_dz = dzs[std::max(0, q.n / 4 - 1)];
    q.p75_dz = dzs[std::min(q.n - 1, (q.n * 3) / 4)];
    return q;
}

SLAMesher::MatchQualityStats SLAMesher::computeObsMatchQuality(const std::vector<RegMatch>& ms)
{
    MatchQualityStats q;
    std::vector<double> dists;
    dists.reserve(ms.size());
    for (const auto& m : ms) {
        if (m.is_ground) continue;
        const double d_n = std::abs((m.p_world - m.curvature.point).dot(m.curvature.normal));
        dists.push_back(d_n);
    }
    q.n = static_cast<int>(dists.size());
    if (q.n == 0) return q;
    double sum = 0, sum2 = 0;
    q.max_dist = 0;
    for (double d : dists) {
        sum += d;
        sum2 += d * d;
        q.max_dist = std::max(q.max_dist, d);
    }
    q.mean_dist = sum / q.n;
    q.std_dist = std::sqrt(std::max(0.0, sum2 / q.n - q.mean_dist * q.mean_dist));
    return q;
}

// 配准关联诊断：点到曲面的真实状态（与 Ceres 残差条数无关）
enum class RegAssocStatus : int {
    NoSurface = 0,      // 关联阶段未找到任何曲面
    FootprintFail = 1,  // 有曲面但 footprint 失败 (d < 1e-6)
    DistTooLarge = 2,   // footprint 成功但 d > match_dist_thr
    Matched = 3,        // d <= match_dist_thr，真正匹配上曲面
};

struct RegPointAssoc {
    int scan_idx = -1;
    Eigen::Vector3d p_world = Eigen::Vector3d::Zero();
    double best_dist = -1.0;
    RegAssocStatus status = RegAssocStatus::NoSurface;
    bool is_gnd_cand = false;
    bool is_obs_cand = false;
};

struct RegMatchFunnel {
    int gnd_cand = 0;
    int gnd_surf = 0;
    int gnd_fp_ok = 0;
    int gnd_dist_fail = 0;
    int gnd_ceres = 0;
    double gnd_mean_dist = -1;
    int obs_cand_px = 0;
    int obs_surf_px = 0;
    int obs_no_surf_px = 0;
    int obs_pairs = 0;
    int obs_fp_ok = 0;
    int obs_dist_fail = 0;
    int obs_ceres = 0;
    double obs_mean_dist = -1;
};

void applyFootprintToAssoc(RegPointAssoc& a,
                           const Eigen::Vector3d& p_w,
                           double d,
                           double match_dist_thr)
{
    a.p_world = p_w;
    if (d < 1e-6) {
        if (a.status == RegAssocStatus::NoSurface)
            a.status = RegAssocStatus::FootprintFail;
        return;
    }
    if (a.best_dist < 0.0 || d < a.best_dist)
        a.best_dist = d;
    if (d <= match_dist_thr)
        a.status = RegAssocStatus::Matched;
    else if (a.status != RegAssocStatus::Matched)
        a.status = RegAssocStatus::DistTooLarge;
}

void logRegMatchFunnel(int step, int iter, const RegMatchFunnel& f, bool snapshot_step)
{
    static std::ofstream fout;
    static bool header_written = false;
    if (!fout.is_open()) {
        std::error_code ec;
        std::filesystem::create_directories(kBsplineBuildDir, ec);
        fout.open(std::string(kBsplineBuildDir) + "/reg_match_funnel.txt",
                  std::ios::out | std::ios::trunc);
        header_written = false;
    }
    if (fout.is_open() && !header_written) {
        fout << "# step iter"
             << " gnd_cand gnd_surf gnd_fp_ok gnd_dist_fail gnd_ceres gnd_mean_dist"
             << " obs_cand_px obs_surf_px obs_no_surf_px"
             << " obs_pairs obs_fp_ok obs_dist_fail obs_ceres obs_mean_dist\n";
        header_written = true;
    }
    if (fout.is_open()) {
        fout << step << " " << iter << " "
             << f.gnd_cand << " " << f.gnd_surf << " " << f.gnd_fp_ok << " "
             << f.gnd_dist_fail << " " << f.gnd_ceres << " " << f.gnd_mean_dist << " "
             << f.obs_cand_px << " " << f.obs_surf_px << " " << f.obs_no_surf_px << " "
             << f.obs_pairs << " " << f.obs_fp_ok << " " << f.obs_dist_fail << " "
             << f.obs_ceres << " " << f.obs_mean_dist << "\n";
        fout.flush();
    }

    std::cout << "  [MatchFunnel] step=" << step << " iter=" << iter
              << " gnd " << f.gnd_cand << "->" << f.gnd_surf << "->"
              << f.gnd_fp_ok << "->" << f.gnd_ceres
              << " md=" << f.gnd_mean_dist
              << " | obs_px " << f.obs_cand_px << "->" << f.obs_surf_px
              << " | pairs " << f.obs_pairs << "->" << f.obs_fp_ok << "->"
              << f.obs_ceres << " md=" << f.obs_mean_dist << std::endl;

    if (snapshot_step) {
        const std::string snap = std::string(kBsplineBuildDir) + "/reg_match_funnel_step"
                                 + std::to_string(step) + ".txt";
        std::ofstream fsnap(snap, std::ios::out | std::ios::trunc);
        if (fsnap.is_open()) {
            fsnap << "# step=" << step << " iter=" << iter << " (snapshot at dump_frame)\n";
            fsnap << "# gnd: cand->surf->fp_ok->ceres | obs: cand_px->surf_px | pairs->fp_ok->ceres\n";
            fsnap << f.gnd_cand << " " << f.gnd_surf << " " << f.gnd_fp_ok << " "
                  << f.gnd_ceres << " " << f.obs_cand_px << " " << f.obs_surf_px << " "
                  << f.obs_no_surf_px << " " << f.obs_pairs << " " << f.obs_fp_ok << " "
                  << f.obs_ceres << "\n";
        }
    }
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
        for(int i = 1; i<= step; i++){
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
    nh.param("slamesher/dump_cluster_step", dump_cluster_step, 0);
    nh.param("slamesher/dump_occluded_step", dump_occluded_step, 0);
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
    std::cout<<"all_surfaces_max_step: "<<all_surfaces_max_step
             <<(all_surfaces_max_step > 0 ? " (export surfaces with created_step <= N)" : " (off)")
             <<std::endl;
    std::cout<<"map_update_interval: "<<map_update_interval
             <<"  ground_build_interval: "<<ground_build_interval<<std::endl;
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
    nh.param("slamesher/ground_cell_size",    ground_cell_size,    6.0);
    nh.param("slamesher/ground_cell_min_pts", ground_cell_min_pts, 80);
    nh.param("slamesher/ground_cell_num_cp",  ground_cell_num_cp,  5);
    nh.param("slamesher/ground_query_radius", ground_query_radius, 1);
    nh.param("slamesher/ground_map_skip_points", ground_map_skip_points, 8);
    nh.param("slamesher/ground_cell_max_pts",    ground_cell_max_pts,    400);
    nh.param("slamesher/ground_fit_max_pts",     ground_fit_max_pts,     150);
    nh.param("slamesher/ground_skip_points",  ground_skip_points,  40);
    nh.param("slamesher/ground_clear_dist",   ground_clear_dist,   150.0);
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
              << "  obs_match_per_surf_max=" << obs_match_per_surf_max << std::endl;
    std::cout << "ground_grid: cell=" << ground_cell_size
              << "m min_pts=" << ground_cell_min_pts
              << " num_cp=" << ground_cell_num_cp
              << " query_r=" << ground_query_radius
              << " map_skip=" << ground_map_skip_points
              << " cell_max=" << ground_cell_max_pts
              << " fit_max=" << ground_fit_max_pts
              << " reg_skip=" << ground_skip_points
              << " clear_dist=" << ground_clear_dist << "m" << std::endl;
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

Transf SLAMesher::registerScanToMap(const pcl::PointCloud<pcl::PointXYZ>& scan_local,
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
                                    bool use_far,
                                    RegFrameDiag* diag_out)
{
    const Transf T_guess_in = T_guess;
    Transf T_curr = T_guess;
    double delta_scale = 100.0;
    int prev_match_count = -1;
    std::unordered_map<int, RegPointAssoc> last_point_assoc;
    std::unordered_map<int, std::vector<int>> obs_prev_indices;
    std::unordered_map<int, std::vector<std::pair<BSplineSurface::Parameter, BSplineSurface::Parameter>>> obs_prev_uv;
    std::unordered_map<const BSplineSurface*, std::vector<int>> gnd_prev_indices;
    std::unordered_map<const BSplineSurface*, std::vector<std::pair<BSplineSurface::Parameter, BSplineSurface::Parameter>>> gnd_prev_uv;

    const int gnd_skip = param.ground_skip_points > 0 ? param.ground_skip_points : skip_points;
    const int col_step = std::max(1, param.obs_rimg_col_step);
    const double far_match_min_z = farLayerZFloor(ground_z_min);

    // ─────────────────────────────────────────────────────────────────────────
    // 地面关联：仅用 z 带点 + GroundGridMap，不碰 range image / 障碍曲面
    // ─────────────────────────────────────────────────────────────────────────
    auto runGroundAssoc = [&](const Transf& T,
                              std::vector<RegMatch>& gnd_out,
                              RegMatchFunnel& funnel_out,
                              std::unordered_map<int, RegPointAssoc>& assoc_out) {
        gnd_out.clear();
        assoc_out.clear();
        funnel_out = {};

        auto toWorld = [&](int idx) -> Eigen::Vector3d {
            return T.block<3,3>(0,0) *
                   Eigen::Vector3d(scan_local[idx].x, scan_local[idx].y, scan_local[idx].z) +
                   T.block<3,1>(0,3);
        };

        std::unordered_map<const BSplineSurface*, std::vector<int>> gnd_surf_to_pts;

        for (int i = 0; i < (int)scan_local.size(); i += gnd_skip) {
            const double lz = scan_local[i].z;
            if (lz < ground_z_min || lz > ground_z_max) continue;
            ++funnel_out.gnd_cand;
            const Eigen::Vector3d p_w = toWorld(i);
            auto& a = assoc_out[i];
            a.scan_idx = i; a.p_world = p_w; a.is_gnd_cand = true;
            const BSplineSurface* sp = ground_grid.queryNearestSurface(p_w);
            if (!sp) continue;
            ++funnel_out.gnd_surf;
            gnd_surf_to_pts[sp].push_back(i);
        }

        for (auto& [surf_ptr, indices] : gnd_surf_to_pts) {
            if (!surf_ptr) continue;
            const auto& knU = surf_ptr->getKnotsU();
            const auto& knV = surf_ptr->getKnotsV();
            const auto& cps = surf_ptr->getControls();
            const int num_cpv = surf_ptr->getNumCpV();

            std::vector<Eigen::Vector3d> pts_world;
            pts_world.reserve(indices.size());
            for (int idx : indices) pts_world.push_back(toWorld(idx));

            std::vector<std::pair<BSplineSurface::Parameter, BSplineSurface::Parameter>> footprints;
            std::vector<double> dists;
            auto prev_idx_it = gnd_prev_indices.find(surf_ptr);
            auto prev_uv_it  = gnd_prev_uv.find(surf_ptr);
            if (prev_idx_it != gnd_prev_indices.end() &&
                prev_uv_it  != gnd_prev_uv.end() &&
                prev_idx_it->second == indices &&
                prev_uv_it->second.size() == indices.size())
                footprints = prev_uv_it->second;

            const_cast<BSplineSurface*>(surf_ptr)->findFootPrintWarm(
                pts_world, footprints, dists, /*newton_steps=*/6);
            gnd_prev_indices[surf_ptr] = indices;
            gnd_prev_uv[surf_ptr]      = footprints;

            for (int k = 0; k < (int)indices.size(); ++k) {
                const double d = std::sqrt(std::abs(dists[k]));
                applyFootprintToAssoc(assoc_out[indices[k]], pts_world[k], d, match_dist_thr);
                if (d < 1e-6) continue;
                ++funnel_out.gnd_fp_ok;
                if (d > match_dist_thr) { ++funnel_out.gnd_dist_fail; continue; }
                auto [paraU, paraV] = footprints[k];
                SurfaceCurvature curv = const_cast<BSplineSurface*>(surf_ptr)
                    ->getCurvature(paraU, paraV, knU, knV, cps, num_cpv);
                if (curv.normal.norm() < 1e-9) continue;
                curv.normal.normalize();
                ++funnel_out.gnd_ceres;
                RegMatch m;
                m.p_local = Eigen::Vector3d(scan_local[indices[k]].x,
                                            scan_local[indices[k]].y,
                                            scan_local[indices[k]].z);
                m.p_world   = pts_world[k];
                m.curvature = curv;
                m.scan_idx  = indices[k];
                m.is_ground = true;
                gnd_out.push_back(m);
            }
        }
    };

    // ─────────────────────────────────────────────────────────────────────────
    // 障碍关联：用给定 T 把 range image 像素投世界 → queryCandidates → footprint
    // ─────────────────────────────────────────────────────────────────────────
    auto runObsAssoc = [&](const Transf& T,
                           std::vector<RegMatch>& obs_out,
                           RegMatchFunnel& funnel_out,
                           std::unordered_map<int, RegPointAssoc>& assoc_out) {
        obs_out.clear();
        funnel_out.obs_cand_px = funnel_out.obs_surf_px = 0;
        funnel_out.obs_no_surf_px = funnel_out.obs_pairs = 0;
        funnel_out.obs_fp_ok = funnel_out.obs_dist_fail = funnel_out.obs_ceres = 0;

        auto toWorld = [&](int idx) -> Eigen::Vector3d {
            return T.block<3,3>(0,0) *
                   Eigen::Vector3d(scan_local[idx].x, scan_local[idx].y, scan_local[idx].z) +
                   T.block<3,1>(0,3);
        };

        std::unordered_map<int, std::vector<int>> obs_surf_to_pts;

        auto collectFromRangeImage = [&](const RangeImageProcessor& rp, double obs_min_z) {
            const int W = rp.W_COLS, H = rp.H_SCANS;
            for (int u = 0; u < H; ++u) {
                for (int v = 0; v < W; v += col_step) {
                    const int px_idx = u * W + v;
                    const auto& px = rp.range_image_[px_idx];
                    if (!px.valid || px.z < obs_min_z) continue;
                    const int cloud_idx = rp.pixel_to_cloud_idx_[px_idx];
                    if (cloud_idx < 0 || cloud_idx >= (int)scan_local.size()) continue;
                    ++funnel_out.obs_cand_px;
                    const Eigen::Vector3d p_w = toWorld(cloud_idx);
                    auto& a = assoc_out[cloud_idx];
                    a.scan_idx = cloud_idx; a.p_world = p_w; a.is_obs_cand = true;
                    std::vector<int> cands = bspline_map.queryCandidates(p_w, 1);
                    if (cands.empty()) continue;
                    ++funnel_out.obs_surf_px;
                    for (int sid : cands)
                        obs_surf_to_pts[sid].push_back(cloud_idx);
                }
            }
        };
        collectFromRangeImage(rp_near, match_min_z);
        if (use_far) collectFromRangeImage(rp_far, far_match_min_z);
        funnel_out.obs_no_surf_px = funnel_out.obs_cand_px - funnel_out.obs_surf_px;

        if (param.obs_match_per_surf_max > 0) {
            for (auto& [sid, idxs] : obs_surf_to_pts) {
                const int cap = param.obs_match_per_surf_max;
                if ((int)idxs.size() > cap) {
                    const int step = (int)idxs.size() / cap;
                    std::vector<int> kept; kept.reserve(cap);
                    for (int i = 0; i < (int)idxs.size() && (int)kept.size() < cap; i += step)
                        kept.push_back(idxs[i]);
                    idxs = std::move(kept);
                }
            }
        }
        for (const auto& [sid, idxs] : obs_surf_to_pts)
            funnel_out.obs_pairs += static_cast<int>(idxs.size());

        for (auto& [sid, indices] : obs_surf_to_pts) {
            const BSplineMapEntry* entry = bspline_map.getEntry(sid);
            if (!entry || !entry->surface || entry->is_ground) continue;
            auto& surf = entry->surface;
            const auto& knU = surf->getKnotsU();
            const auto& knV = surf->getKnotsV();
            const auto& cps = surf->getControls();
            const int num_cpv = surf->getNumCpV();

            std::vector<Eigen::Vector3d> pts_world;
            pts_world.reserve(indices.size());
            for (int idx : indices) pts_world.push_back(toWorld(idx));

            std::vector<std::pair<BSplineSurface::Parameter, BSplineSurface::Parameter>> footprints;
            std::vector<double> dists;
            auto prev_idx_it = obs_prev_indices.find(sid);
            auto prev_uv_it  = obs_prev_uv.find(sid);
            if (prev_idx_it != obs_prev_indices.end() &&
                prev_uv_it  != obs_prev_uv.end() &&
                prev_idx_it->second == indices &&
                prev_uv_it->second.size() == indices.size())
                footprints = prev_uv_it->second;

            surf->findFootPrintWarm(pts_world, footprints, dists, /*newton_steps=*/6);
            obs_prev_indices[sid] = indices;
            obs_prev_uv[sid]      = footprints;

            for (int k = 0; k < (int)indices.size(); ++k) {
                const double d = std::sqrt(std::abs(dists[k]));
                applyFootprintToAssoc(assoc_out[indices[k]], pts_world[k], d, match_dist_thr);
                if (d < 1e-6) { ++funnel_out.obs_dist_fail; continue; }
                ++funnel_out.obs_fp_ok;
                if (d > match_dist_thr) { ++funnel_out.obs_dist_fail; continue; }
                auto [paraU, paraV] = footprints[k];
                SurfaceCurvature curv = surf->getCurvature(paraU, paraV, knU, knV, cps, num_cpv);
                if (curv.normal.norm() < 1e-9) continue;
                curv.normal.normalize();
                ++funnel_out.obs_ceres;
                RegMatch m;
                m.p_local = Eigen::Vector3d(scan_local[indices[k]].x,
                                            scan_local[indices[k]].y,
                                            scan_local[indices[k]].z);
                m.p_world   = pts_world[k];
                m.curvature = curv;
                m.scan_idx  = indices[k];
                m.is_ground = false;
                obs_out.push_back(m);
            }
        }
    };

    // ─────────────────────────────────────────────────────────────────────────
    // 辅助：运行单步 Ceres，更新 T_curr，返回 delta_scale
    // ─────────────────────────────────────────────────────────────────────────
    auto solvePose = [&](const std::vector<RegMatch>& ms, const char* tag) -> double {
        Eigen::Quaterniond q_init(T_curr.block<3,3>(0,0));
        double parameters[7];
        parameters[0] = q_init.x(); parameters[1] = q_init.y();
        parameters[2] = q_init.z(); parameters[3] = q_init.w();
        parameters[4] = T_curr(0,3);
        parameters[5] = T_curr(1,3);
        parameters[6] = T_curr(2,3);

        ceres::LossFunction* loss = new ceres::HuberLoss(0.5);
        ceres::Problem problem;
        problem.AddParameterBlock(parameters, 7, new PoseSE3Parameterization());
        for (const auto& m : ms)
            problem.AddResidualBlock(
                new SDMRegistrationCostFunction(m.p_local, m.p_world, m.curvature),
                loss, parameters);

        ceres::Solver::Options opts;
        opts.linear_solver_type          = ceres::DENSE_QR;
        opts.max_num_iterations          = 1;
        opts.minimizer_progress_to_stdout = false;
        opts.num_threads                 = 4;
        ceres::Solver::Summary summary;
        ceres::Solve(opts, &problem, &summary);

        Eigen::Map<Eigen::Quaterniond> q_opt(parameters);
        Eigen::Map<Eigen::Vector3d>    t_opt(parameters + 4);
        Transf T_new = Eigen::Matrix4d::Identity();
        T_new.block<3,3>(0,0) = q_opt.normalized().toRotationMatrix();
        T_new.block<3,1>(0,3) = t_opt;
        const double ds = (T_new.block<3,1>(0,3) - T_curr.block<3,1>(0,3)).norm() +
                          5.0 * (T_new.block<3,3>(0,0) - T_curr.block<3,3>(0,0)).norm();
        T_curr = T_new;
        std::cout << "    [" << tag << "] n=" << ms.size() << " delta=" << ds << std::endl;
        return ds;
    };

    // ─────────────────────────────────────────────────────────────────────────
    // gnd_match_dist 日志
    // ─────────────────────────────────────────────────────────────────────────
    static std::ofstream gnd_dist_file;
    if (!gnd_dist_file.is_open()) {
        gnd_dist_file.open(std::string(kBsplineBuildDir) + "/gnd_match_dist.txt",
                           std::ios::out | std::ios::trunc);
        gnd_dist_file << std::fixed << std::setprecision(5);
        gnd_dist_file << "# dist_normal  dist_abs  p_w_x  p_w_y  p_w_z  surf_z\n";
    }
    auto logGndDist = [&](const std::vector<RegMatch>& ms, int iter) {
        double sum_d = 0, sum_dz = 0; int cnt = 0;
        for (const auto& m : ms) {
            if (!m.is_ground) continue;
            const double d_n = (m.p_world - m.curvature.point).dot(m.curvature.normal);
            gnd_dist_file << d_n << "  " << std::abs(d_n)
                          << "  " << m.p_world.x() << "  " << m.p_world.y()
                          << "  " << m.p_world.z() << "  " << m.curvature.point.z() << "\n";
            sum_d += std::abs(d_n); sum_dz += m.p_world.z() - m.curvature.point.z(); ++cnt;
        }
        if (cnt > 0) {
            gnd_dist_file << "# step=" << g_data.step << " iter=" << iter
                          << " n_gnd=" << cnt << " mean|d|=" << sum_d/cnt
                          << " mean_dz=" << sum_dz/cnt << "\n";
            gnd_dist_file.flush();
        }
    };

    // ─────────────────────────────────────────────────────────────────────────
    // 阶段 1：地面点 → 地面曲面关联 → Ceres（1 次）→ T_gnd
    // ─────────────────────────────────────────────────────────────────────────
    std::vector<RegMatch> gnd_matches_final;
    Transf T_after_gnd = T_curr;
    {
        std::vector<RegMatch> gnd_matches;
        RegMatchFunnel gnd_funnel{};
        std::unordered_map<int, RegPointAssoc> gnd_assoc;
        runGroundAssoc(T_curr, gnd_matches, gnd_funnel, gnd_assoc);
        gnd_funnel.gnd_mean_dist = computeGndMatchQuality(gnd_matches).mean_dist;
        logRegMatchFunnel(g_data.step, -2, gnd_funnel, false);
        last_point_assoc = gnd_assoc;
        gnd_matches_final = gnd_matches;

        std::cout << "  [gnd-assoc] cand=" << gnd_funnel.gnd_cand
                  << " surf=" << gnd_funnel.gnd_surf
                  << " ceres=" << gnd_funnel.gnd_ceres << std::endl;

        if ((int)gnd_matches.size() >= 5) {
            delta_scale = solvePose(gnd_matches, "gnd");
            logGndDist(gnd_matches, -1);
        } else {
            std::cout << "  [gnd] skip, too few matches=" << gnd_matches.size() << std::endl;
        }
        T_after_gnd = T_curr;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 阶段 2：障碍点（用 T_gnd 投世界）→ 障碍曲面关联 → Ceres × max_iters
    // ─────────────────────────────────────────────────────────────────────────
    std::vector<RegMatch> obs_matches_final;
    double delta_obs_iter0 = 0;
    delta_scale = 100.0;
    for (int iter = 0; iter < max_iters && delta_scale > converge_thr; iter++) {
        std::vector<RegMatch> obs_matches;
        RegMatchFunnel obs_funnel{};
        std::unordered_map<int, RegPointAssoc> obs_assoc = last_point_assoc;
        runObsAssoc(T_curr, obs_matches, obs_funnel, obs_assoc);
        obs_funnel.obs_mean_dist = computeObsMatchQuality(obs_matches).mean_dist;
        last_point_assoc = obs_assoc;
        obs_matches_final = obs_matches;

        appendMatchDropEvent(g_data.step, iter, prev_match_count,
                             static_cast<int>(obs_matches.size()), 0,
                             static_cast<int>(obs_matches.size()));
        prev_match_count = static_cast<int>(obs_matches.size());
        logRegMatchFunnel(g_data.step, iter, obs_funnel, false);

        if ((int)obs_matches.size() < 5) {
            ROS_WARN("Obs iter %d: only %d obs matches, skip", iter, (int)obs_matches.size());
            break;
        }

        delta_scale = solvePose(obs_matches, "obs");
        if (iter == 0)
            delta_obs_iter0 = poseDeltaNorm(T_after_gnd, T_curr);
        std::cout << "  iter " << iter << ": obs=" << obs_matches.size()
                  << " delta=" << delta_scale << std::endl;
    }

    if (diag_out) {
        diag_out->valid = true;
        diag_out->gnd = computeGndMatchQuality(gnd_matches_final);
        diag_out->obs = computeObsMatchQuality(obs_matches_final);
        diag_out->delta_gnd = poseDeltaNorm(T_guess_in, T_after_gnd);
        diag_out->delta_obs_iter0 = delta_obs_iter0;
        diag_out->delta_obs_final = poseDeltaNorm(T_after_gnd, T_curr);
    }

    if (param.dump_frame > 0 && g_data.step == param.dump_frame) {
        const std::string base = kBsplineBuildDir;
        std::error_code ec;
        std::filesystem::create_directories(base, ec);
        if (ec) {
            std::cerr << "  [Dump] failed to create dir: " << base << " (" << ec.message() << ")\n";
        } else {
            // 用最终位姿再跑一遍地面 + 障碍关联，保证 dump 与配准候选集一致
            std::vector<RegMatch> dump_gnd, dump_obs;
            RegMatchFunnel dump_gnd_funnel{}, dump_obs_funnel{};
            std::unordered_map<int, RegPointAssoc> dump_assoc;
            runGroundAssoc(T_curr, dump_gnd, dump_gnd_funnel, dump_assoc);
            runObsAssoc(T_curr, dump_obs, dump_obs_funnel, dump_assoc);
            last_point_assoc = std::move(dump_assoc);
            // 合并漏斗用于日志
            RegMatchFunnel dump_funnel = dump_gnd_funnel;
            dump_funnel.obs_cand_px    = dump_obs_funnel.obs_cand_px;
            dump_funnel.obs_surf_px    = dump_obs_funnel.obs_surf_px;
            dump_funnel.obs_no_surf_px = dump_obs_funnel.obs_no_surf_px;
            dump_funnel.obs_pairs      = dump_obs_funnel.obs_pairs;
            dump_funnel.obs_fp_ok      = dump_obs_funnel.obs_fp_ok;
            dump_funnel.obs_dist_fail  = dump_obs_funnel.obs_dist_fail;
            dump_funnel.obs_ceres      = dump_obs_funnel.obs_ceres;
            std::vector<RegMatch> dump_matches;
            dump_matches.insert(dump_matches.end(), dump_gnd.begin(), dump_gnd.end());
            dump_matches.insert(dump_matches.end(), dump_obs.begin(), dump_obs.end());
            logRegMatchFunnel(g_data.step, -1, dump_funnel, true);

            const Eigen::Matrix3d R = T_curr.block<3, 3>(0, 0);
            const Eigen::Vector3d t = T_curr.block<3, 1>(0, 3);

            std::ofstream f_scan(base + "/scan_world.txt", std::ios::out | std::ios::trunc);
            f_scan << std::fixed << std::setprecision(6);
            for (const auto& pt : scan_local.points) {
                if (pt.z >= ground_z_min && pt.z <= ground_z_max) continue;
                const Eigen::Vector3d p_w = R * Eigen::Vector3d(pt.x, pt.y, pt.z) + t;
                f_scan << p_w.x() << " " << p_w.y() << " " << p_w.z() << "\n";
            }
            f_scan.close();

            std::ofstream f_gnd(base + "/scan_gnd.txt", std::ios::out | std::ios::trunc);
            std::ofstream f_obs(base + "/scan_obs.txt", std::ios::out | std::ios::trunc);
            f_gnd << std::fixed << std::setprecision(6);
            f_obs << std::fixed << std::setprecision(6);
            for (const auto& [idx, a] : last_point_assoc) {
                if (a.is_gnd_cand)
                    f_gnd << a.p_world.x() << " " << a.p_world.y() << " " << a.p_world.z() << "\n";
                if (a.is_obs_cand)
                    f_obs << a.p_world.x() << " " << a.p_world.y() << " " << a.p_world.z() << "\n";
            }
            f_gnd.close();
            f_obs.close();

            int n_matched_pts = 0, n_unmatched_pts = 0;
            int n_no_surf = 0, n_fp_fail = 0, n_dist_fail = 0;
            std::ofstream f_matched(base + "/matched.txt", std::ios::out | std::ios::trunc);
            std::ofstream f_unmatched(base + "/unmatched.txt", std::ios::out | std::ios::trunc);
            std::ofstream f_dist(base + "/match_dist.txt", std::ios::out | std::ios::trunc);
            f_matched << std::fixed << std::setprecision(6);
            f_unmatched << std::fixed << std::setprecision(6);
            f_dist << std::fixed << std::setprecision(6);
            f_dist << "# x y z dist status  (status: 0=no_surf 1=fp_fail 2=dist>thr 3=matched)\n";

            for (const auto& [idx, a] : last_point_assoc) {
                f_dist << a.p_world.x() << " " << a.p_world.y() << " " << a.p_world.z()
                       << " " << a.best_dist << " " << static_cast<int>(a.status) << "\n";
                if (a.status == RegAssocStatus::Matched) {
                    f_matched << a.p_world.x() << " " << a.p_world.y() << " " << a.p_world.z() << "\n";
                    ++n_matched_pts;
                } else {
                    f_unmatched << a.p_world.x() << " " << a.p_world.y() << " " << a.p_world.z() << "\n";
                    ++n_unmatched_pts;
                    if (a.status == RegAssocStatus::NoSurface) ++n_no_surf;
                    else if (a.status == RegAssocStatus::FootprintFail) ++n_fp_fail;
                    else if (a.status == RegAssocStatus::DistTooLarge) ++n_dist_fail;
                }
            }
            f_matched.close();
            f_unmatched.close();
            f_dist.close();

            std::cout << "  [Dump] step=" << g_data.step
                      << " candidates=" << last_point_assoc.size()
                      << " matched_pts=" << n_matched_pts
                      << " unmatched_pts=" << n_unmatched_pts
                      << " (no_surf=" << n_no_surf
                      << " fp_fail=" << n_fp_fail
                      << " d>thr=" << n_dist_fail << ")"
                      << " ceres_residuals=" << dump_matches.size()
                      << " -> " << base
                      << "/{matched,unmatched,match_dist,scan_gnd,scan_obs}.txt" << std::endl;
        }
    }
    return T_curr;
}

void SLAMesher::runMapBuild(const pcl::PointCloud<pcl::PointXYZ>& scan_local,
                            const Transf& T_world,
                            RangeImageProcessor& range_proc,
                            RangeImageProcessor& range_proc_far,
                            BSplineMap& bspline_map,
                            GroundGridMap& ground_grid,
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
    // 近层：range ∈ [MIN_RANGE, range_image_split)；如不启用双层则全范围投影
    const double split_dist = param.range_image_split;
    const bool use_far_layer = (split_dist > 0.0);
    if (dump_clusters) {
        saveClusterFilterStatsTxt(scan_local, g_data.step, ground_z_min, ground_z_max,
                                  split_dist, range_proc, range_proc_far);
    }
    // range image 不再提前删地面带：地面链已由 z 带直接选点，障碍链需保留低矮障碍
    // z_floor = ground_z_min：允许与地面带下沿齐平的低矮障碍进入 range image
    range_proc.generateRangeImage(scan_local, ground_z_min, ground_z_max, false,
                                  0.0, use_far_layer ? split_dist : 1e9, ground_z_min);

    // 远层：range ∈ [range_image_split, MAX_RANGE]，z_floor 低于近层
    const double far_z_floor = farLayerZFloor(ground_z_min);
    if (use_far_layer)
        range_proc_far.generateRangeImage(scan_local, ground_z_min, ground_z_max, false,
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

    // exclude_ground_band=false：地面链已独立走 z 带选点，障碍链不再预删地面
    SegmentationResult seg = range_proc.segmentRangeImage(
        5, 0.1, MIN_CLUSTER_PTS, ground_z_min, ground_z_max, false);
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
            5, 0.1, far_min, ground_z_min, ground_z_max, false);
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

    // ── 地面分支：雷达系按 z 高度带直接筛点，不经过 range image ──
    if (do_ground) {
        pcl::PointCloud<pcl::PointXYZ> cloud_gnd_world;
        std::vector<Eigen::Vector3d> gnd_lidar_pts;
        gnd_lidar_pts.reserve(scan_local.size() / 16);
        const int gnd_map_skip = std::max(1, param.ground_map_skip_points);
        for (int i = 0; i < (int)scan_local.size(); i += gnd_map_skip) {
            const auto& pt = scan_local.points[i];
            if (pt.z < ground_z_min || pt.z > ground_z_max) continue;
            gnd_lidar_pts.emplace_back(pt.x, pt.y, pt.z);
            Eigen::Vector3d pw = R_w * Eigen::Vector3d(pt.x, pt.y, pt.z) + t_w;
            cloud_gnd_world.push_back(pcl::PointXYZ(
                static_cast<float>(pw.x()), static_cast<float>(pw.y()), static_cast<float>(pw.z())));
        }

        if (g_data.step == 1)
            saveFrame1GroundPoints(gnd_lidar_pts);

        if (!cloud_gnd_world.empty()) {
            // 投格 + 标记需重拟合的格
            auto touched = ground_grid.addPoints(cloud_gnd_world, g_data.step);
            // 对积累点数达标的格重新拟合曲面
            if (profile_frame1_apply)
                BSplineSurface::setApplyProfileLabel("gnd");
            n_added_gnd = ground_grid.refitCells(touched, param.num_thread);
        }
    }

    // ── 障碍分支：仅对非地面 cluster 建图（地面已在分割前排除） ──
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

    // 障碍建图 lambda：A 收集 → B 并行 apply → C 串行入库
    auto buildObsFromSeg = [&](RangeImageProcessor& proc, SegmentationResult& s, int min_pts) {
        // ── 阶段 A：串行筛 cluster + footprint 匹配，打包 tasks ──
        std::vector<ObsBuildTask> tasks;
        for (int cid = 0; cid < (int)s.clusters.size(); cid++) {
            const auto& pixels = s.clusters[cid];
            if ((int)pixels.size() < min_pts) continue;

            if (!do_obstacle) break;
            if (g_data.step != 1 && (n_added_obs + (int)tasks.size()) >= MAX_NEW_SURFACES) break;

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

            const int num_cp = chooseControlGridSize(n_unmatched);
            auto cloud_world = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
            pcl::transformPointCloud(*cloud_local, *cloud_world, T_world.cast<float>());

            auto init_cp = proc.computeInitControlPoints(s, cid, num_cp, num_cp, 2);
            if (init_cp.empty()) continue;
            for (auto& cp : init_cp) cp = R_w * cp + t_w;

            ObsBuildTask t;
            t.cloud_world = cloud_world;
            t.init_cp     = std::move(init_cp);
            t.num_cp      = num_cp;
            tasks.push_back(std::move(t));
        }

        // ── 阶段 B：OMP 并行 apply（每个 task 独立，无共享写） ──
        const int n_omp = std::max(1, param.num_thread);
        // Frame1Apply profile 在并行区不可用（全局变量），先暂停
        const bool had_profile = profile_frame1_apply && do_obstacle;
        if (had_profile) BSplineSurface::setApplyProfileLogPath("");
#pragma omp parallel for schedule(dynamic) num_threads(n_omp)
        for (int i = 0; i < (int)tasks.size(); ++i) {
            auto& t = tasks[i];
            t.surf = std::make_shared<BSplineSurface>(3, 3, t.num_cp, t.num_cp, 0.25);
            t.surf->setExternalInitControls(t.init_cp);
            t.ok = t.surf->apply(t.cloud_world, 10, 1, 1, 0.05);
        }

        // ── 阶段 C：串行入库 ──
        for (auto& t : tasks) {
            if (!t.ok) continue;
            bspline_map.addSurface(t.surf, t.cloud_world, /*is_ground=*/false, g_data.step);
            ++n_added_obs;
        }
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

void SLAMesher::saveGroundGridZ(const GroundGridMap& ground_grid) const
{
    const std::string out_path = std::string(kBsplineBuildDir) + "/ground_grid_z.txt";
    std::ofstream fout(out_path);
    if (!fout.is_open()) return;

    fout << std::fixed << std::setprecision(5);

    std::vector<GroundCellKey> keys;
    keys.reserve(ground_grid.cells().size());
    for (const auto& [k, cell] : ground_grid.cells()) {
        if (cell.surf) keys.push_back(k);
    }
    std::sort(keys.begin(), keys.end(), [](const GroundCellKey& a, const GroundCellKey& b) {
        return a.ix < b.ix || (a.ix == b.ix && a.iy < b.iy);
    });

    for (const auto& k : keys) {
        const auto& cell = ground_grid.cells().at(k);
        const auto& cps = cell.surf->getControls();
        if (cps.empty()) continue;
        double z_sum = 0;
        for (const auto& cp : cps) z_sum += cp.z();
        fout << (z_sum / static_cast<double>(cps.size()))
             << " " << k.ix << " " << k.iy << "\n";
    }
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
    GroundGridMap ground_grid(param.ground_cell_size,
                              param.ground_cell_min_pts,
                              param.ground_cell_num_cp,
                              param.ground_query_radius,
                              param.ground_cell_max_pts,
                              param.ground_fit_max_pts);
    RangeImageProcessor range_proc;
    RangeImageProcessor range_proc_far;  // 远层（range >= range_image_split）

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
        traj_file.open("/tmp/bspline_traj_xyz.txt", std::ios::out | std::ios::trunc);
        traj_file << std::fixed << std::setprecision(6);
    }

    // buildGroundMap 用：传感器系下地面点 z 范围
    const double GROUND_Z_MIN    = -3.0;
    const double GROUND_Z_MAX    = -1.5;
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
            processFirstFrame(T_world);
            runMapBuild(scan_local, T_world, range_proc, range_proc_far, bspline_map, ground_grid, match_dist_thr, GROUND_Z_MIN, GROUND_Z_MAX);
            logPerFrameDiagnostics(g_data.step, T_world, T_world, nullptr, 0.0);
            continue;
        }

        TicToc t_register;
        Transf T_guess = getOdom();
        const double delta_guess_vs_last = (g_data.step >= 2)
            ? poseDeltaNorm(g_data.T_seq[g_data.step - 1], T_guess) : 0.0;

        // 为配准生成 range image（近/远层），与 runMapBuild 同参数
        {
            const double split = param.range_image_split;
            const bool use_far = (split > 0.0);
            const double far_z_floor = farLayerZFloor(GROUND_Z_MIN);
            range_proc.generateRangeImage(scan_local, GROUND_Z_MIN, GROUND_Z_MAX, false,
                                          0.0, use_far ? split : 1e9, GROUND_Z_MIN);
            if (use_far)
                range_proc_far.generateRangeImage(scan_local, GROUND_Z_MIN, GROUND_Z_MAX, false,
                                                  split, 1e9, far_z_floor);
        }

        RegFrameDiag reg_diag;
        T_world = registerScanToMap(scan_local, T_guess, bspline_map, ground_grid,
                                    max_rg_iters, converge_thr, match_dist_thr, skip_points,
                                    GROUND_Z_MIN, GROUND_Z_MIN, GROUND_Z_MAX,
                                    range_proc, range_proc_far, (param.range_image_split > 0.0),
                                    &reg_diag);

        g_data.updatePose(T_world);
        pubTf();
        logPerFrameDiagnostics(g_data.step, T_world, T_guess, &reg_diag, delta_guess_vs_last);

        std::cout << "  Registration: " << t_register.toc() << " ms | Pose: "
                  << T_world(0,3) << " " << T_world(1,3) << " " << T_world(2,3) << std::endl;

        if (traj_file.is_open()) {
            traj_file << T_world(0,3) << " "
                      << T_world(1,3) << " "
                      << T_world(2,3) << "\n";
            traj_file.flush();
        }

        runMapBuild(scan_local, T_world, range_proc, range_proc_far, bspline_map, ground_grid, match_dist_thr, GROUND_Z_MIN, GROUND_Z_MAX);

        path_pub.publish(g_data.path);
        std::cout << "===STEP " << g_data.step << "=== Total: " << t_step.toc() << " ms===" << std::endl;
        std::cout.flush();
    }

    std::cout << "Process finished. Total time: " << t_whole.toc() / 1000.0 << " s" << std::endl;

    printMapSummary(bspline_map);
    saveGroundGridZ(ground_grid);
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

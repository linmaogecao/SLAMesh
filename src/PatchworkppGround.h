#ifndef SLAMESH_PATCHWORKPP_GROUND_H
#define SLAMESH_PATCHWORKPP_GROUND_H

// Patchwork++ 地面分割（Lee et al., IROS 2022）移植。
// 对照实现：patchwork-plusplus-ros/include/patchworkpp/patchworkpp.hpp
//
// 与原版的差异，仅此三处：
//   1. 去掉 ROS / jsk_recognition_msgs 依赖与全部可视化分支；
//   2. 输入点的原始下标经 PointXYZL::label 穿过同心圆环模型，输出为按输入点云下标
//      对齐的 mask，使建图链与配准链能共用同一份判定；
//   3. RNR 需要 intensity，estimateGround 的 intensity 入参为空时自动跳过该步
//      （SLAMesh 的 scan_local 是 pcl::PointXYZ，不携带 intensity）。
// 算法逻辑（CZM / R-VPF / R-GPF / A-GLE / TGR）与阈值判定顺序与原版逐行一致。

#include <Eigen/Dense>
#include <pcl/common/centroid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 默认值取自 patchwork-plusplus-ros/config/params.yaml
struct PatchworkppConfig {
    double sensor_height{1.723};
    int    num_iter{3};        // 平面拟合迭代次数
    int    num_lpr{20};        // 参与最低点代表（LPR）均值的点数上限
    int    num_min_pts{10};    // patch 内少于此数直接判非地面
    double th_seeds{0.3};      // 地面种子高度阈值（相对 LPR）
    double th_dist{0.125};     // 地面厚度阈值
    double th_seeds_v{0.25};   // 竖直结构种子高度阈值
    double th_dist_v{0.1};     // 竖直结构厚度阈值
    double max_range{80.0};
    double min_range{2.7};
    double uprightness_thr{0.707};
    double adaptive_seed_selection_margin{-1.2};
    int    max_flatness_storage{1000};
    int    max_elevation_storage{1000};
    bool   enable_RNR{true};   // 仅在调用方提供 intensity 时生效
    bool   enable_RVPF{true};  // Region-wise Vertical Plane Fitting
    bool   enable_TGR{true};   // Temporal Ground Revert
    double RNR_ver_angle_thr{-15.0};
    double RNR_intensity_thr{0.2};

    // 同心圆环模型（CZM）：固定 4 个 zone，min_ranges 的划分公式与之绑定
    std::vector<int>    num_sectors_each_zone{16, 32, 54, 32};
    std::vector<int>    num_rings_each_zone{2, 4, 4, 4};
    std::vector<double> elevation_thresholds{0.0, 0.0, 0.0, 0.0};
    std::vector<double> flatness_thresholds{0.0, 0.0, 0.0, 0.0};
};

class PatchworkppGround {
public:
    using PointL = pcl::PointXYZL;   // label 承载输入点云下标
    using Patch  = pcl::PointCloud<PointL>;
    using Ring   = std::vector<Patch>;
    using Zone   = std::vector<Ring>;

    static constexpr int kNumZones = 4;

    PatchworkppGround() { reset(PatchworkppConfig{}); }
    explicit PatchworkppGround(const PatchworkppConfig& cfg) { reset(cfg); }

    void reset(const PatchworkppConfig& cfg) {
        cfg_ = cfg;
        if ((int)cfg_.num_sectors_each_zone.size() != kNumZones ||
            (int)cfg_.num_rings_each_zone.size()   != kNumZones) {
            const PatchworkppConfig d;
            cfg_.num_sectors_each_zone = d.num_sectors_each_zone;
            cfg_.num_rings_each_zone   = d.num_rings_each_zone;
        }
        if (cfg_.elevation_thresholds.size() != cfg_.flatness_thresholds.size() ||
            cfg_.elevation_thresholds.empty()) {
            const PatchworkppConfig d;
            cfg_.elevation_thresholds = d.elevation_thresholds;
            cfg_.flatness_thresholds  = d.flatness_thresholds;
        }

        sensor_height_ = cfg_.sensor_height;
        elevation_thr_ = cfg_.elevation_thresholds;
        flatness_thr_  = cfg_.flatness_thresholds;

        // 阈值表按 concentric_idx 索引，长度不得超过总环数
        const int total_rings = std::accumulate(cfg_.num_rings_each_zone.begin(),
                                                cfg_.num_rings_each_zone.end(), 0);
        num_rings_of_interest_ = std::min((int)elevation_thr_.size(), total_rings);
        elevation_thr_.resize(num_rings_of_interest_);
        flatness_thr_.resize(num_rings_of_interest_);
        update_elevation_.assign(num_rings_of_interest_, {});
        update_flatness_.assign(num_rings_of_interest_, {});

        const double r0 = cfg_.min_range, r_max = cfg_.max_range;
        min_ranges_ = {r0,
                       (7.0 * r0 + r_max) / 8.0,
                       (3.0 * r0 + r_max) / 4.0,
                       (r0 + r_max) / 2.0};
        ring_sizes_.resize(kNumZones);
        sector_sizes_.resize(kNumZones);
        for (int z = 0; z < kNumZones; ++z) {
            const double hi = (z + 1 < kNumZones) ? min_ranges_[z + 1] : r_max;
            ring_sizes_[z]   = (hi - min_ranges_[z]) / cfg_.num_rings_each_zone[z];
            sector_sizes_[z] = 2.0 * M_PI / cfg_.num_sectors_each_zone[z];
        }

        czm_.assign(kNumZones, Zone{});
        for (int z = 0; z < kNumZones; ++z)
            czm_[z].assign(cfg_.num_rings_each_zone[z],
                           Ring(cfg_.num_sectors_each_zone[z], Patch{}));
    }

    // ground_mask[i] != 0 表示 cloud_in[i] 被判为地面点。
    void estimateGround(const pcl::PointCloud<pcl::PointXYZ>& cloud_in,
                        std::vector<uint8_t>& ground_mask) {
        estimateGround(cloud_in, std::vector<float>{}, ground_mask);
    }

    // intensities 为空则跳过 RNR；非空时长度须与 cloud_in 一致。
    void estimateGround(const pcl::PointCloud<pcl::PointXYZ>& cloud_in,
                        const std::vector<float>& intensities,
                        std::vector<uint8_t>& ground_mask) {
        ground_mask.assign(cloud_in.size(), 0);
        if (cloud_in.empty()) return;

        // 1. Reflected Noise Removal (RNR)
        noise_.assign(cloud_in.size(), 0);
        const bool do_rnr = cfg_.enable_RNR && intensities.size() == cloud_in.size();
        if (do_rnr) {
            for (size_t i = 0; i < cloud_in.size(); ++i) {
                const auto& p = cloud_in.points[i];
                const double r = std::sqrt(p.x * p.x + p.y * p.y);
                const double ver_angle_in_deg = std::atan2(p.z, r) * 180.0 / M_PI;
                if (ver_angle_in_deg < cfg_.RNR_ver_angle_thr &&
                    p.z < -sensor_height_ - 0.8 &&
                    intensities[i] < cfg_.RNR_intensity_thr)
                    noise_[i] = 1;
            }
        }

        // 2. Concentric Zone Model (CZM)
        flushPatches();
        pc2czm(cloud_in);

        int concentric_idx = 0;
        std::vector<RevertCandidate> candidates;
        std::vector<double> ringwise_flatness;

        for (int zone_idx = 0; zone_idx < kNumZones; ++zone_idx) {
            for (int ring_idx = 0; ring_idx < cfg_.num_rings_each_zone[zone_idx]; ++ring_idx) {
                for (int sec_idx = 0; sec_idx < cfg_.num_sectors_each_zone[zone_idx]; ++sec_idx) {
                    Patch& patch = czm_[zone_idx][ring_idx][sec_idx];
                    if ((int)patch.points.size() < cfg_.num_min_pts) continue;

                    // region-wise sorting（原版按 z 升序）
                    std::sort(patch.points.begin(), patch.points.end(),
                              [](const PointL& a, const PointL& b) { return a.z < b.z; });

                    extractPiecewiseGround(zone_idx, patch);

                    const double ground_uprightness = normal_(2);
                    const double ground_elevation   = pc_mean_(2);
                    const double ground_flatness    = singular_values_.minCoeff();
                    const double line_variable      = (singular_values_(1) != 0.0f)
                            ? singular_values_(0) / singular_values_(1)
                            : std::numeric_limits<double>::max();
                    double heading = 0.0;
                    for (int i = 0; i < 3; ++i) heading += pc_mean_(i) * normal_(i);

                    const bool is_near_zone       = concentric_idx < num_rings_of_interest_;
                    const bool is_upright         = ground_uprightness > cfg_.uprightness_thr;
                    // 原版此处直接索引 elevation_thr_[concentric_idx]（远环越界读），
                    // 但越界值只在 is_near_zone 为真的分支被使用，故加 near 保护后等价
                    const bool is_not_elevated    = is_near_zone &&
                            ground_elevation < elevation_thr_[concentric_idx];
                    const bool is_flat            = is_near_zone &&
                            ground_flatness < flatness_thr_[concentric_idx];
                    const bool is_heading_outside = heading < 0.0;

                    // A-GLE / TGR 的统计量累积
                    if (is_upright && is_not_elevated && is_near_zone) {
                        update_elevation_[concentric_idx].push_back(ground_elevation);
                        update_flatness_[concentric_idx].push_back(ground_flatness);
                        ringwise_flatness.push_back(ground_flatness);
                    }

                    if (!is_upright) {
                        // 非地面
                    } else if (!is_near_zone) {
                        markGround(regionwise_ground_, ground_mask);
                    } else if (!is_heading_outside) {
                        // 非地面
                    } else if (is_not_elevated || is_flat) {
                        markGround(regionwise_ground_, ground_mask);
                    } else {
                        RevertCandidate c;
                        c.ground_flatness = ground_flatness;
                        c.line_variable   = line_variable;
                        c.indices.reserve(regionwise_ground_.points.size());
                        for (const auto& p : regionwise_ground_.points)
                            c.indices.push_back(p.label);
                        candidates.push_back(std::move(c));
                    }
                }

                if (!candidates.empty()) {
                    if (cfg_.enable_TGR)
                        temporalGroundRevert(ringwise_flatness, candidates,
                                             concentric_idx, ground_mask);
                    candidates.clear();
                    ringwise_flatness.clear();
                }
                ++concentric_idx;
            }
        }

        updateElevationThr();
        updateFlatnessThr();
    }

    const PatchworkppConfig& config()       const { return cfg_; }
    double                   sensorHeight() const { return sensor_height_; }

private:
    struct RevertCandidate {
        double ground_flatness{0.0};
        double line_variable{0.0};
        std::vector<uint32_t> indices;
    };

    void flushPatches() {
        for (int z = 0; z < kNumZones; ++z)
            for (auto& ring : czm_[z])
                for (auto& patch : ring) patch.points.clear();
    }

    void pc2czm(const pcl::PointCloud<pcl::PointXYZ>& src) {
        for (size_t i = 0; i < src.size(); ++i) {
            if (!noise_.empty() && noise_[i]) continue;
            const auto& pt = src.points[i];
            const double r = std::sqrt(pt.x * pt.x + pt.y * pt.y);
            if (r > cfg_.max_range || r <= cfg_.min_range) continue;

            double theta = std::atan2(pt.y, pt.x);
            if (theta <= 0.0) theta += 2.0 * M_PI;

            int z = kNumZones - 1;
            if      (r < min_ranges_[1]) z = 0;
            else if (r < min_ranges_[2]) z = 1;
            else if (r < min_ranges_[3]) z = 2;

            int ring = std::min((int)((r - min_ranges_[z]) / ring_sizes_[z]),
                                cfg_.num_rings_each_zone[z] - 1);
            int sec  = std::min((int)(theta / sector_sizes_[z]),
                                cfg_.num_sectors_each_zone[z] - 1);
            ring = std::max(ring, 0);
            sec  = std::max(sec, 0);

            PointL p;
            p.x = pt.x;
            p.y = pt.y;
            p.z = pt.z;
            p.label = static_cast<uint32_t>(i);
            czm_[z][ring][sec].points.push_back(p);
        }
    }

    static void markGround(const Patch& pts, std::vector<uint8_t>& mask) {
        for (const auto& p : pts.points)
            if (p.label < mask.size()) mask[p.label] = 1;
    }

    void estimatePlane(const Patch& ground) {
        pcl::computeMeanAndCovarianceMatrix(ground, cov_, pc_mean_);
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov_, Eigen::ComputeFullU);
        singular_values_ = svd.singularValues();
        normal_ = svd.matrixU().col(2);
        if (normal_(2) < 0.0f) normal_ = -normal_;
        d_ = -normal_.dot(pc_mean_.head<3>());
    }

    void extractInitialSeeds(int zone_idx, const Patch& p_sorted,
                             Patch& init_seeds, double th_seed) {
        init_seeds.points.clear();
        if (p_sorted.points.empty()) return;

        // zone 0 先跳过明显低于地面的点（反射噪声等）
        size_t init_idx = 0;
        if (zone_idx == 0) {
            const double z_floor = cfg_.adaptive_seed_selection_margin * sensor_height_;
            while (init_idx < p_sorted.points.size() &&
                   p_sorted.points[init_idx].z < z_floor)
                ++init_idx;
        }

        double sum = 0.0;
        int cnt = 0;
        for (size_t i = init_idx; i < p_sorted.points.size() && cnt < cfg_.num_lpr; ++i) {
            sum += p_sorted.points[i].z;
            ++cnt;
        }
        const double lpr_height = (cnt != 0) ? sum / cnt : 0.0;

        for (const auto& p : p_sorted.points)
            if (p.z < lpr_height + th_seed) init_seeds.points.push_back(p);
    }

    // 结果写入 regionwise_ground_ / regionwise_nonground_，平面参数留在成员里供调用方读取
    void extractPiecewiseGround(int zone_idx, const Patch& src) {
        ground_pc_.points.clear();
        regionwise_ground_.points.clear();
        regionwise_nonground_.points.clear();
        src_wo_verticals_ = src;

        // 1. R-VPF：剔除地面之下的竖直结构
        if (cfg_.enable_RVPF) {
            for (int i = 0; i < cfg_.num_iter; ++i) {
                extractInitialSeeds(zone_idx, src_wo_verticals_, ground_pc_, cfg_.th_seeds_v);
                if (ground_pc_.points.size() < 3) break;
                estimatePlane(ground_pc_);

                if (zone_idx != 0 || normal_(2) >= cfg_.uprightness_thr) break;

                scratch_ = src_wo_verticals_;
                src_wo_verticals_.points.clear();
                for (const auto& p : scratch_.points) {
                    const float proj = normal_(0) * p.x + normal_(1) * p.y + normal_(2) * p.z;
                    if (proj < cfg_.th_dist_v - d_ && proj > -cfg_.th_dist_v - d_)
                        regionwise_nonground_.points.push_back(p);
                    else
                        src_wo_verticals_.points.push_back(p);
                }
            }
        }

        extractInitialSeeds(zone_idx, src_wo_verticals_, ground_pc_, cfg_.th_seeds);
        if (ground_pc_.points.size() < 3) {
            // 种子不足以定平面：整块判非地面。法向置零使调用方的 uprightness 检查必然失败
            regionwise_nonground_.points.insert(regionwise_nonground_.points.end(),
                                                src_wo_verticals_.points.begin(),
                                                src_wo_verticals_.points.end());
            normal_.setZero();
            singular_values_.setZero();
            pc_mean_.setZero();
            return;
        }
        estimatePlane(ground_pc_);

        // 2. R-GPF：迭代拟合地面
        for (int i = 0; i < cfg_.num_iter; ++i) {
            const bool final_iter = (i == cfg_.num_iter - 1);
            ground_pc_.points.clear();
            for (const auto& p : src_wo_verticals_.points) {
                const float proj = normal_(0) * p.x + normal_(1) * p.y + normal_(2) * p.z;
                const bool is_gnd = (proj < cfg_.th_dist - d_);
                if (!final_iter) {
                    if (is_gnd) ground_pc_.points.push_back(p);
                } else if (is_gnd) {
                    regionwise_ground_.points.push_back(p);
                } else {
                    regionwise_nonground_.points.push_back(p);
                }
            }
            const Patch& fit = final_iter ? regionwise_ground_ : ground_pc_;
            if (fit.points.size() >= 3) estimatePlane(fit);
        }
    }

    void temporalGroundRevert(const std::vector<double>& ring_flatness,
                              const std::vector<RevertCandidate>& candidates,
                              int concentric_idx,
                              std::vector<uint8_t>& mask) {
        if (concentric_idx >= num_rings_of_interest_) return;

        double mean_flatness = 0.0, stdev_flatness = 0.0;
        calcMeanStdev(ring_flatness, mean_flatness, stdev_flatness);

        const double mu_flatness = mean_flatness + 1.5 * stdev_flatness;
        const double denom = std::max(mu_flatness / 10.0, 1e-9);

        for (const auto& c : candidates) {
            double prob_flatness =
                    1.0 / (1.0 + std::exp((c.ground_flatness - mu_flatness) / denom));
            if (c.indices.size() > 1500 && c.ground_flatness < cfg_.th_dist * cfg_.th_dist)
                prob_flatness = 1.0;
            const double prob_line = (c.line_variable > 8.0) ? 0.0 : 1.0;
            if (prob_line * prob_flatness > 0.5)
                for (uint32_t idx : c.indices)
                    if (idx < mask.size()) mask[idx] = 1;
        }
    }

    void updateElevationThr() {
        for (int i = 0; i < num_rings_of_interest_; ++i) {
            if (update_elevation_[i].empty()) continue;
            double mean = 0.0, stdev = 0.0;
            calcMeanStdev(update_elevation_[i], mean, stdev);
            if (i == 0) {
                elevation_thr_[i] = mean + 3.0 * stdev;
                sensor_height_ = -mean;
            } else {
                elevation_thr_[i] = mean + 2.0 * stdev;
            }
            const int excess = (int)update_elevation_[i].size() - cfg_.max_elevation_storage;
            if (excess > 0)
                update_elevation_[i].erase(update_elevation_[i].begin(),
                                           update_elevation_[i].begin() + excess);
        }
    }

    void updateFlatnessThr() {
        for (int i = 0; i < num_rings_of_interest_; ++i) {
            if (update_flatness_[i].size() <= 1) break;
            double mean = 0.0, stdev = 0.0;
            calcMeanStdev(update_flatness_[i], mean, stdev);
            flatness_thr_[i] = mean + stdev;
            const int excess = (int)update_flatness_[i].size() - cfg_.max_flatness_storage;
            if (excess > 0)
                update_flatness_[i].erase(update_flatness_[i].begin(),
                                          update_flatness_[i].begin() + excess);
        }
    }

    static void calcMeanStdev(const std::vector<double>& v, double& mean, double& stdev) {
        mean = 0.0;
        stdev = 0.0;
        if (v.size() <= 1) return;
        mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        double acc = 0.0;
        for (double x : v) acc += (x - mean) * (x - mean);
        stdev = std::sqrt(acc / (v.size() - 1));
    }

    PatchworkppConfig cfg_;

    double sensor_height_{1.723};
    int    num_rings_of_interest_{4};

    std::vector<double> min_ranges_, ring_sizes_, sector_sizes_;
    std::vector<double> elevation_thr_, flatness_thr_;
    std::vector<std::vector<double>> update_elevation_, update_flatness_;

    std::vector<Zone> czm_;
    std::vector<uint8_t> noise_;

    Patch ground_pc_, regionwise_ground_, regionwise_nonground_, src_wo_verticals_, scratch_;

    Eigen::Matrix3f cov_{Eigen::Matrix3f::Zero()};
    Eigen::Vector4f pc_mean_{Eigen::Vector4f::Zero()};
    Eigen::Vector3f normal_{Eigen::Vector3f::Zero()};
    Eigen::Vector3f singular_values_{Eigen::Vector3f::Zero()};
    float d_{0.0f};
};

#endif  // SLAMESH_PATCHWORKPP_GROUND_H

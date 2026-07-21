//
// Created by albus on 2026/2/3.
//

#ifndef SPLINE_FITTING_RANGEMAP_H
#define SPLINE_FITTING_RANGEMAP_H

#include <iostream>
#include <array>
#include <fstream>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <opencv4/opencv2/opencv.hpp>
#include <Eigen/Eigenvalues>
#include <opencv2/opencv.hpp>
#include "GroundMatchPolicy.h"

struct RangePixel {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double range = 0.0;
    bool valid = false;
    std::pair<double, double> curvature;
    bool is_visted = false;
};

struct SegmentationResult {
    std::vector<int> label_map;             // 全图标签
    std::vector<std::vector<int>> clusters; // 每个面的索引集合
};

struct GroundExtractDebugStats {
    int input_valid_points = 0;
    int seed_cols = 0;
    int accepted_col_points = 0;
    int cut_by_zmax = 0;
    int cut_by_step = 0;
    int cell_count = 0;
    int after_cell_filter = 0;
    int cell_cut_total = 0;  // Step-3 被格内 z 容差砍掉的点数
    std::array<int, ground_match_policy::kDistanceBinCount> col_bin_counts{};
    std::array<int, ground_match_policy::kDistanceBinCount> final_bin_counts{};
    std::array<int, ground_match_policy::kDistanceBinCount> cell_cut_bin_counts{}; // 格过滤砍点按距离 bin
    std::array<int, ground_match_policy::kDistanceBinCount> cut_zmax_bins{};       // 触发 z_max 整列停止的点所在 bin
    std::array<int, ground_match_policy::kDistanceBinCount> cut_step_bins{};       // 触发 |Δz| 断链（开新段）的点所在 bin
};

// ---------- 体素化分割结果存储 ----------
// 设计:
//   - 每个 cluster 跨多个 voxel 时, 会按 voxel 拆分成多个 "SubCluster"
//   - 每个 SubCluster 记录它来自的全局 cluster_id, 反查就能知道一个 cluster 跨了哪些 voxel
//   - 一个 voxel 内可能有多个 SubCluster (来自不同的 cluster_id)
struct VoxelKey {
    int x = 0, y = 0, z = 0;
    bool operator==(const VoxelKey& o) const { return x == o.x && y == o.y && z == o.z; }
};

struct VoxelKeyHash {
    size_t operator()(const VoxelKey& k) const {
        size_t h1 = std::hash<int>()(k.x);
        size_t h2 = std::hash<int>()(k.y);
        size_t h3 = std::hash<int>()(k.z);
        return h1 ^ (h2 * 73856093u) ^ (h3 * 83492791u);
    }
};

struct SubCluster {
    int cluster_id = -1;          // 来自哪个全局 cluster (SegmentationResult.clusters 的下标)
    VoxelKey voxel_key;           // 所在体素
    std::vector<int> indices;     // range_image_ 中的像素索引
};

struct VoxelCell {
    std::vector<int> sub_cluster_ids; // 指向 VoxelizedClusters.sub_clusters 的下标
};

struct VoxelizedClusters {
    double voxel_size = 1.0;
    std::vector<SubCluster> sub_clusters;                                  // 所有子聚类的扁平列表
    std::unordered_map<VoxelKey, VoxelCell, VoxelKeyHash> voxels;          // 体素 -> 该体素内的子聚类
    std::unordered_map<int, std::vector<int>> cluster_to_subclusters;      // cluster_id -> 它被拆出的 sub_cluster 下标
};


class RangeImageProcessor {
public:
    const int H_SCANS = 64;
    const int W_COLS = 1500;
    const float FOV_UP = 2.0f;
    const float FOV_DOWN = -24.8f;
    const double MIN_RANGE = 1.0;
    const double MAX_RANGE = 100.0;
    const double MIN_Z = -2.5;   // 低于此高度的点不进 range image (世界系/雷达系)
    double alpha_vert_rad_;
    double alpha_horiz_rad_;
    std::vector<RangePixel> range_image_;
    // pixel_to_cloud_idx_[pixel_idx] = 对应点云中的点下标（-1 表示该 pixel 无效）
    // 由 generateRangeImage 填充，与 range_image_ 同步
    std::vector<int> pixel_to_cloud_idx_;
    // 本次 generateRangeImage 中：已通过筛选但因同像素存在更近点而未写入 range image 的点（雷达系）
    std::vector<Eigen::Vector3d> occluded_points_;
    // ground_cloud_mask_[i] = true 表示点云第 i 个点被 extractGroundByCellFilter 判为地面
    std::vector<bool> ground_cloud_mask_;
    GroundExtractDebugStats ground_debug_stats_;
    pcl::PointCloud<pcl::PointXYZ> ouyt;
    RangeImageProcessor() {
        range_image_.resize(H_SCANS * W_COLS);
        pixel_to_cloud_idx_.resize(H_SCANS * W_COLS, -1);
        alpha_vert_rad_ = (FOV_UP * M_PI / 180.0f - FOV_DOWN * M_PI / 180.0f)/(H_SCANS - 1);
        alpha_horiz_rad_ = (2.0 * M_PI) / W_COLS;

    }
    void saveRangeImageBin(const std::string& filename);
    // -----------------------------------------------------------------
    // 2. 核心函数: PointCloud -> RangeImage
    // -----------------------------------------------------------------
    // exclude_ground_band=true 时，z∈[ground_z_min,ground_z_max] 的点不投影进 range image
    // layer_range_min/max：只投影 range∈[layer_range_min, layer_range_max) 的点（默认全范围）
    // z_floor：低于此值的点不进 range image（默认 MIN_Z=-2.5；传 -1e9 表示不做高度下限过滤）
    // exclude_mask：若非 null，则跳过 cloud[i] 中 (*exclude_mask)[i]==true 的点（优先级高于 z 带排除）
    void generateRangeImage(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                            double ground_z_min = -1e9, double ground_z_max = 1e9,
                            bool exclude_ground_band = false,
                            double layer_range_min = 0.0,
                            double layer_range_max = 1e9,
                            double z_floor = -2.5,
                            const std::vector<bool>* exclude_mask = nullptr);

    const std::vector<Eigen::Vector3d>& getOccludedPoints() const { return occluded_points_; }
    bool getPoint(int u, int v, Eigen::Vector3d& out_point) const {
        // 处理 V 方向 (水平) 的周期性
        v = v % W_COLS;
        v += (v < 0) * W_COLS;

        bool u_valid = (unsigned)u < (unsigned)H_SCANS;

        int idx = u * W_COLS + v;
        const auto& px = range_image_[idx];
        out_point << px.x, px.y, px.z;
        return u_valid & px.valid;
    }
    int wrapCol(int v) const {
        int nv = (v + W_COLS) % W_COLS;
        return nv;
    }

    // 地面点提取：全图 range image → 按列从底环向上传播
    //             → XY 格子内严格 z 分位过滤
    // 结果写入 ground_cloud_mask_（按点云下标），并返回地面点下标列表
    // z_max：雷达系硬性高度上限（如 -0.5）；仅 z>=z_max 时整列停止
    // z_min：硬性下限（如 -3.8）；低于则跳过该环，不停止
    // col_max_step：同列相邻环 z 变化上限（m）；超出只断连续性并开新段，不截断整列
    // cell_size：XY 格子边长；cell_z_pct：格内低分位；cell_z_tol：容差（m，应偏严）
    std::vector<int> extractGroundByCellFilter(
        const pcl::PointCloud<pcl::PointXYZ>& cloud,
        double z_min, double z_max,
        double col_max_step,
        double cell_size, double cell_z_pct, double cell_z_tol);

    // exclude_ground_band=true 时，z∈[ground_z_min,ground_z_max] 的像素不参与 BFS 聚类（仅障碍聚类）
    SegmentationResult segmentRangeImage(double theta_deg, double max_dist, int min_cluster_size,
                                         double ground_z_min = -1e9, double ground_z_max = 1e9,
                                         bool exclude_ground_band = false);

    // 分割后按 range-image (u,v) 网格对每个 cluster 做自适应均匀稀疏：
    // 像素数 <= min_pts 不抽稀；否则 2D 步长使保留量约 target_max（0=关闭）。
    // min_keep：抽稀后若少于该值则回退为原始 cluster。
    void downsampleClusters(SegmentationResult& result,
                            int min_pts,
                            int target_max,
                            int min_keep = 30) const;

    void saveClustersToTxt(const SegmentationResult& result, const std::string& folder_path);

    // 与 saveClustersToTxt 相同，但将每个点用 transform (4×4) 变换到目标坐标系后再写入。
    // 用于 range image 基于局部系生成后，把聚类保存到世界系下便于可视化。
    void saveClustersWorldToTxt(const SegmentationResult& result,
                                const std::string& folder_path,
                                const Eigen::Matrix4d& transform);
    bool findValidNeighborPt(int u, int v, const Eigen::Vector3d& center_pt, Eigen::Vector3d& neighbor_pt, bool is_vertical = false, int dir = 1) const;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> generateClusterClouds(const SegmentationResult& result);

    // 单 cluster 点云提取 (避免一次全量, SLAM 场景按需取)
    pcl::PointCloud<pcl::PointXYZ>::Ptr generateOneClusterCloud(const SegmentationResult& result, int cluster_id) const;

    // 查 cluster 占据的 voxel keys (复用 voxelize 结果, 不重新计算)
    std::vector<VoxelKey> getClusterOccupiedVoxels(const VoxelizedClusters& vc, int cluster_id) const;

    // ---------- 体素化 ----------
    // 把分割结果按 voxel_size 拆分成"子聚类". 每个跨多 voxel 的 cluster 会被拆成多个 SubCluster.
    // 跨体素聚类太小 (< min_subcluster_size) 的子块会被丢弃.
    VoxelizedClusters voxelizeClusters(const SegmentationResult& result,
                                       double voxel_size = 1.0,
                                       int min_subcluster_size = 5) const;

    // 把每个 SubCluster 输出到 txt: <folder>/voxel_<x>_<y>_<z>_cid<id>.txt
    void saveVoxelizedClustersToTxt(const VoxelizedClusters& vc, const std::string& folder_path, std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clouds) const;

    // 把每个 SubCluster 转成 PCL 点云, 顺序与 vc.sub_clusters 一致
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> generateSubClusterClouds(const VoxelizedClusters& vc) const;
    std::vector<Eigen::Vector3d> extractClusterBoundary3D(const SegmentationResult& result, int cluster_id,
    int K_ring) const;

    std::vector<Eigen::Vector3d> computeInitControlPoints(const SegmentationResult& result, int cluster_id,int num_u, int num_v, int smooth_window = 2) const;
    bool getPointCloud(pcl::PointCloud<pcl::PointXYZ> & pcl_got, double voxel_filter_size);
};


#endif //SPLINE_FITTING_RANGEMAP_H
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
#include <Eigen/Eigenvalues>

struct RangePixel {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double range = 0.0;
    bool valid = false;
};

struct SegmentationResult {
    std::vector<int> label_map;             // 全图标签
    std::vector<std::vector<int>> clusters; // 每个面的索引集合
};

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
    // 本次 generateRangeImage 中：已通过筛选但因同像素存在更近点而未写入 range image 的点（雷达系）。
    // 仅在 collect_occluded_ 打开时收集（默认关闭，只有 dump 路径需要）。
    std::vector<Eigen::Vector3d> occluded_points_;
    bool collect_occluded_ = false;
    RangeImageProcessor() {
        range_image_.resize(H_SCANS * W_COLS);
        pixel_to_cloud_idx_.resize(H_SCANS * W_COLS, -1);
        alpha_vert_rad_ = (FOV_UP * M_PI / 180.0f - FOV_DOWN * M_PI / 180.0f)/(H_SCANS - 1);
        alpha_horiz_rad_ = (2.0 * M_PI) / W_COLS;

    }
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
    // 遮挡点收集只服务 dump，默认关闭；打开后下一次 generateRangeImage 才开始记录
    void setCollectOccluded(bool on) { collect_occluded_ = on; }
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

    // exclude_ground_band=true 时，z∈[ground_z_min,ground_z_max] 的像素不参与 BFS 聚类（仅障碍聚类）
    SegmentationResult segmentRangeImage(double theta_deg, double max_dist, int min_cluster_size,
                                         double ground_z_min = -1e9, double ground_z_max = 1e9,
                                         bool exclude_ground_band = false,
                                         double normal_angle_deg = 0.0);

    // 分割后按 range-image (u,v) 网格对每个 cluster 做自适应均匀稀疏：
    // 像素数 <= min_pts 不抽稀；否则 2D 步长使保留量约 target_max（0=关闭）。
    // min_keep：抽稀后若少于该值则回退为原始 cluster。
    void downsampleClusters(SegmentationResult& result,
                            int min_pts,
                            int target_max,
                            int min_keep = 30) const;

    // 把每个点用 transform (4×4) 变换到目标坐标系后写入 <folder>/cluster_<i>.txt。
    // 用于 range image 基于局部系生成后，把聚类保存到世界系下便于可视化。
    void saveClustersWorldToTxt(const SegmentationResult& result,
                                const std::string& folder_path,
                                const Eigen::Matrix4d& transform);

    std::vector<Eigen::Vector3d> computeInitControlPoints(const SegmentationResult& result, int cluster_id,int num_u, int num_v, int smooth_window = 2) const;
    bool getPointCloud(pcl::PointCloud<pcl::PointXYZ> & pcl_got, double voxel_filter_size);
};


#endif //SPLINE_FITTING_RANGEMAP_H
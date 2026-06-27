//
// Created by albus on 2026/2/3.
//

#ifndef SPLINE_FITTING_RANGEMAP_H
#define SPLINE_FITTING_RANGEMAP_H

#include <iostream>
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
    void generateRangeImage(const pcl::PointCloud<pcl::PointXYZ>& cloud);
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
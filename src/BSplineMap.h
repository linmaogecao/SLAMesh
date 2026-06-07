//
// BSplineMap.h
// 全局 B 样条曲面地图管理。
//
// 核心设计：
//   - 每个拟合好的曲面对应一个 BSplineMapEntry，持有 surface_id 和曲面指针。
//   - 注册时直接遍历 cluster 原始点云确定占据体素（O(N_points)），
//     不遍历 getSamples()，速度快且与真实观测范围一致。
//   - voxel_index_[k] 记录所有与体素 k 关联的 surface_id，定位时反查用。
//
// 依赖：RangeMap.h 中已有的 VoxelKey / VoxelKeyHash。
//

#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <limits>
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include "BSpline.h"    // BSplineSurface
#include "RangeMap.h"   // VoxelKey, VoxelKeyHash

// -----------------------------------------------------------------------
// 单个曲面地图条目
// -----------------------------------------------------------------------
struct BSplineMapEntry {
    int                              surface_id = -1;
    std::shared_ptr<BSplineSurface>  surface;            // 拟合完成的曲面
    std::vector<VoxelKey>            occupied_voxels;    // 该曲面覆盖的体素（无重复）
};

// -----------------------------------------------------------------------
// 全局 B 样条曲面地图
// -----------------------------------------------------------------------
class BSplineMap {
public:
    explicit BSplineMap(double voxel_size = 1.0) : voxel_size_(voxel_size) {}

    // ------------------------------------------------------------------
    // 注册一个已拟合的曲面到地图。
    // 用 cluster 原始点云确定占据体素 —— O(N_points)，不遍历 getSamples()。
    // voxel_index_[k] 同时追加该 surface_id，供定位时反查。
    // 返回分配给该曲面的 surface_id。
    // ------------------------------------------------------------------
    int addSurface(std::shared_ptr<BSplineSurface>            surf,
                   const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster_cloud)
    {
        int sid = static_cast<int>(entries_.size());

        BSplineMapEntry entry;
        entry.surface_id = sid;
        entry.surface    = surf;

        // 遍历 cluster 原始点注册体素（去重），体素 -> 关联曲面列表
        std::unordered_set<VoxelKey, VoxelKeyHash> seen;
        if (cluster_cloud) {
            for (const auto& pt : cluster_cloud->points) {
                VoxelKey k = toVoxelKey(pt.x, pt.y, pt.z);
                if (seen.insert(k).second) {
                    entry.occupied_voxels.push_back(k);
                    voxel_index_[k].push_back(sid);
                }
            }
        }

        entries_.push_back(std::move(entry));
        return sid;
    }

    // ------------------------------------------------------------------
    // 查询候选曲面。
    // 给定地图系下的 3D 点 q，在 q 所在体素及其 (2*radius+1)^3 邻域内
    // 返回所有关联的 surface_id（去重）。
    // radius=1 → 3×3×3=27 个体素；radius=0 → 仅中心体素。
    // ------------------------------------------------------------------
    std::vector<int> queryCandidates(const Eigen::Vector3d& q, int radius = 1) const {
        VoxelKey center = toVoxelKey(q.x(), q.y(), q.z());
        std::unordered_set<int> seen;

        for (int dx = -radius; dx <= radius; ++dx)
        for (int dy = -radius; dy <= radius; ++dy)
        for (int dz = -radius; dz <= radius; ++dz) {
            VoxelKey k{center.x + dx, center.y + dy, center.z + dz};
            auto it = voxel_index_.find(k);
            if (it != voxel_index_.end())
                for (int sid : it->second) seen.insert(sid);
        }
        return {seen.begin(), seen.end()};
    }

    // 按 surface_id 取条目
    const BSplineMapEntry* getEntry(int sid) const {
        if (sid < 0 || sid >= (int)entries_.size()) return nullptr;
        return &entries_[sid];
    }

    int size() const { return static_cast<int>(entries_.size()); }

    const std::vector<BSplineMapEntry>& entries() const { return entries_; }

private:
    VoxelKey toVoxelKey(double x, double y, double z) const {
        VoxelKey k;
        k.x = static_cast<int>(std::floor(x / voxel_size_));
        k.y = static_cast<int>(std::floor(y / voxel_size_));
        k.z = static_cast<int>(std::floor(z / voxel_size_));
        return k;
    }

    double voxel_size_;
    std::vector<BSplineMapEntry>                                  entries_;
    std::unordered_map<VoxelKey, std::vector<int>, VoxelKeyHash>  voxel_index_;
};

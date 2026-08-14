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
#include <algorithm>
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
    bool                             is_ground = false;  // 是否是地面曲面
    int                              created_step = -1;  // 写入地图时的 g_data.step
    // 仍在 voxel_index_ 中挂着的体素数；被新面按体素抢占后递减，归零即该面不可能
    // 再被 queryCandidates 命中，此时释放曲面内存。
    int                              live_voxels = 0;
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
    // replace_frac > 0 时启用覆盖式替换：新面若把某张旧障碍面的占据体素覆盖到该比例
    // 以上，就退役那张旧面。没有这一步时同一面墙会被每次建图各自拟合一张，几百帧后
    // 单帧参与候选的曲面数涨 5~6 倍（seq06 实测 91 -> 570），而这些碎片是在逐渐漂移
    // 的位姿下拟合的、彼此位置不一致，扫描点只能在一堆互相矛盾的同墙碎片里挑最近。
    int addSurface(std::shared_ptr<BSplineSurface>            surf,
                   const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster_cloud,
                   bool is_ground = false,
                   int created_step = -1,
                   double replace_frac = 0.0,
                   bool voxel_owner = false)
    {
        int sid = static_cast<int>(entries_.size());

        BSplineMapEntry entry;
        entry.surface_id   = sid;
        entry.surface      = surf;
        entry.is_ground    = is_ground;
        entry.created_step = created_step;

        std::unordered_set<VoxelKey, VoxelKeyHash> seen;
        if (cluster_cloud) {
            for (const auto& pt : cluster_cloud->points) {
                VoxelKey k = toVoxelKey(pt.x, pt.y, pt.z);
                if (seen.insert(k).second) entry.occupied_voxels.push_back(k);
            }
        }

        // 判定用退役前的索引，各旧面之间互不影响，故退役集合与遍历顺序无关
        if (replace_frac > 0.0 && !is_ground) {
            std::unordered_map<int, int> overlap;
            for (const auto& k : entry.occupied_voxels) {
                auto it = voxel_index_.find(k);
                if (it == voxel_index_.end()) continue;
                for (int old_sid : it->second) ++overlap[old_sid];
            }
            for (const auto& [old_sid, n_shared] : overlap) {
                BSplineMapEntry& old = entries_[old_sid];
                if (old.is_ground || !old.surface || old.occupied_voxels.empty()) continue;
                // 分母取旧面自身的体素数：只在新观测把旧面基本盖住时才退役，
                // 不会因为一小片新面碰到大墙的一角就把大墙删掉
                if ((double)n_shared / (double)old.occupied_voxels.size() >= replace_frac)
                    retireSurface(old_sid);
            }
        }

        // 按体素论归属：新面占据的体素里把所有旧障碍面摘掉，扫描点在该体素只会拿到最新
        // 的一张。整片覆盖率判据抓不到重复，因为同一面墙每次观测到的是不同片段、互相
        // 覆盖率上不去；而"同一体素里塞着十几张漂移位姿下拟合的碎片"正是要消灭的东西。
        if (voxel_owner && !is_ground) {
            for (const auto& k : entry.occupied_voxels) {
                auto it = voxel_index_.find(k);
                if (it == voxel_index_.end()) continue;
                auto& v = it->second;
                const auto new_end = std::remove_if(v.begin(), v.end(), [&](int s) {
                    if (entries_[s].is_ground) return false;
                    if (--entries_[s].live_voxels <= 0) {
                        entries_[s].surface.reset();
                        ++n_retired_;
                    }
                    return true;
                });
                v.erase(new_end, v.end());
                if (v.empty()) voxel_index_.erase(it);
            }
        }

        entry.live_voxels = static_cast<int>(entry.occupied_voxels.size());
        for (const auto& k : entry.occupied_voxels) voxel_index_[k].push_back(sid);

        entries_.push_back(std::move(entry));
        return sid;
    }

    int retiredCount() const { return n_retired_; }

    int aliveSurfaceCount() const {
        int cnt = 0;
        for (const auto& e : entries_) if (e.surface) ++cnt;
        return cnt;
    }

    double getVoxelSize() const { return voxel_size_; }

    int groundSurfaceCount() const {
        int cnt = 0;
        for (const auto& e : entries_) if (e.is_ground) ++cnt;
        return cnt;
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
    // sid 就是 entries_ 下标，不能真删；摘掉体素索引 + 释放曲面即可让它不再被
    // queryCandidates 返回，各消费点都已有 !surface 的跳过分支。
    void retireSurface(int sid) {
        BSplineMapEntry& e = entries_[sid];
        for (const auto& k : e.occupied_voxels) {
            auto it = voxel_index_.find(k);
            if (it == voxel_index_.end()) continue;
            auto& v = it->second;
            v.erase(std::remove(v.begin(), v.end(), sid), v.end());
            if (v.empty()) voxel_index_.erase(it);
        }
        e.occupied_voxels.clear();
        e.occupied_voxels.shrink_to_fit();
        e.live_voxels = 0;
        e.surface.reset();
        ++n_retired_;
    }

    VoxelKey toVoxelKey(double x, double y, double z) const {
        VoxelKey k;
        k.x = static_cast<int>(std::floor(x / voxel_size_));
        k.y = static_cast<int>(std::floor(y / voxel_size_));
        k.z = static_cast<int>(std::floor(z / voxel_size_));
        return k;
    }

    double voxel_size_;
    int    n_retired_ = 0;
    std::vector<BSplineMapEntry>                                  entries_;
    std::unordered_map<VoxelKey, std::vector<int>, VoxelKeyHash>  voxel_index_;
};

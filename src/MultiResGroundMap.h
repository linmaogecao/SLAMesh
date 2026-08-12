// MultiResGroundMap.h
// 多分辨率 XY 地面图：N 层 GroundGridMap（从细到粗，layers_[0] 最细）
//
// 建图：同一世界系点云同时投所有层；各层独立拟合 BSpline。
//
// 查询（细优先 fallback）：
//   queryNearest(p) 从层 0 开始依次 queryNearestSurface，
//   找到第一个非 null 曲面即返回。
//   ⟹ 配准互斥自然成立：某点被细层命中后不再向粗层分配。
#pragma once
#include "GroundGridMap.h"
#include <vector>
#include <cassert>
#include <iostream>

struct MRGndLayerConfig {
    double cell_size     = 6.0;
    int    min_pts       = 80;
    int    num_cp        = 5;
    int    active_radius = 1;
    int    cell_max_pts  = 400;
    int    fit_max_pts   = 150;
    double cell_z_pct    = 0.0;
    double cell_z_tol    = 0.2;
    double normal_z_min  = 0.0;  // 法向 |z| 最小值（0=关闭）
    // 建图格内相对高度过滤总开关；false = 强制关闭（等价 cell_z_tol<=0），
    // 但保留 cell_z_tol 数值不动，便于 A/B 对比时来回切换而不用改数值。
    bool   z_filter_enable = true;
};

class MultiResGroundMap {
public:
    MultiResGroundMap() = default;

    explicit MultiResGroundMap(const std::vector<MRGndLayerConfig>& cfgs) {
        layers_.reserve(cfgs.size());
        for (const auto& c : cfgs)
            layers_.emplace_back(c.cell_size, c.min_pts, c.num_cp,
                                 c.active_radius, c.cell_max_pts, c.fit_max_pts,
                                 c.cell_z_pct,
                                 c.z_filter_enable ? c.cell_z_tol : -1.0,
                                 c.normal_z_min);
        per_layer_touched_.resize(layers_.size());
    }

    int numLayers() const { return static_cast<int>(layers_.size()); }

    // 建图：点云同时投所有层，每层独立分格
    void addPoints(const pcl::PointCloud<pcl::PointXYZ>& pts, int step) {
        for (int li = 0; li < (int)layers_.size(); ++li)
            per_layer_touched_[li] = layers_[li].addPoints(pts, step);
    }

    // 拟合：各层独立拟合上次 addPoints 触及的格，返回各层合计新增拟合数
    int refitAll(int n_threads = 1) {
        int total = 0;
        for (int li = 0; li < (int)layers_.size(); ++li)
            total += layers_[li].refitCells(per_layer_touched_[li], n_threads);
        return total;
    }

    // 细优先 fallback 查询
    // layer_idx（可选）: 命中层编号；-1 = 全未命中
    const BSplineSurface* queryNearest(const Eigen::Vector3d& p,
                                       int* layer_idx = nullptr) const {
        for (int li = 0; li < (int)layers_.size(); ++li) {
            const BSplineSurface* sp = layers_[li].queryNearestSurface(p);
            if (sp) {
                if (layer_idx) *layer_idx = li;
                return sp;
            }
        }
        if (layer_idx) *layer_idx = -1;
        return nullptr;
    }

    const GroundGridMap& layer(int i) const {
        assert(i >= 0 && i < (int)layers_.size());
        return layers_[static_cast<size_t>(i)];
    }
    GroundGridMap& layer(int i) {
        assert(i >= 0 && i < (int)layers_.size());
        return layers_[static_cast<size_t>(i)];
    }

    int totalFitted() const {
        int n = 0;
        for (const auto& L : layers_) n += L.fittedCells();
        return n;
    }
    int totalCells() const {
        int n = 0;
        for (const auto& L : layers_) n += L.totalCells();
        return n;
    }

    void printStats(const std::string& label = "") const {
        if (!label.empty()) std::cout << label << " ";
        for (int li = 0; li < (int)layers_.size(); ++li) {
            std::cout << "[L" << li
                      << " cs=" << layers_[li].cellSize() << "m"
                      << " fitted=" << layers_[li].fittedCells()
                      << "/" << layers_[li].totalCells() << "]";
            if (li + 1 < (int)layers_.size()) std::cout << " ";
        }
        std::cout << "\n";
    }

private:
    std::vector<GroundGridMap> layers_;
    std::vector<std::vector<GroundCellKey>> per_layer_touched_;
};

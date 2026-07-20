// GroundGridMap.h
// XY 2D 栅格地面地图。
//
// 设计思路：
//   世界系 XY 面以固定分辨率 cell_size 划格，每个格子独立维护：
//     - 落在其中的地面点（世界系 xyz）
//     - 拟合完成的局部 BSpline 曲面（可为 nullptr）
//     - 最后更新的 step，用于滑动窗口清理
//
// 地面点分 cluster（格）流程：
//   1. 当前帧地面点（雷达系）→ 变换到世界系
//   2. 按 (floor(x/cell_size), floor(y/cell_size)) 投格
//   3. 格内相对高度过滤（均值/分位 + 上容差），再封顶下采样
//   4. 达到 min_pts 后标记 needs_refit，refitCells 拟合 BSpline
//
// 配准查询：
//   给定世界系点 p，取 (ix,iy) 及其 ±radius 格内所有有效曲面作为候选。
//   只查当前 step 附近的格（滑动窗口）。
//
#pragma once

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <memory>
#include <limits>
#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <omp.h>
#include "BSpline.h"   // BSplineSurface, SurfaceCurvature

// -----------------------------------------------------------------------
// XY 格子 key（只用 ix, iy，z 方向不划格）
// -----------------------------------------------------------------------
struct GroundCellKey {
    int ix = 0, iy = 0;
    bool operator==(const GroundCellKey& o) const { return ix == o.ix && iy == o.iy; }
};

struct GroundCellKeyHash {
    size_t operator()(const GroundCellKey& k) const {
        return std::hash<int>()(k.ix) ^ (std::hash<int>()(k.iy) * 2654435761u);
    }
};

// -----------------------------------------------------------------------
// 单个地面格子
// -----------------------------------------------------------------------
struct GroundCell {
    pcl::PointCloud<pcl::PointXYZ> pts;   // 积累的世界系地面点（未采样原始）
    std::shared_ptr<BSplineSurface> surf;  // 拟合好的局部曲面（nullptr = 未拟合）
    int last_update_step = -1;            // 最后有新点进来的 step
    bool needs_refit = false;             // pts 发生变化，需重新拟合
};

// -----------------------------------------------------------------------
// XY 2D 栅格地面地图
// -----------------------------------------------------------------------
class GroundGridMap {
public:
    // cell_size   : XY 栅格边长（米），建议 4–8 m
    // min_pts     : 格内点数达到此值才触发拟合
    // num_cp      : BSpline 每维控制点数
    // active_radius : 配准查询时格坐标半径（格数）
    // cell_max_pts: 每格最多保留点数（超出均匀下采样）
    // fit_max_pts : 拟合 BSpline 时最多使用的点数
    // cell_z_pct  : (0,1] 时用格内 z 分位作参考；<=0 用均值
    // cell_z_tol  : 相对参考高度上容差（米）；<=0 关闭过滤
    explicit GroundGridMap(double cell_size   = 6.0,
                           int    min_pts     = 80,
                           int    num_cp      = 5,
                           int    active_radius = 6,
                           int    cell_max_pts = 400,
                           int    fit_max_pts  = 150,
                           double cell_z_pct  = 0.0,
                           double cell_z_tol  = 0.0)
        : cell_size_(cell_size),
          min_pts_(min_pts),
          num_cp_(num_cp),
          active_radius_(active_radius),
          cell_max_pts_(std::max(1, cell_max_pts)),
          fit_max_pts_(std::max(1, fit_max_pts)),
          cell_z_pct_(cell_z_pct),
          cell_z_tol_(cell_z_tol)
    {}

    // -----------------------------------------------------------------------
    // 把一帧地面点（世界系）投格，按 XY 分组写入对应 GroundCell
    // 投格后可按格内平均/分位高度砍掉偏高点，再封顶下采样
    // 返回新增/更新的格 key 列表
    // -----------------------------------------------------------------------
    std::vector<GroundCellKey> addPoints(
        const pcl::PointCloud<pcl::PointXYZ>& pts_world,
        int current_step)
    {
        std::unordered_set<GroundCellKey, GroundCellKeyHash> touched;
        for (const auto& pt : pts_world) {
            GroundCellKey k = toCellKey(pt.x, pt.y);
            auto& cell = cells_[k];
            cell.pts.push_back(pt);
            cell.last_update_step = current_step;
            cell.needs_refit = true;
            touched.insert(k);
        }
        for (const auto& k : touched) {
            auto it = cells_.find(k);
            if (it == cells_.end()) continue;
            filterCellByRelativeZ(it->second.pts, cell_z_pct_, cell_z_tol_);
            capPointsInPlace(it->second.pts, cell_max_pts_);
        }
        return {touched.begin(), touched.end()};
    }

    // -----------------------------------------------------------------------
    // 对所有 needs_refit 的格执行 BSpline 拟合（或重拟）
    // 点数不足 min_pts 的格跳过（曲面保持 nullptr 或上次的旧曲面）
    // -----------------------------------------------------------------------
    int refitDirty()
    {
        int n_fit = 0;
        for (auto& [k, cell] : cells_) {
            if (!cell.needs_refit) continue;
            if (refitCell(k, cell)) ++n_fit;
        }
        return n_fit;
    }

    // 仅重拟合本帧触及的格（OMP 并行 apply，串行写回曲面）
    int refitCells(const std::vector<GroundCellKey>& keys, int n_threads = 1)
    {
        // A: 串行收集需要拟合的格及下采样点云
        struct GndTask {
            GroundCellKey key;
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
            std::shared_ptr<BSplineSurface> surf;
            bool ok = false;
        };
        std::vector<GndTask> tasks;
        tasks.reserve(keys.size());
        for (const auto& k : keys) {
            auto it = cells_.find(k);
            if (it == cells_.end() || !it->second.needs_refit) continue;
            const int n = (int)it->second.pts.size();
            if (n < min_pts_) continue;
            GndTask t;
            t.key   = k;
            t.cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
            subsampleToCloud(it->second.pts, fit_max_pts_, *t.cloud);
            tasks.push_back(std::move(t));
        }

        // B: OMP 并行 apply
        const int npar = std::max(1, n_threads);
#pragma omp parallel for schedule(dynamic) num_threads(npar)
        for (int i = 0; i < (int)tasks.size(); ++i) {
            auto& t = tasks[i];
            t.surf = std::make_shared<BSplineSurface>(3, 3, num_cp_, num_cp_, 0.25);
            t.ok = t.surf->apply(t.cloud, 30, 1, 1, 0.05);
        }

        // C: 串行写回（不同 key 理论可并行，但 unordered_map 写不安全）
        int n_fit = 0;
        for (auto& t : tasks) {
            auto it = cells_.find(t.key);
            if (it == cells_.end()) continue;
            it->second.needs_refit = false;
            if (!t.ok) continue;
            it->second.surf = std::move(t.surf);
            ++n_fit;
        }
        return n_fit;
    }

    // 配准：在候选曲面中取 footprint 距离最近的一张
    const BSplineSurface* queryNearestSurface(const Eigen::Vector3d& p) const
    {
        const BSplineSurface* best = nullptr;
        double best_d = std::numeric_limits<double>::max();
        for (const BSplineSurface* sp : querySurfaces(p)) {
            std::vector<Eigen::Vector3d> batch{p};
            std::vector<std::pair<BSplineSurface::Parameter, BSplineSurface::Parameter>> fps;
            std::vector<double> dists;
            const_cast<BSplineSurface*>(sp)->findFootPrint(batch, fps, dists);
            if (dists.empty()) continue;
            double d = std::sqrt(std::abs(dists[0]));
            if (d < best_d) { best_d = d; best = sp; }
        }
        return best;
    }

    // -----------------------------------------------------------------------
    // 配准查询：给定世界系点 p，返回附近格的曲面列表（非 nullptr）
    // 只搜 ±active_radius_ 格范围（2D，z 不限）
    // -----------------------------------------------------------------------
    std::vector<const BSplineSurface*> querySurfaces(
        const Eigen::Vector3d& p) const
    {
        std::vector<const BSplineSurface*> result;
        GroundCellKey center = toCellKey(p.x(), p.y());
        for (int di = -active_radius_; di <= active_radius_; ++di)
        for (int dj = -active_radius_; dj <= active_radius_; ++dj) {
            GroundCellKey k{center.ix + di, center.iy + dj};
            auto it = cells_.find(k);
            if (it == cells_.end()) continue;
            if (it->second.surf) result.push_back(it->second.surf.get());
        }
        return result;
    }

    // 同上，但返回 (BSplineSurface*, cell_key) 对，便于外部区分
    struct CandEntry {
        const BSplineSurface* surf;
        GroundCellKey key;
    };
    std::vector<CandEntry> queryCandidates(const Eigen::Vector3d& p) const {
        std::vector<CandEntry> result;
        GroundCellKey center = toCellKey(p.x(), p.y());
        for (int di = -active_radius_; di <= active_radius_; ++di)
        for (int dj = -active_radius_; dj <= active_radius_; ++dj) {
            GroundCellKey k{center.ix + di, center.iy + dj};
            auto it = cells_.find(k);
            if (it == cells_.end()) continue;
            if (it->second.surf) result.push_back({it->second.surf.get(), k});
        }
        return result;
    }

    // -----------------------------------------------------------------------
    // 删除距车辆超过 max_dist 米（XY 曼哈顿格数）的旧格，释放内存
    // 通常每隔几十帧调用一次
    // -----------------------------------------------------------------------
    int removeDistant(const Eigen::Vector3d& vehicle_pos, double max_dist) {
        GroundCellKey center = toCellKey(vehicle_pos.x(), vehicle_pos.y());
        int radius_cells = static_cast<int>(std::ceil(max_dist / cell_size_));
        std::vector<GroundCellKey> to_erase;
        for (const auto& [k, cell] : cells_) {
            if (std::abs(k.ix - center.ix) > radius_cells ||
                std::abs(k.iy - center.iy) > radius_cells)
                to_erase.push_back(k);
        }
        for (const auto& k : to_erase) cells_.erase(k);
        return static_cast<int>(to_erase.size());
    }

    // 统计
    int totalCells() const { return static_cast<int>(cells_.size()); }
    int fittedCells() const {
        int n = 0;
        for (const auto& [k, c] : cells_) if (c.surf) ++n;
        return n;
    }

    // 遍历所有已拟合格（用于 all_surfaces 导出）
    const std::unordered_map<GroundCellKey, GroundCell, GroundCellKeyHash>& cells() const {
        return cells_;
    }

    double cellSize() const { return cell_size_; }

    // 全图地面高度统计（世界系 z）
    struct HeightStats {
        int total_cells = 0;
        int fitted_cells = 0;
        int total_pts = 0;
        double pts_z_mean = 0;      // 所有格内点的 z 加权平均
        double pts_z_min = 0;
        double pts_z_max = 0;
        double cell_z_mean = 0;     // 每格点云 z 均值的平均（每格等权）
        double cell_z_min = 0;      // 各格 z 均值的最小
        double cell_z_max = 0;      // 各格 z 均值的最大
        double surf_cp_z_mean = 0;  // 已拟合格：控制点 z 均值的平均
        double surf_cp_z_min = 0;
        double surf_cp_z_max = 0;
    };

    HeightStats computeHeightStats() const
    {
        HeightStats s;
        s.total_cells = totalCells();
        s.fitted_cells = fittedCells();

        double pts_z_sum = 0;
        bool any_pts = false;
        bool any_cell_mean = false;
        bool any_surf = false;

        for (const auto& [k, cell] : cells_) {
            (void)k;
            if (cell.pts.empty()) continue;

            double cell_sum = 0;
            for (const auto& pt : cell.pts) {
                cell_sum += pt.z;
                pts_z_sum += pt.z;
                ++s.total_pts;
                if (!any_pts) {
                    s.pts_z_min = s.pts_z_max = pt.z;
                    any_pts = true;
                } else {
                    s.pts_z_min = std::min(s.pts_z_min, static_cast<double>(pt.z));
                    s.pts_z_max = std::max(s.pts_z_max, static_cast<double>(pt.z));
                }
            }

            const double cell_mean = cell_sum / static_cast<double>(cell.pts.size());
            if (!any_cell_mean) {
                s.cell_z_min = s.cell_z_max = cell_mean;
                any_cell_mean = true;
            } else {
                s.cell_z_min = std::min(s.cell_z_min, cell_mean);
                s.cell_z_max = std::max(s.cell_z_max, cell_mean);
            }
            s.cell_z_mean += cell_mean;

            if (!cell.surf) continue;
            const auto& cps = cell.surf->getControls();
            if (cps.empty()) continue;

            double cp_sum = 0;
            for (const auto& cp : cps) cp_sum += cp.z();
            const double cp_mean = cp_sum / static_cast<double>(cps.size());
            if (!any_surf) {
                s.surf_cp_z_min = s.surf_cp_z_max = cp_mean;
                any_surf = true;
            } else {
                s.surf_cp_z_min = std::min(s.surf_cp_z_min, cp_mean);
                s.surf_cp_z_max = std::max(s.surf_cp_z_max, cp_mean);
            }
            s.surf_cp_z_mean += cp_mean;
        }

        if (s.total_pts > 0)
            s.pts_z_mean = pts_z_sum / static_cast<double>(s.total_pts);
        if (s.total_cells > 0)
            s.cell_z_mean /= static_cast<double>(s.total_cells);
        if (s.fitted_cells > 0)
            s.surf_cp_z_mean /= static_cast<double>(s.fitted_cells);

        return s;
    }

private:
    // 格内相对高度过滤：砍掉明显高于参考高度的点（障碍抬高地面）
    // pct in (0,1] → 用该分位作 z_ref；否则用均值
    // 保留 z <= z_ref + tol；过滤后过少则回退，避免空格
    static void filterCellByRelativeZ(pcl::PointCloud<pcl::PointXYZ>& pts,
                                      double pct,
                                      double tol)
    {
        if (tol <= 0.0) return;
        const int n = static_cast<int>(pts.size());
        if (n < 3) return;

        std::vector<float> zs;
        zs.reserve(static_cast<size_t>(n));
        for (const auto& p : pts) zs.push_back(p.z);

        double z_ref = 0.0;
        if (pct > 0.0 && pct <= 1.0) {
            const size_t k = static_cast<size_t>(
                std::min(n - 1, std::max(0, static_cast<int>(std::floor(pct * (n - 1))))));
            std::nth_element(zs.begin(), zs.begin() + static_cast<std::ptrdiff_t>(k), zs.end());
            z_ref = zs[k];
        } else {
            double sum = 0.0;
            for (float z : zs) sum += z;
            z_ref = sum / static_cast<double>(n);
        }

        const float z_cut = static_cast<float>(z_ref + tol);
        pcl::PointCloud<pcl::PointXYZ> kept;
        kept.reserve(static_cast<size_t>(n));
        for (const auto& p : pts) {
            if (p.z <= z_cut) kept.push_back(p);
        }
        // 过滤后太少则回退，避免格被掏空无法拟合
        const int min_keep = std::max(3, n / 5);
        if (static_cast<int>(kept.size()) >= min_keep)
            pts.swap(kept);
    }

    static void capPointsInPlace(pcl::PointCloud<pcl::PointXYZ>& pts, int max_pts)
    {
        const int n = static_cast<int>(pts.size());
        if (n <= max_pts) return;
        pcl::PointCloud<pcl::PointXYZ> out;
        out.reserve(static_cast<size_t>(max_pts));
        const int step = std::max(1, (n + max_pts - 1) / max_pts);
        for (int i = 0; i < n && static_cast<int>(out.size()) < max_pts; i += step)
            out.push_back(pts[static_cast<size_t>(i)]);
        pts.swap(out);
    }

    static void subsampleToCloud(const pcl::PointCloud<pcl::PointXYZ>& src,
                                 int max_pts,
                                 pcl::PointCloud<pcl::PointXYZ>& dst)
    {
        dst.clear();
        const int n = static_cast<int>(src.size());
        if (n <= max_pts) {
            dst = src;
            return;
        }
        dst.reserve(static_cast<size_t>(max_pts));
        const int step = std::max(1, (n + max_pts - 1) / max_pts);
        for (int i = 0; i < n && static_cast<int>(dst.size()) < max_pts; i += step)
            dst.push_back(src[static_cast<size_t>(i)]);
    }

    bool refitCell(const GroundCellKey& k, GroundCell& cell)
    {
        cell.needs_refit = false;
        const int n = (int)cell.pts.size();
        if (n < min_pts_) return false;

        auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        subsampleToCloud(cell.pts, fit_max_pts_, *cloud);

        auto new_surf = std::make_shared<BSplineSurface>(3, 3, num_cp_, num_cp_, 0.25);
        if (!new_surf->apply(cloud, 30, 1, 1, 0.05))
            return false;  // 跑满迭代未收敛，保留旧曲面（如有）
        cell.surf = std::move(new_surf);
        return true;
    }

    GroundCellKey toCellKey(double x, double y) const {
        return {static_cast<int>(std::floor(x / cell_size_)),
                static_cast<int>(std::floor(y / cell_size_))};
    }

    double cell_size_;
    int    min_pts_;
    int    num_cp_;
    int    active_radius_;
    int    cell_max_pts_;
    int    fit_max_pts_;
    double cell_z_pct_;   // (0,1]=分位参考；<=0 用均值
    double cell_z_tol_;   // 上容差（米）；<=0 关闭
    std::unordered_map<GroundCellKey, GroundCell, GroundCellKeyHash> cells_;
};

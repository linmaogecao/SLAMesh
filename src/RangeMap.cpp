//
// Created by albus on 2026/2/3.
//

#include "RangeMap.h"
#include "slamesher_node.h"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
extern Parameter param;//in SLAMesh node
extern Log g_data;//in SLAMesh node
struct QueueItem {
    int index;
    Eigen::Vector3d accumulated_direction;  // Smoothed direction over last few steps
    int step_count;
};

void RangeImageProcessor::generateRangeImage(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                                             double ground_z_min, double ground_z_max,
                                             bool exclude_ground_band,
                                             double layer_range_min, double layer_range_max,
                                             double z_floor,
                                             const std::vector<bool>* exclude_mask)
{
    std::fill(range_image_.begin(), range_image_.end(), RangePixel());
    std::fill(pixel_to_cloud_idx_.begin(), pixel_to_cloud_idx_.end(), -1);
    occluded_points_.clear();

    float fov_up_rad = FOV_UP * M_PI / 180.0f;
    float fov_down_rad = FOV_DOWN * M_PI / 180.0f;
    float fov_total_rad = std::abs(fov_up_rad - fov_down_rad);

    // 有效 range 区间：类常量 ∩ 本层指定区间
    const double eff_range_min = std::max((double)MIN_RANGE, layer_range_min);
    const double eff_range_max = std::min((double)MAX_RANGE, layer_range_max);

    //#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cloud.size(); ++i) {
        const auto& pt = cloud.points[i];
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
            continue;
        // 优先使用 mask 排除（地面点）；没有 mask 时退回 z 带排除
        if (exclude_mask) {
            if (i < exclude_mask->size() && (*exclude_mask)[i]) continue;
        } else if (exclude_ground_band && pt.z >= ground_z_min && pt.z <= ground_z_max) {
            continue;
        }
        double range = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
        if (range < eff_range_min || range > eff_range_max || pt.z < z_floor) {
            continue;
        }

        double angle_vert = std::asin(pt.z / range);

        // 归一化到 [0, 1]
        double row_ratio = (angle_vert - fov_down_rad) / fov_total_rad;
        int row = std::round(row_ratio * (H_SCANS - 1));

        double angle_horiz = std::atan2(pt.y, pt.x);
        int col = std::round((angle_horiz + M_PI) / (2.0 * M_PI) * W_COLS);
        if (col >= W_COLS) col -= W_COLS;
        if (col < 0) col += W_COLS;

        if (row >= 0 && row < H_SCANS && col >= 0 && col < W_COLS) {
            int idx = row * W_COLS + col;
            RangePixel& px = range_image_[idx];
            if (!px.valid || range < px.range) {
                px.x = pt.x;
                px.y = pt.y;
                px.z = pt.z;
                px.range = range;
                px.valid = true;
                pixel_to_cloud_idx_[idx] = static_cast<int>(i);
            } else if (px.valid && collect_occluded_) {
                occluded_points_.emplace_back(pt.x, pt.y, pt.z);
            }
        }
    }
}

SegmentationResult RangeImageProcessor::segmentRangeImage(double theta_deg, double max_dist, int min_cluster_size,
                                                        double ground_z_min, double ground_z_max,
                                                        bool exclude_ground_band,
                                                        double normal_angle_deg) {
    SegmentationResult result;
    const int pixel_num = H_SCANS * W_COLS;
    result.label_map.assign(pixel_num, 0);

    const double theta_rad = theta_deg * M_PI / 180.0;
    const bool use_normal_gate = normal_angle_deg > 0.0;
    const double normal_cos_thresh = use_normal_gate
        ? std::cos(normal_angle_deg * M_PI / 180.0) : -2.0;

    auto isGroundPixel = [&](const RangePixel& px) {
        return px.z >= ground_z_min && px.z <= ground_z_max;
    };

    // ---------- 1. 标记有效像素（可选排除地面高度带） ----------
    std::vector<bool> pixel_valid(pixel_num, false);
    for (int idx = 0; idx < pixel_num; ++idx) {
        const auto& px = range_image_[idx];
        if (!px.valid) continue;
        if (exclude_ground_band && isGroundPixel(px)) continue;
        pixel_valid[idx] = true;
    }

    // ---------- 2. 预计算法向量 ----------
    // 以当前点 P 为顶点, 取左邻居 P_left = P(u, v-1) 和下邻居 P_down = P(u+1, v)
    // n = (P_left - P) x (P_down - P), 然后归一化
    // 朝向: 让法向量大致指向传感器原点 O (即 -P 方向)
    std::vector<Eigen::Vector3d> normals(pixel_num, Eigen::Vector3d::Zero());
    std::vector<bool> normal_valid(pixel_num, false);
    for (int u = 0; u < H_SCANS; ++u) {
        for (int v = 0; v < W_COLS; ++v) {
            int idx = u * W_COLS + v;
            if (!pixel_valid[idx]) continue;

            Eigen::Vector3d P, P_left, P_down;
            if (!getPoint(u, v, P)) continue;
            if (!getPoint(u, v - 1, P_left)) continue;          // 左邻居 (列周期已处理)
            if (u + 1 >= H_SCANS) continue;                      // 下邻居超界
            if (!getPoint(u + 1, v, P_down)) continue;

            Eigen::Vector3d e1 = P_left - P;
            Eigen::Vector3d e2 = P_down - P;
            Eigen::Vector3d n = e1.cross(e2);
            double len = n.norm();
            if (len < 1e-6) continue;
            n /= len;
            // 朝向传感器 (O = 原点, 视线方向是 -P)
            if (n.dot(-P) < 0) n = -n;
            normals[idx] = n;
            normal_valid[idx] = true;
        }
    }

    // ---------- 3. BFS 分割 ----------
    int current_label = 0;
    const int dir_u[4] = {-1, 1, 0, 0};
    const int dir_v[4] = { 0, 0,-1, 1};

    for (int seed = 0; seed < pixel_num; ++seed) {
        if (!pixel_valid[seed]) continue;
        if (result.label_map[seed] != 0) continue;

        ++current_label;
        std::vector<int> current_cluster;
        std::deque<int> q;
        result.label_map[seed] = current_label;
        current_cluster.push_back(seed);
        q.push_back(seed);

        while (!q.empty()) {
            int cur_index = q.front(); q.pop_front();
            int u = cur_index / W_COLS;
            int v = cur_index % W_COLS;
            const auto& cur_px = range_image_[cur_index];
            double crange = cur_px.range;//当前点到雷达距离

            for (int k = 0; k < 4; ++k) {
                int nu = u + dir_u[k];
                int nv = wrapCol(v + dir_v[k]);
                if (nu < 0 || nu >= H_SCANS) continue;

                int n_idx = nu * W_COLS + nv;
                if (!pixel_valid[n_idx]) continue;
                if (result.label_map[n_idx] != 0) continue;

                const auto& n_px = range_image_[n_idx];
                double nrange = n_px.range;//选中点到雷达的距离

                // ---- 条件 1: 欧氏距离粗筛 (基于角分辨率自适应) ----
                // 同一 scan 相邻两束光在距离 r 处的自然点间距 ≈ r * alpha_rad
                // k<2 为垂直方向邻居, k>=2 为水平方向邻居, 二者角分辨率不同
                double avg_range = (crange + nrange) * 0.5;
                double angle_res = (k < 2) ? alpha_vert_rad_ : alpha_horiz_rad_;
                // 容差系数 2.5: 允许倾斜面使点间距最多扩大到自然间距的 2.5 倍
                // 同时保留 max_dist 作为近距离的绝对下限, 防止阈值过小
                double adaptive_max_dist = std::max(max_dist, avg_range * angle_res * 2.5);
                double dx = cur_px.x - n_px.x;
                double dy = cur_px.y - n_px.y;
                double dz = cur_px.z - n_px.z;
                double euc_dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (euc_dist > adaptive_max_dist) continue;

                // ---- 条件 2: β 角度判断 (论文公式) ----
                // β = atan2(d2 * sin α, d1 - d2 * cos α)
                double d1 = std::max(crange, nrange);
                double d2 = std::min(crange, nrange);
                double alpha = (k < 2) ? alpha_vert_rad_ : alpha_horiz_rad_;
                double denom = d1 - d2 * std::cos(alpha);
                if (std::abs(denom) < 1e-9) continue;
                double beta = std::atan2(d2 * std::sin(alpha), denom);
                if (beta < theta_rad) continue;

                // 两侧法向任一无效则不据此拒绝；夹角阈值见 normal_angle_deg。
                if (use_normal_gate && normal_valid[cur_index] && normal_valid[n_idx]) {
                    if (normals[cur_index].dot(normals[n_idx]) < normal_cos_thresh) continue;
                }

                // ---- 接受 ----
                result.label_map[n_idx] = current_label;
                current_cluster.push_back(n_idx);
                q.push_back(n_idx);
            }
        }

        if ((int)current_cluster.size() > min_cluster_size) {
            result.clusters.push_back(std::move(current_cluster));
        }
    }
    std::cout << "Total clusters:-------------- " << result.clusters.size() << std::endl;
    return result;
}

void RangeImageProcessor::downsampleClusters(SegmentationResult& result,
                                             int min_pts,
                                             int target_max,
                                             int min_keep) const
{
    if (target_max <= 0 || min_pts <= 0) return;

    for (auto& indices : result.clusters) {
        const int n = static_cast<int>(indices.size());
        if (n <= min_pts) continue;

        int u_min = H_SCANS, u_max = 0;
        int v_min = W_COLS, v_max = 0;
        for (int idx : indices) {
            if (idx < 0 || idx >= static_cast<int>(range_image_.size())) continue;
            const int u = idx / W_COLS;
            const int v = idx % W_COLS;
            u_min = std::min(u_min, u);
            u_max = std::max(u_max, u);
            v_min = std::min(v_min, v);
            v_max = std::max(v_max, v);
        }
        if (u_max < u_min || v_max < v_min) continue;

        const int u_span = u_max - u_min + 1;
        const int v_span = v_max - v_min + 1;
        const double inv_keep = static_cast<double>(n) / static_cast<double>(target_max);
        const int denom = std::max(u_span, v_span);
        int row_step = std::max(1, static_cast<int>(std::lround(
            std::sqrt(inv_keep * static_cast<double>(u_span) / denom))));
        int col_step = std::max(1, static_cast<int>(std::lround(
            std::sqrt(inv_keep * static_cast<double>(v_span) / denom))));

        std::vector<int> kept;
        kept.reserve(target_max + 8);
        std::unordered_set<int> removed;
        removed.reserve(static_cast<size_t>(n));

        for (int idx : indices) {
            if (idx < 0 || idx >= static_cast<int>(range_image_.size())) continue;
            const int u = idx / W_COLS;
            const int v = idx % W_COLS;
            if (((u - u_min) % row_step == 0) && ((v - v_min) % col_step == 0)) {
                kept.push_back(idx);
            } else {
                removed.insert(idx);
            }
        }

        if (static_cast<int>(kept.size()) < min_keep) continue;

        for (int idx : removed) {
            if (idx >= 0 && idx < static_cast<int>(result.label_map.size()))
                result.label_map[idx] = 0;
        }
        indices = std::move(kept);
    }
}

void RangeImageProcessor::saveClustersWorldToTxt(const SegmentationResult& result,
                                                  const std::string& folder_path,
                                                  const Eigen::Matrix4d& transform)
{
    if (result.clusters.empty()) {
        std::cout << "--------------No clusters to save!" << std::endl;
        return;
    }

    const Eigen::Matrix3d R = transform.block<3,3>(0,0);
    const Eigen::Vector3d t = transform.block<3,1>(0,3);

    for (size_t i = 0; i < result.clusters.size(); ++i) {
        const auto& cluster_indices = result.clusters[i];

        std::string filename = folder_path + "/cluster_" + std::to_string(i) + ".txt";
        std::ofstream outfile(filename);
        if (!outfile.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            continue;
        }
        outfile << std::fixed << std::setprecision(4);

        for (int idx : cluster_indices) {
            const auto& px = range_image_[idx];
            if (!px.valid) continue;
            Eigen::Vector3d pw = R * Eigen::Vector3d(px.x, px.y, px.z) + t;
            outfile << pw.x() << " " << pw.y() << " " << pw.z() << "\n";
        }
        outfile.close();
    }
}

std::vector<Eigen::Vector3d> RangeImageProcessor::computeInitControlPoints(
    const SegmentationResult& result, int cluster_id,
    int num_u, int num_v, int smooth_window) const
{
    std::vector<Eigen::Vector3d> controlPs(num_u * num_v, Eigen::Vector3d::Zero());
    if (cluster_id < 0 || cluster_id >= (int)result.clusters.size()) return controlPs;
    const auto& idxs = result.clusters[cluster_id];
    // ---- 1. 接缝处理：找最长连续空列段，旋转列方向使 cluster 在列方向连续 ----
    std::vector<char> col_used(W_COLS, 0);
    for (int idx : idxs) col_used[idx % W_COLS] = 1;

    int best_start = 0, best_len = 0, run = 0, run_start = 0;
    for (int k = 0; k < 2 * W_COLS; ++k) {
        if (!col_used[k % W_COLS]) {
            if (run == 0) run_start = k % W_COLS;
            if (++run > best_len) { best_len = run; best_start = run_start; }
        } else run = 0;
    }
    int shift = (best_len > 0) ? ((best_start + best_len) % W_COLS) : 0;
    auto shiftCol   = [&](int v) { return (v - shift + W_COLS) % W_COLS; };
    auto unshiftCol = [&](int sv){ return (sv + shift) % W_COLS; };

    // ---- 2. 统计每行的 col_lo / col_hi ----
    std::vector<int> col_lo(H_SCANS, W_COLS);
    std::vector<int> col_hi(H_SCANS, -1);
    int u_min = H_SCANS, u_max = -1;

    //u 是行号 最后更新到u_min u_max是最高最低的有效行
    for (int raw_idx : idxs) {
        int u  = raw_idx / W_COLS;
        int sv = shiftCol(raw_idx % W_COLS);
        col_lo[u] = std::min(col_lo[u], sv);
        col_hi[u] = std::max(col_hi[u], sv);
        u_min = std::min(u_min, u);
        u_max = std::max(u_max, u);
    }

    // ---- 3. 填充空行（线性插值）+ 中值平滑，使 col_lo/col_hi 稳定 ----
    auto smoothBound = [&](std::vector<int>& arr, int fill_sentinel) {
        // 先线性插值填充空行
        int prev = -1;
        for (int u = u_min; u <= u_max; ++u) {
            if (arr[u] != fill_sentinel) {
                if (prev < 0) {
                    for (int k = u_min; k < u; ++k) arr[k] = arr[u];
                } else if (u > prev + 1) {
                    int v0 = arr[prev], v1 = arr[u];
                    for (int k = prev + 1; k < u; ++k) {
                        double t = (double)(k - prev) / (u - prev);
                        arr[k] = (int)std::round(v0 + t * (v1 - v0));
                    }
                }
                prev = u;
            }
        }
        if (prev >= 0)
            for (int k = prev + 1; k <= u_max; ++k) arr[k] = arr[prev];

        // 中值平滑（半窗口 smooth_window）
        std::vector<int> tmp = arr;
        for (int u = u_min; u <= u_max; ++u) {
            std::vector<int> win;
            for (int d = -smooth_window; d <= smooth_window; ++d) {
                int k = u + d;
                if (k >= u_min && k <= u_max) win.push_back(tmp[k]);
            }
            std::sort(win.begin(), win.end());
            arr[u] = win[win.size() / 2];
        }
    };

    smoothBound(col_lo, W_COLS);
    smoothBound(col_hi, -1);

    // ---- 4. 收集有效行 ----
    std::vector<int> valid_rings;
    valid_rings.reserve(u_max - u_min + 1);
    for (int u = u_min; u <= u_max; ++u)
        if (col_hi[u] >= col_lo[u]) valid_rings.push_back(u);

    //if ((int)valid_rings.size() < num_u) return controlPs;

    // ---- 5. 查 range image：取真实 xyz ----
    // 查单像素是否有效
    auto getPixel = [&](int ring, int orig_col) -> std::pair<bool, Eigen::Vector3d> {
        if (ring < 0 || ring >= H_SCANS) return {false, {}};
        orig_col = (orig_col % W_COLS + W_COLS) % W_COLS;
        const auto& px = range_image_[ring * W_COLS + orig_col];
        if (px.valid) return {true, {px.x, px.y, px.z}};
        return {false, {}};
    };

    auto getPoint3D = [&](int ring, int shifted_col) -> Eigen::Vector3d {
        int orig = unshiftCol(shifted_col);

        // 1. 直接查目标像素
        auto [ok, pt] = getPixel(ring, orig);
        if (ok) return pt;

        // 2. 用该行左右边界两端线性插值（同一 ring，几何最合理）
        int lo = col_lo[ring], hi = col_hi[ring];
        if (hi > lo) {
            auto [ok_lo, pt_lo] = getPixel(ring, unshiftCol(lo));
            auto [ok_hi, pt_hi] = getPixel(ring, unshiftCol(hi));
            if (ok_lo && ok_hi) {
                double t = (double)(shifted_col - lo) / (hi - lo);
                t = std::max(0.0, std::min(1.0, t));
                return pt_lo + t * (pt_hi - pt_lo);
            }
            if (ok_lo) return pt_lo;
            if (ok_hi) return pt_hi;
        }

        // 3. 兜底：向上下相邻有效行查同一列位置
        for (int dr = 1; dr < (int)valid_rings.size(); ++dr) {
            for (int sr : {1, -1}) {
                int r2 = ring + sr * dr;
                if (r2 < u_min || r2 > u_max) continue;
                auto [ok2, pt2] = getPixel(r2, orig);
                if (ok2) return pt2;
            }
        }

        // 4. 最终兜底：valid_rings 里第一个能找到边界点的行（理论上不会到这里）
        for (int vr : valid_rings) {
            auto [ok3, pt3] = getPixel(vr, unshiftCol(col_lo[vr]));
            if (ok3) return pt3;
        }
        return Eigen::Vector3d::Zero();
    };

    // ---- 6. 均匀选 num_u 行 × num_v 列，填充控制点网格 ----
    int n_valid = (int)valid_rings.size();
    for (int i = 0; i < num_u; ++i) {
        int vi = (int)std::round((double)i * (n_valid - 1) / (num_u - 1));
        vi = std::max(0, std::min(vi, n_valid - 1));
        int ring = valid_rings[vi];
        int lo = col_lo[ring], hi = col_hi[ring];

        for (int j = 0; j < num_v; ++j) {
            int sv = (num_v > 1)
                ? (int)std::round(lo + (double)j * (hi - lo) / (num_v - 1))
                : lo;
            sv = std::max(lo, std::min(sv, hi));
            controlPs[i * num_v + j] = getPoint3D(ring, sv);
        }
    }

    return controlPs;
}

bool RangeImageProcessor::getPointCloud(pcl::PointCloud<pcl::PointXYZ> & pcl_got, double voxel_filter_size)
{
    // get the new point cloud, from pcd file or from ros message
    ROS_DEBUG("getPointCloud");
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_raw_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_filtered_ptr(new pcl::PointCloud<pcl::PointXYZ>);//downsampled
    int data_set = param.dataset;//1  kitti, 2 maicity,
    //read
    if(param.read_offline_pcd) {
        const int frame_idx = std::max(0, g_data.step - 1);
        if(!readKitti(param.file_loc_dataset, param.seq, frame_idx, data_set,  *pcl_raw_ptr)){
            std::cout << "No more PCD file!" << std::endl;
            return false;
        }
    }
    else{
    }
    if(voxel_filter_size > 0){
        *pcl_filtered_ptr = pclVoxelFilter(pcl_raw_ptr, voxel_filter_size);
    }
    else{
        pcl_filtered_ptr = pcl_raw_ptr;
    }
    pcl_got = *pcl_filtered_ptr;
    return true;
}
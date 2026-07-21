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
            } else if (px.valid) {
                occluded_points_.emplace_back(pt.x, pt.y, pt.z);
            }
        }
    }
}

void RangeImageProcessor::saveRangeImageBin(const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "无法打开文件 " << filename << " 进行写入!\n";
        return;
    }

    // 1. 写入头信息：高 (H_SCANS) 和 宽 (W_COLS)
    int h = H_SCANS;
    int w = W_COLS;
    out.write(reinterpret_cast<const char*>(&h), sizeof(int));
    out.write(reinterpret_cast<const char*>(&w), sizeof(int));

    // 2. 逐个像素写入数据
    for (int i = 0; i < H_SCANS * W_COLS; ++i) {
        const auto& px = range_image_[i];

        // 提取出基本数据类型，直接取地址写入，不搞任何花里胡哨的数组
        int valid = px.valid ? 1 : 0;
        double r = px.range;
        double x = px.x;
        double y = px.y;
        double z = px.z;

        // 逐个变量写入二进制文件
        out.write(reinterpret_cast<const char*>(&valid), sizeof(int));
        out.write(reinterpret_cast<const char*>(&r), sizeof(double));
        out.write(reinterpret_cast<const char*>(&x), sizeof(double));
        out.write(reinterpret_cast<const char*>(&y), sizeof(double));
        out.write(reinterpret_cast<const char*>(&z), sizeof(double));
    }

    out.close();
    std::cout << ">>> Range Image 已成功保存至: " << filename << " <<<\n";
}

SegmentationResult RangeImageProcessor::segmentRangeImage(double theta_deg, double max_dist, int min_cluster_size,
                                                        double ground_z_min, double ground_z_max,
                                                        bool exclude_ground_band) {
    SegmentationResult result;
    const int pixel_num = H_SCANS * W_COLS;
    result.label_map.assign(pixel_num, 0);

    const double theta_rad = theta_deg * M_PI / 180.0;
    //const double normal_cos_thresh = std::cos(normal_angle_deg * M_PI / 180.0);

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

                // // ---- 条件 3: 法向量一致性 ----
                // if (normal_valid[cur_index] && normal_valid[n_idx]) {
                //     double cos_n = normals[cur_index].dot(normals[n_idx]);
                //     if (cos_n < normal_cos_thresh) continue;
                // }

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

void RangeImageProcessor::saveClustersToTxt(const SegmentationResult& result, const std::string& folder_path) {
    if (result.clusters.empty()) {
        std::cout << "--------------No clusters to save!" << std::endl;
        return;
    }

    //std::cout << "Saving " << result.clusters.size() << " clusters to " << folder_path << " ..." << std::endl;

    // 遍历每一个聚类
    for (size_t i = 0; i < result.clusters.size(); ++i) {
        const auto& cluster_indices = result.clusters[i];

        // 1. 构造文件名: folder/cluster_0.txt
        // 注意：请确保 folder_path 文件夹已经存在，否则 ofstream 会打开失败
        std::string filename = folder_path + "/cluster_" + std::to_string(i) + ".txt";

        std::ofstream outfile(filename);
        if (!outfile.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            continue;
        }

        // 2. 设置输出精度 (保留4位小数)
        outfile << std::fixed << std::setprecision(4);

        // 3. 遍历聚类中的每一个索引，从 range_image_ 中取 XYZ
        for (int idx : cluster_indices) {
            // 你的 range_image_ 是成员变量，直接访问
            const auto& px = range_image_[idx];

            // 这里不需要再 check valid 了，因为分割时已经过滤过了
            outfile << px.x << " " << px.y << " " << px.z << "\n";
        }

        outfile.close();
    }

    //std::cout << "All clusters saved." << std::endl;
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

bool RangeImageProcessor::findValidNeighborPt(int u, int v, const Eigen::Vector3d& center_pt, Eigen::Vector3d &neighbor_pt, bool is_vertical, int dir) const {
    double best_dist = std::numeric_limits<double>::max();
    bool found = false;
    
    // Adaptive distance threshold based on range to center
    double center_range = center_pt.norm();
    double max_neighbor_dist = 0.1 + center_range * 0.03;  // Closer points need smaller threshold
    
    if (!is_vertical) {
        // Horizontal direction: search in a small window
        for (int du = -1; du <= 1; ++du) {
            int curr_u = u + du;
            if (curr_u < 0 || curr_u >= H_SCANS) continue;
            
            for (int i = 1; i <= 5; ++i) {
                int curr_v = wrapCol(v + dir * i);
                Eigen::Vector3d temp_pt;
                
                if (getPoint(curr_u, curr_v, temp_pt)) {
                    double dist = (temp_pt - center_pt).norm();
                    
                    // Must be within reasonable distance AND closer than previous best
                    if (dist < max_neighbor_dist && dist < best_dist && dist > 1e-6) {
                        best_dist = dist;
                        neighbor_pt = temp_pt;
                        found = true;
                    }
                }
            }
        }
    } else {
        // Vertical direction: search in a window around expected position
        for (int i = 1; i <= 5; ++i) {
            int curr_u = u + dir * i;
            if (curr_u < 0 || curr_u >= H_SCANS) continue;
            
            for (int dv = -10; dv <= 10; ++dv) {
                int curr_v = wrapCol(v + dv);
                Eigen::Vector3d temp_pt;
                
                if (getPoint(curr_u, curr_v, temp_pt)) {
                    double dist = (temp_pt - center_pt).norm();
                    
                    if (dist < max_neighbor_dist && dist < best_dist && dist > 1e-6) {
                        best_dist = dist;
                        neighbor_pt = temp_pt;
                        found = true;
                    }
                }
            }
            
            // If we found a good neighbor in this row, don't search further rows
            if (found) break;
        }
    }
    
    return found;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr
RangeImageProcessor::generateOneClusterCloud(const SegmentationResult& result, int cluster_id) const {
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    if (cluster_id < 0 || cluster_id >= (int)result.clusters.size()) return cloud;
    const auto& indices = result.clusters[cluster_id];
    cloud->reserve(indices.size());
    for (int idx : indices) {
        if (idx < 0 || idx >= (int)range_image_.size()) continue;
        const auto& px = range_image_[idx];
        if (!px.valid) continue;
        cloud->push_back(pcl::PointXYZ(px.x, px.y, px.z));
    }
    cloud->width  = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;
    return cloud;
}

std::vector<VoxelKey>
RangeImageProcessor::getClusterOccupiedVoxels(const VoxelizedClusters& vc, int cluster_id) const {
    std::vector<VoxelKey> out;
    auto it = vc.cluster_to_subclusters.find(cluster_id);
    if (it == vc.cluster_to_subclusters.end()) return out;
    out.reserve(it->second.size());
    for (int sub_id : it->second) {
        if (sub_id < 0 || sub_id >= (int)vc.sub_clusters.size()) continue;
        out.push_back(vc.sub_clusters[sub_id].voxel_key);
    }
    return out;
}

std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> RangeImageProcessor::generateClusterClouds(const SegmentationResult& result) {
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_list;

    if (result.clusters.empty()) {
        return cloud_list;
    }
    for (size_t i = 0; i < result.clusters.size(); ++i) {
        const auto& cluster_indices = result.clusters[i];
        // 创建一个新的点云对象
        pcl::PointCloud<pcl::PointXYZ>::Ptr current_cluster(new pcl::PointCloud<pcl::PointXYZ>);

        // 预分配内存优化
        current_cluster->reserve(cluster_indices.size());

        // 遍历索引填充点
        int bool1 = false;
        int bool2 = false;
        int bool3 = false;
        for (int idx : cluster_indices) {
            const auto& px = range_image_[idx];
            current_cluster->push_back(pcl::PointXYZ(px.x, px.y, px.z));
            if (px.x > -4 && px.x < -2 && px.y > 11 && px.y < 12 ) {
                //std::cout<<"px.x "<<px.x<<" px.y "<<px.y<<std::endl;
                bool1 = true;
            }

            if (px.x > 0 && px.x < 2 && px.y > 11 && px.y < 12 ) {
                bool2 = true;
            }
            if (px.x > 2 && px.x < 4 && px.y > 11 && px.y < 12 ) {
                bool3 = true;
            }

        }
        if (bool1 && bool2 && bool3) {
            std::cout << "Saved ------" << i << std::endl;
        }
        // 设置点云属性
        current_cluster->width = current_cluster->points.size();
        current_cluster->height = 1;
        current_cluster->is_dense = true;

        // 加入列表
        cloud_list.push_back(current_cluster);
    }

    return cloud_list;
}

// ============================================================================
// 体素化分割结果
// ============================================================================
VoxelizedClusters RangeImageProcessor::voxelizeClusters(const SegmentationResult& result,
                                                        double voxel_size,
                                                        int min_subcluster_size) const {
    VoxelizedClusters vc;
    vc.voxel_size = voxel_size;
    if (voxel_size <= 1e-6) return vc;
    const double inv_size = 1.0 / voxel_size;

    auto keyFromPoint = [&](double x, double y, double z) {
        VoxelKey k;
        k.x = static_cast<int>(std::floor(x * inv_size));
        k.y = static_cast<int>(std::floor(y * inv_size));
        k.z = static_cast<int>(std::floor(z * inv_size));
        return k;
    };

    // 先按 cluster 遍历, cluster 内按 voxel 分桶
    for (size_t cid = 0; cid < result.clusters.size(); ++cid) {
        const auto& cluster_indices = result.clusters[cid];
        if (cluster_indices.empty()) continue;

        std::unordered_map<VoxelKey, std::vector<int>, VoxelKeyHash> bucket;
        bucket.reserve(cluster_indices.size() / 4 + 1);

        for (int idx : cluster_indices) {
            const auto& px = range_image_[idx];
            if (!px.valid) continue;
            VoxelKey k = keyFromPoint(px.x, px.y, px.z);
            bucket[k].push_back(idx);
        }

        // 每个 (cluster_id, voxel_key) 组合产生一个 SubCluster
        for (auto& kv : bucket) {
            if ((int)kv.second.size() < min_subcluster_size) continue;

            SubCluster sc;
            //cid 聚类的id
            sc.cluster_id = static_cast<int>(cid);
            //kvfirst 体素id的key
            sc.voxel_key = kv.first;
            //体素包含的range image id'
            sc.indices = std::move(kv.second);

            int sub_idx = static_cast<int>(vc.sub_clusters.size());
            vc.sub_clusters.push_back(std::move(sc));
            vc.voxels[kv.first].sub_cluster_ids.push_back(sub_idx);
            vc.cluster_to_subclusters[static_cast<int>(cid)].push_back(sub_idx);
        }
    }


    return vc;
}

void RangeImageProcessor::saveVoxelizedClustersToTxt(const VoxelizedClusters& vc,
                                                     const std::string& folder_path,
                                                     std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clouds) const {

    if (vc.sub_clusters.empty()) {
        std::cout << "No sub-clusters to save!" << std::endl;
        return;
    }
    int cnt = 0;
    for (const auto& sc : vc.sub_clusters) {
        auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        std::ostringstream fn;
        fn << folder_path << "/cluster_" << cnt << ".txt";
        cnt++;
        std::ofstream out(fn.str());
        if (!out.is_open()) {
            std::cerr << "Cannot open " << fn.str() << std::endl;
            continue;
        }
        out << std::fixed << std::setprecision(4);
        int bool1 = false;
        int bool2 = false;
        for (int idx : sc.indices) {
            const auto& px = range_image_[idx];
            out << px.x << " " << px.y << " " << px.z << "\n";
            cloud->push_back(pcl::PointXYZ(px.x, px.y, px.z));
            if (px.x > -9 && px.x < -8.5 && px.y > -6.5 && px.y < -6.25 ) {
                //std::cout<<"px.x "<<px.x<<" px.y "<<px.y<<std::endl;
                bool1 = true;
            }

            if (px.x > -8.75 && px.x < -8.5 && px.y > -7 && px.y < -6.6 ) {
                bool2 = true;
            }
        }
        if (bool1 && bool2) {
            std::cout << "Saved " << cnt << std::endl;
        }
        cloud->width = cloud->points.size();
        cloud->height = 1;
        cloud->is_dense = true;
        clouds.push_back(cloud);
    }
    std::cout << "Saved " << vc.sub_clusters.size() << " sub-clusters to "
              << folder_path << std::endl;
}

std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>
RangeImageProcessor::generateSubClusterClouds(const VoxelizedClusters& vc) const {
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds;
    clouds.reserve(vc.sub_clusters.size());
    for (const auto& sc : vc.sub_clusters) {
        auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        cloud->reserve(sc.indices.size());
        for (int idx : sc.indices) {
            const auto& px = range_image_[idx];
            cloud->push_back(pcl::PointXYZ(px.x, px.y, px.z));
        }
        cloud->width = cloud->points.size();
        cloud->height = 1;
        cloud->is_dense = true;
        clouds.push_back(cloud);
    }
    return clouds;
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

std::vector<Eigen::Vector3d> RangeImageProcessor::extractClusterBoundary3D(const SegmentationResult& result, int cluster_id,
    int K_ring) const {
std::vector<Eigen::Vector3d> out;
    if (cluster_id < 0 || cluster_id >= (int)result.clusters.size()) return out;
    const auto& idxs = result.clusters[cluster_id];
    if ((int)idxs.size() < 3) return out;
    // ---- 1. 接缝处理:找最长空列段,旋转列方向使 cluster 在列方向连续 ----
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
    auto shiftCol = [&](int v) { return (v - shift + W_COLS) % W_COLS; };
    // ---- 2. 每行统计 col_lo / col_hi ----
    std::vector<int> col_lo(H_SCANS, W_COLS);
    std::vector<int> col_hi(H_SCANS, -1);
    int u_min = H_SCANS, u_max = -1;
    for (int idx : idxs) {
        int u  = idx / W_COLS;
        int sv = shiftCol(idx % W_COLS);   // 旋转后的列
        col_lo[u] = std::min(col_lo[u], sv);
        col_hi[u] = std::max(col_hi[u], sv);
        u_min = std::min(u_min, u);
        u_max = std::max(u_max, u);
    }
    // ---- 3. 收集有效行 (每隔 K_ring 取一行,空行跳过) ----
    std::vector<int> valid_rings;
    for (int u = u_min; u <= u_max; u += K_ring) {
        if (col_hi[u] >= col_lo[u]) valid_rings.push_back(u);
    }
    // 确保最后一行也包含
    if (!valid_rings.empty() && valid_rings.back() != u_max && col_hi[u_max] >= col_lo[u_max])
        valid_rings.push_back(u_max);
    if (valid_rings.size() < 2) return out;
    // ---- 4. 构建多边形:左边 lo 从上到下,右边 hi 从下到上 ----
    // 把旋转后的列还原成原始列:orig = (sv + shift) % W_COLS
    auto unshiftCol = [&](int sv) { return (sv + shift) % W_COLS; };
    auto getPoint3D = [&](int u, int sv) -> Eigen::Vector3d {
        int orig_col = unshiftCol(sv);
        int ridx = u * W_COLS + orig_col;
        // 如果该格无效,在同行内线性搜最近有效点
        if (range_image_[ridx].valid) {
            const auto& px = range_image_[ridx];
            return {px.x, px.y, px.z};
        }
        // 向两侧各搜 5 格
        for (int d = 1; d <= 5; ++d) {
            for (int sign : {1, -1}) {
                int c2 = (orig_col + sign * d + W_COLS) % W_COLS;
                int r2 = u * W_COLS + c2;
                if (range_image_[r2].valid) {
                    const auto& px = range_image_[r2];
                    return {px.x, px.y, px.z};
                }
            }
        }
        return Eigen::Vector3d::Zero();   // 实在找不到就原点(极少发生)
    };
    // 左边界:lo 从上到下
    for (int u : valid_rings)
        out.push_back(getPoint3D(u, col_lo[u]));
    // 右边界:hi 从下到上(反向)
    for (int i = (int)valid_rings.size() - 1; i >= 0; --i)
        out.push_back(getPoint3D(valid_rings[i], col_hi[valid_rings[i]]));
    return out;   // 首尾相连即为闭合多边形
}

// ─────────────────────────────────────────────────────────────────────────────
// extractGroundByCellFilter — 列向传播 + XY 格子严格过滤
//   Step-1  全点云建内部 range image（不做 z 预过滤，障碍物也投影进来）
//   Step-2  逐列从底环向上：仅雷达系 z >= z_max 时整列停止；
//           |Δz| > col_max_step 只断连续性并以当前点（仍在带内）开新段，继续向上
//           z < z_min 跳过该环，不停止
//   Step-3  列传播候选点分配到 XY 格子，每格内严格 z 分位过滤
//   结果存入 ground_cloud_mask_（按点云下标），并返回地面点下标列表
// ─────────────────────────────────────────────────────────────────────────────
std::vector<int> RangeImageProcessor::extractGroundByCellFilter(
    const pcl::PointCloud<pcl::PointXYZ>& cloud,
    double z_min, double z_max,
    double col_max_step,
    double cell_size, double cell_z_pct, double cell_z_tol)
{
    ground_cloud_mask_.assign(cloud.size(), false);
    ground_debug_stats_ = GroundExtractDebugStats{};

    // 诊断用：与 GroundMatchPolicy::distanceBin 同一套 r_signed bin
    auto distanceBinRS = [](float x, float y) -> int {
        const double r = (x < 0.0f ? -1.0 : 1.0) *
                         std::hypot(static_cast<double>(x), static_cast<double>(y));
        return ground_match_policy::distanceBin(r);
    };

    // ---------- Step-1 : 全图内部 range image（障碍物也投影，用于列向截止判断）----------
    struct GndPix { float z = 0.f; int cidx = -1; float range = 1e9f; };
    std::vector<GndPix> gnd_ri(H_SCANS * W_COLS);

    const float fov_up_rad   =  FOV_UP   * (float)M_PI / 180.f;
    const float fov_down_rad =  FOV_DOWN * (float)M_PI / 180.f;
    const float fov_total    =  fov_up_rad - fov_down_rad;   // 正值

    for (int i = 0; i < (int)cloud.size(); ++i) {
        const auto& pt = cloud.points[i];
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;

        const float r2 = pt.x*pt.x + pt.y*pt.y + pt.z*pt.z;
        if (r2 < (float)(MIN_RANGE * MIN_RANGE)) continue;
        if (r2 > (float)(MAX_RANGE * MAX_RANGE)) continue;
        ++ground_debug_stats_.input_valid_points;
        const float r = std::sqrt(r2);

        const float ang_v = std::asin(pt.z / r);
        const float row_f = (ang_v - fov_down_rad) / fov_total * (float)(H_SCANS - 1);
        const int   row   = (int)std::round(row_f);
        if (row < 0 || row >= H_SCANS) continue;

        const float ang_h = std::atan2(pt.y, pt.x);
        int col = (int)std::round((ang_h + (float)M_PI) / (2.f*(float)M_PI) * (float)W_COLS);
        if (col >= W_COLS) col -= W_COLS;
        if (col < 0)       col += W_COLS;

        const int idx = row * W_COLS + col;
        if (r < gnd_ri[idx].range) {
            gnd_ri[idx] = {pt.z, i, r};
        }
    }

    // ---------- Step-2 : 列向传播（底环 → 顶环）----------
    // row 0 对应最低仰角（FOV_DOWN = -24.8°），row H_SCANS-1 对应最高仰角（FOV_UP = +2°）
    // 仅 z >= z_max（雷达系硬上限）整列停止；|Δz| 过大只断链并以当前带内点开新段
    const float z_min_f   = (float)z_min;
    const float z_max_f   = (float)z_max;
    const float step_max  = (float)col_max_step;

    std::vector<int> col_gnd_cidx;
    col_gnd_cidx.reserve(W_COLS * H_SCANS / 8);

    auto acceptColPoint = [&](int cidx, float z, float& last_z, bool& seeded) {
        if (!seeded) {
            seeded = true;
            ++ground_debug_stats_.seed_cols;
        }
        last_z = z;
        col_gnd_cidx.push_back(cidx);
        ++ground_debug_stats_.accepted_col_points;
        const auto& _p = cloud.points[cidx];
        const int bin = distanceBinRS(_p.x, _p.y);
        if (bin >= 0) ++ground_debug_stats_.col_bin_counts[bin];
    };

    for (int v = 0; v < W_COLS; ++v) {
        float last_z = std::numeric_limits<float>::quiet_NaN();
        bool seeded = false;
        for (int u = 0; u < H_SCANS; ++u) {  // 从底环(u=0)向顶环(u=H_SCANS-1)传播
            const auto& gp = gnd_ri[u * W_COLS + v];
            if (gp.cidx < 0) continue;          // 该环无点，跳过继续向上
            const float z = gp.z;

            // 整列唯一停止条件：雷达系 z 超出硬上限
            if (z >= z_max_f) {
                ++ground_debug_stats_.cut_by_zmax;
                { const auto& _p = cloud.points[gp.cidx];
                  const int bin = distanceBinRS(_p.x, _p.y);
                  if (bin >= 0) ++ground_debug_stats_.cut_zmax_bins[bin]; }
                break;
            }
            // 低于硬下限：跳过该环，不停止
            if (z < z_min_f) {
                continue;
            }

            if (std::isnan(last_z)) {
                acceptColPoint(gp.cidx, z, last_z, seeded);
            } else if (std::abs(z - last_z) > step_max) {
                // 台阶过大：断连续性，但继续向上；当前点仍在带内则开新段
                ++ground_debug_stats_.cut_by_step;
                { const auto& _p = cloud.points[gp.cidx];
                  const int bin = distanceBinRS(_p.x, _p.y);
                  if (bin >= 0) ++ground_debug_stats_.cut_step_bins[bin]; }
                acceptColPoint(gp.cidx, z, last_z, seeded);
            } else {
                acceptColPoint(gp.cidx, z, last_z, seeded);
            }
        }
    }

    // ---------- Step-3 : XY 格子严格过滤 ----------
    using CellKey = std::pair<int32_t, int32_t>;
    struct CellHash {
        size_t operator()(const CellKey& k) const noexcept {
            return std::hash<int64_t>()((int64_t(k.first) << 32) | (uint32_t)k.second);
        }
    };
    std::unordered_map<CellKey, std::vector<int>, CellHash> cells;
    const float cs = (float)cell_size;

    for (int ci : col_gnd_cidx) {
        const auto& pt = cloud.points[ci];
        const int32_t cx = (int32_t)std::floor(pt.x / cs);
        const int32_t cy = (int32_t)std::floor(pt.y / cs);
        cells[{cx, cy}].push_back(ci);
    }
    ground_debug_stats_.cell_count = (int)cells.size();

    std::vector<int> result;
    result.reserve(col_gnd_cidx.size());

    for (auto& [key, indices] : cells) {
        if (indices.empty()) continue;
        std::vector<float> zs;
        zs.reserve(indices.size());
        for (int ci : indices) zs.push_back(cloud.points[ci].z);
        std::sort(zs.begin(), zs.end());
        // 取低分位作参考，严格容差（偏紧，避免低矮障碍物混入）
        const int k = std::max(0, std::min((int)zs.size() - 1,
                               (int)std::floor(cell_z_pct * (double)(zs.size() - 1))));
        const float z_cut = zs[k] + (float)cell_z_tol;
        for (int ci : indices) {
            if (cloud.points[ci].z <= z_cut) {
                ground_cloud_mask_[ci] = true;
                result.push_back(ci);
                { const auto& _p = cloud.points[ci];
                  const int bin = distanceBinRS(_p.x, _p.y);
                  if (bin >= 0) ++ground_debug_stats_.final_bin_counts[bin]; }
            } else {
                ++ground_debug_stats_.cell_cut_total;
                { const auto& _p = cloud.points[ci];
                  const int bin = distanceBinRS(_p.x, _p.y);
                  if (bin >= 0) ++ground_debug_stats_.cell_cut_bin_counts[bin]; }
            }
        }
    }
    ground_debug_stats_.after_cell_filter = (int)result.size();
    return result;
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
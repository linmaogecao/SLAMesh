//
// Created by albus on 2026/1/18.
//
#include <fstream>
#include <ANN/ANN.h>

#include "BSpline.h"
#include "BSplineSDMErr.h"

using namespace Eigen;
Vector3d BSplineSurface::getPos(const Parameter& paraU, const Parameter& paraV, const vector<double>& knotsU, const vector<double>& knotsV, const std::vector<Vector3d>& controls, int num_cp_v) {
    double tf_u = paraU.second;
    int ki_u = paraU.first;
    Matrix4d matU = ComputeNonUniformBsplineMatrix(ki_u, knotsU);
    double dt_u = knotsU[ki_u+1] - knotsU[ki_u];
    double u = (tf_u - knotsU[ki_u]) / dt_u;
    Vector4d U_pos;

    U_pos << 1.0, u, u * u, u * u * u;
    RowVector4d weights_u = U_pos.transpose() * matU;

    double tf_v = paraV.second;
    int ki_v = paraV.first;
    Matrix4d matV = ComputeNonUniformBsplineMatrix(ki_v, knotsV);
    double dt_v = knotsV[ki_v+1] - knotsV[ki_v];
    double v = (tf_v - knotsV[ki_v]) / dt_v;
    Vector4d V_pos;

    V_pos << 1.0, v, v * v, v * v * v;
    RowVector4d weights_v = V_pos.transpose() * matV;


    Vector3d pos = Vector3d::Zero();
    for (int i = 0; i < 4; ++i) {
        int global_u_idx = ki_u - 3 + i;
        for (int j = 0; j < 4; ++j) {
            int global_v_idx = ki_v - 3 + j;
            int flat_index = global_u_idx * num_cp_v + global_v_idx;

            if (flat_index >= 0 && flat_index < controls.size()) {
                double weight = weights_u(i) * weights_v(j);
                pos += weight * controls[flat_index];
            }
        }
    }
    return pos;
}

Vector3d BSplineSurface::getFirstDiff(const Parameter& paraU, const Parameter& paraV, const vector<double>& knotsU, const vector<double>& knotsV, const std::vector<Vector3d>& controls,int num_cp_v, bool is_diff_u)
{
    double tf_u = paraU.second;
    int ki_u = paraU.first;
    Matrix4d matU = ComputeNonUniformBsplineMatrix(ki_u, knotsU);
    double dt_u = knotsU[ki_u+1] - knotsU[ki_u];
    double u = (dt_u > 1e-9) ? (tf_u - knotsU[ki_u]) / dt_u : 0.0;

    Vector4d vec_u;
    RowVector4d weights_u;

    if (is_diff_u) {
        vec_u << 0.0, 1.0, 2.0 * u, 3.0 * u * u;
        double scale = (dt_u > 1e-9) ? (1.0 / dt_u) : 0.0;
        weights_u = (vec_u.transpose() * matU) * scale;
    } else {
        vec_u << 1.0, u, u * u, u * u * u;
        weights_u = vec_u.transpose() * matU;
    }

    double tf_v = paraV.second;
    int ki_v = paraV.first;
    Matrix4d matV = ComputeNonUniformBsplineMatrix(ki_v, knotsV);
    double dt_v = knotsV[ki_v+1] - knotsV[ki_v];
    double v = (dt_v > 1e-9) ? (tf_v - knotsV[ki_v]) / dt_v : 0.0;

    Vector4d vec_v;
    RowVector4d weights_v;

    if (!is_diff_u) {
        vec_v << 0.0, 1.0, 2.0 * v, 3.0 * v * v;
        double scale = (dt_v > 1e-9) ? (1.0 / dt_v) : 0.0;
        weights_v = (vec_v.transpose() * matV) * scale;
    } else {
        vec_v << 1.0, v, v * v, v * v * v;
        weights_v = vec_v.transpose() * matV;
    }
    Vector3d result = Vector3d::Zero();

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            int u_idx_global = ki_u - 3 + i;
            int v_idx_global = ki_v - 3 + j;

            int flat_idx = u_idx_global * num_cp_v + v_idx_global;

            if (flat_idx >= 0 && flat_idx < controls.size()) {
                double w = weights_u(i) * weights_v(j);
                result += w * controls[flat_idx];
            }
        }
    }
    return result;
}

Vector3d BSplineSurface::getSecondDiff(const Parameter& paraU, const Parameter& paraV, const vector<double>& knotsU, const vector<double>& knotsV, const std::vector<Vector3d>& controls,int num_cp_v, int type) {
    double tf_u = paraU.second;
    int ki_u = paraU.first;
    Matrix4d matU = ComputeNonUniformBsplineMatrix(ki_u, knotsU);
    double dt_u = knotsU[ki_u+1] - knotsU[ki_u];
    double u = (dt_u > 1e-9) ? (tf_u - knotsU[ki_u]) / dt_u : 0.0;

    Vector4d vec_u;
    double scale_u = 1.0;

    double tf_v = paraV.second;
    int ki_v = paraV.first;
    Matrix4d matV = ComputeNonUniformBsplineMatrix(ki_v, knotsV);
    double dt_v = knotsV[ki_v+1] - knotsV[ki_v];
    double v = (dt_v > 1e-9) ? (tf_v - knotsV[ki_v]) / dt_v : 0.0;

    Vector4d vec_v;
    double scale_v = 1.0;

    if (type == 0) {
        vec_u << 0.0, 0.0, 2.0, 6.0 * u;
        scale_u = (dt_u > 1e-9) ? (1.0 / (dt_u * dt_u)) : 0.0;
        vec_v << 1.0, v, v * v, v * v * v;
        scale_v = 1.0;

    } else if (type == 1) {
        vec_u << 1.0, u, u * u, u * u * u;
        scale_u = 1.0;
        vec_v << 0.0, 0.0, 2.0, 6.0 * v;
        scale_v = (dt_v > 1e-9) ? (1.0 / (dt_v * dt_v)) : 0.0;

    } else if (type == 2) {
        vec_u << 0.0, 1.0, 2.0 * u, 3.0 * u * u;
        scale_u = (dt_u > 1e-9) ? (1.0 / dt_u) : 0.0;
        vec_v << 0.0, 1.0, 2.0 * v, 3.0 * v * v;
        scale_v = (dt_v > 1e-9) ? (1.0 / dt_v) : 0.0;
    }

    RowVector4d weights_u = (vec_u.transpose() * matU) * scale_u;
    RowVector4d weights_v = (vec_v.transpose() * matV) * scale_v;

    Vector3d result = Vector3d::Zero();

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            int u_idx_global = ki_u - 3 + i;
            int v_idx_global = ki_v - 3 + j;
            int flat_idx = u_idx_global * num_cp_v + v_idx_global;

            if (flat_idx >= 0 && flat_idx < controls.size()) {
                double w = weights_u(i) * weights_v(j);
                result += w * controls[flat_idx];
            }
        }
    }

    return result;
}


SurfaceCurvature BSplineSurface::getCurvature(const Parameter& paraU, const Parameter& paraV, const vector<double>& knotsU, const vector<double>& knotsV, const std::vector<Vector3d>& controls,int num_cp_v) {

    SurfaceCurvature result;
    Vector3d Su = getFirstDiff(paraU, paraV, knotsU, knotsV, controls, num_cp_v, true);  // true for u
    Vector3d Sv = getFirstDiff(paraU, paraV, knotsU, knotsV, controls, num_cp_v, false); // false for v

    Vector3d Suu = getSecondDiff(paraU, paraV, knotsU, knotsV, controls, num_cp_v, 0); // type 0 = Suu
    Vector3d Svv = getSecondDiff(paraU, paraV, knotsU, knotsV, controls, num_cp_v, 1); // type 1 = Svv
    Vector3d Suv = getSecondDiff(paraU, paraV, knotsU, knotsV, controls, num_cp_v, 2); // type 2 = Suv

    Vector3d normal_raw = Su.cross(Sv);
    double area = normal_raw.norm();

    if (area < 1e-9) {
        result.normal = Vector3d::UnitZ();
        result.tangent1 = Vector3d::UnitX();
        result.tangent2 = Vector3d::UnitY();
        result.E=1; result.G=1; result.F=0;
        result.L=0; result.M=0; result.N=0;
        result.k1=0; result.k2=0; result.K=0; result.H=0;
        return result;
    }
    result.normal = normal_raw / area;

    result.E = Su.dot(Su);
    result.F = Su.dot(Sv);
    result.G = Sv.dot(Sv);

    result.L = Suu.dot(result.normal);
    result.M = Suv.dot(result.normal);
    result.N = Svv.dot(result.normal);

    double det_I = result.E * result.G - result.F * result.F; // EG - F^2
    if (std::abs(det_I) < 1e-9) {
        result.K = 0; result.H = 0; result.k1 = 0; result.k2 = 0;
        return result;
    }

    double det_II = result.L * result.N - result.M * result.M; // LN - M^2

    result.K = det_II / det_I; // Gaussian
    result.H = (result.E * result.N + result.G * result.L - 2 * result.F * result.M) / (2 * det_I); // Mean


    double discriminant = result.H * result.H - result.K;
    discriminant = (discriminant < 0) ? 0 : discriminant;
    double root_dis = std::sqrt(discriminant);

    result.k1 = result.H + root_dis;
    result.k2 = result.H - root_dis;

    double A = result.L - result.k1 * result.E;
    double B = result.M - result.k1 * result.F;

    Vector3d T1;
    if (std::abs(A) < 1e-8 && std::abs(B) < 1e-8) {
        T1 = Su.normalized();
    } else {
        T1 = (-B * Su + A * Sv).normalized();
    }

    // T2 垂直于 T1 和 N
    Vector3d T2 = result.normal.cross(T1).normalized();

    result.tangent1 = T1;
    result.tangent2 = T2;
    result.point = getPos(paraU, paraV, knotsU, knotsV, controls, num_cp_v);
    return result;
}

void BSplineSurface::buildRangeGrid(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, int grid_res) {
    if (!cloud || cloud->empty()) return;

    // Use global bounds (already computed in initControlPoint)
    grid_origin_x_ = min_x;
    grid_origin_y_ = min_y;

    double range_x = max_x - min_x;
    double range_y = max_y - min_y;

    // Determine grid resolution based on data range
    grid_res_x_ = grid_res;
    grid_res_y_ = grid_res;
    grid_cell_size_x_ = range_x / grid_res_x_;
    grid_cell_size_y_ = range_y / grid_res_y_;

    // Avoid division by zero
    if (grid_cell_size_x_ < 1e-9) grid_cell_size_x_ = 1.0;
    if (grid_cell_size_y_ < 1e-9) grid_cell_size_y_ = 1.0;

    // Initialize grid
    range_grid_.clear();
    range_grid_.resize(grid_res_x_, std::vector<LocalRange>(grid_res_y_));

    // Fill grid with point data
    for (const auto& pt : cloud->points) {
        int ix = static_cast<int>((pt.x - grid_origin_x_) / grid_cell_size_x_);
        int iy = static_cast<int>((pt.y - grid_origin_y_) / grid_cell_size_y_);

        // Clamp to valid range
        ix = std::max(0, std::min(ix, grid_res_x_ - 1));
        iy = std::max(0, std::min(iy, grid_res_y_ - 1));

        LocalRange& cell = range_grid_[ix][iy];
        if (!cell.has_data) {
            cell.min_x = cell.max_x = pt.x;
            cell.min_y = cell.max_y = pt.y;
            cell.min_z = cell.max_z = pt.z;
            cell.has_data = true;
        } else {
            cell.min_x = std::min(cell.min_x, (double)pt.x);
            cell.max_x = std::max(cell.max_x, (double)pt.x);
            cell.min_y = std::min(cell.min_y, (double)pt.y);
            cell.max_y = std::max(cell.max_y, (double)pt.y);
            cell.min_z = std::min(cell.min_z, (double)pt.z);
            cell.max_z = std::max(cell.max_z, (double)pt.z);
        }
    }
}

double BSplineSurface::findFootPrint(const vector<Vector3d> &givepoints, vector<pair<Parameter, Parameter>> &footPrints, vector<double> &point_dists) {
    footPrints.assign(givepoints.size(), {Parameter(0, 0.0), Parameter(0, 0.0)});
    point_dists.assign(givepoints.size(), 0.0);

    const int n_pos = (int)positions.size();
    if (n_pos == 0 || givepoints.empty()) return 0.0;

    const int nu_spans = (int)span_sample_index_.size();
    const int nv_spans = (nu_spans > 0) ? (int)span_sample_index_[0].size() : 0;

    // Use UV-projected span lookup when plane_frame_ is valid and span index was built.
    const bool use_span = (plane_frame_.valid
                           && nu_spans > 0 && nv_spans > 0
                           && (plane_frame_.u_hi - plane_frame_.u_lo) > 1e-9
                           && (plane_frame_.v_hi - plane_frame_.v_lo) > 1e-9);

    // Find knot span index for a normalized parameter t in [0,1].
    auto findSpan = [&](double t, const vector<double>& knots, int num_cp) -> int {
        if (t >= 1.0 - 1e-9) return num_cp - 1;
        for (int k = 3; k < num_cp; ++k)
            if (knots[k] <= t && t < knots[k + 1]) return k;
        return 3;
    };

    double squareSum = 0.0;
    for (int i = 0; i < (int)givepoints.size(); ++i) {
        const Vector3d& p = givepoints[i];

        // ---- Step 1: find best initial sample via UV-span lookup or brute force ----
        int best_idx = 0;
        double best_dist_sq = std::numeric_limits<double>::max();

        if (use_span) {
            // Project query point onto PCA plane, map to normalized BSpline parameter.
            Vector3d diff = p - plane_frame_.centroid;
            double u_proj  = diff.dot(plane_frame_.u_axis);
            double v_proj  = diff.dot(plane_frame_.v_axis);
            double u_norm  = (u_proj - plane_frame_.u_lo) / (plane_frame_.u_hi - plane_frame_.u_lo);
            double v_norm  = (v_proj - plane_frame_.v_lo) / (plane_frame_.v_hi - plane_frame_.v_lo);
            u_norm = std::max(0.0, std::min(1.0, u_norm));
            v_norm = std::max(0.0, std::min(1.0, v_norm));

            // Estimate span indices (offset by -3 for span_sample_index_ indexing).
            int sp_u = std::max(0, std::min(nu_spans - 1, findSpan(u_norm, knots_u, controls_num_u) - 3));
            int sp_v = std::max(0, std::min(nv_spans - 1, findSpan(v_norm, knots_v, controls_num_v) - 3));

            // Search estimated span ± 1 in both directions (~9 patches at most).
            for (int dsi = -1; dsi <= 1; ++dsi) {
                int si = sp_u + dsi;
                if (si < 0 || si >= nu_spans) continue;
                for (int dsj = -1; dsj <= 1; ++dsj) {
                    int sj = sp_v + dsj;
                    if (sj < 0 || sj >= nv_spans) continue;
                    for (int idx : span_sample_index_[si][sj]) {
                        double d = (positions[idx] - p).squaredNorm();
                        if (d < best_dist_sq) { best_dist_sq = d; best_idx = idx; }
                    }
                }
            }
            // Fallback: shouldn't happen on a well-fitted surface, but be safe.
            if (best_dist_sq == std::numeric_limits<double>::max()) {
                for (int k = 0; k < n_pos; ++k) {
                    double d = (positions[k] - p).squaredNorm();
                    if (d < best_dist_sq) { best_dist_sq = d; best_idx = k; }
                }
            }
        } else {
            // Fallback: linear scan over all samples (no heap alloc, no ANN rebuild).
            for (int k = 0; k < n_pos; ++k) {
                double d = (positions[k] - p).squaredNorm();
                if (d < best_dist_sq) { best_dist_sq = d; best_idx = k; }
            }
        }

        // ---- Step 2: Newton refinement from best initial (u,v) ----
        // Solves r·Su = 0, r·Sv = 0 where r = S(u,v) - p.
        auto [paraU, paraV] = sampling_paras_[best_idx];

        for (int nr = 0; nr < 3; ++nr) {
            Vector3d S  = getPos(paraU, paraV, knots_u, knots_v, controls, controls_num_v);
            Vector3d Su = getFirstDiff(paraU, paraV, knots_u, knots_v, controls, controls_num_v, true);
            Vector3d Sv = getFirstDiff(paraU, paraV, knots_u, knots_v, controls, controls_num_v, false);
            Vector3d r  = S - p;

            double a00 = Su.dot(Su), a01 = Su.dot(Sv), a11 = Sv.dot(Sv);
            double b0  = -r.dot(Su), b1  = -r.dot(Sv);
            double det = a00 * a11 - a01 * a01;
            if (std::abs(det) < 1e-10) break;

            double du = (b0 * a11 - b1 * a01) / det;
            double dv = (a00 * b1 - a01 * b0) / det;

            double new_tf_u = std::max(0.0, std::min(1.0, paraU.second + du));
            double new_tf_v = std::max(0.0, std::min(1.0, paraV.second + dv));

            paraU = {findSpan(new_tf_u, knots_u, controls_num_u), new_tf_u};
            paraV = {findSpan(new_tf_v, knots_v, controls_num_v), new_tf_v};

            if (std::abs(du) < 1e-6 && std::abs(dv) < 1e-6) break;
        }

        Vector3d S_final    = getPos(paraU, paraV, knots_u, knots_v, controls, controls_num_v);
        double true_dist_sq = (S_final - p).squaredNorm();

        squareSum      += std::sqrt(true_dist_sq);
        point_dists[i]  = true_dist_sq;
        footPrints[i]   = {paraU, paraV};
    }

    return squareSum;
}

std::pair<BSplineSurface::Parameter, BSplineSurface::Parameter> BSplineSurface:: getPara(int index) {
    if (index < 0 || index >= sampling_paras_.size()) {
        return {Parameter(0, 0.0), Parameter(0, 0.0)};
    }
    return sampling_paras_[index];
}

void BSplineSurface::pclToEigenVector(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
    std::vector<Vector3d> &out_vec) {
    if (!cloud || cloud->empty()) {
        return;
    }
    out_vec.clear();
    out_vec.reserve(cloud->size());

    for (const auto& pt : cloud->points) {
        out_vec.emplace_back(pt.x, pt.y, pt.z);
    }
}

void BSplineSurface::initControlPoint(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, vector<Vector3d> &controlPs, int num_u,
                                      int num_v) {
    // 默认走 PCA 平面版本; 想回到旧版的硬编码三标准面方案, 在调用方手工切回即可
    initControlPointPCA(cloud, controlPs, num_u, num_v);
    // const double cx = 10.5;
    // const double cy = -9.0;
    // const double r = 3.6;
    //
    // const int num_points = 20;
    //
    // // 角度范围（弧度）
    // double start_deg = 45.0;
    // double end_deg   = 225.0;
    //
    // double start_rad = start_deg * M_PI / 180.0;
    // double end_rad   = end_deg   * M_PI / 180.0;
    //
    // // 步长（包含首尾）
    // double step = (end_rad - start_rad) / (num_points - 1);
    //
    // controlPs.assign(num_u * num_v, Vector3d::Zero());
    // for (int i = 0; i < num_u; ++i) {
    //     double zz = 1.6 - i*0.2;
    //     for (int j = 0; j < num_v; ++j) {
    //         double theta = start_rad + j * step;
    //         double xx = cx + r * std::cos(theta);
    //         double yy = cy + r * std::sin(theta);
    //         controlPs[i * num_u + j] = Vector3d(xx, yy, zz);
    //     }
    // }
    // readWrite::writeDate("./../build/initial_control.txt", controlPs, false);
}

void BSplineSurface::computePlaneFrame(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    plane_frame_.valid = false;
    if (!cloud || cloud->points.size() < 3) return;

    const int N = static_cast<int>(cloud->points.size());

    // 用 Eigen::Map 避免大矩阵拷贝 (但 PCL 的 xyz 是 float, 这里仍要复制成 double)
    Eigen::Matrix<double, 3, Eigen::Dynamic> P(3, N);
    for (int i = 0; i < N; ++i) {
        P(0, i) = cloud->points[i].x;
        P(1, i) = cloud->points[i].y;
        P(2, i) = cloud->points[i].z;
    }

    plane_frame_.centroid = P.rowwise().mean();
    Eigen::Matrix<double, 3, Eigen::Dynamic> Q = P.colwise() - plane_frame_.centroid;
    Eigen::Matrix3d cov = (Q * Q.transpose()) / static_cast<double>(N);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
    // eigenvalues() 升序: [小, 中, 大]
    plane_frame_.u_axis = es.eigenvectors().col(2).normalized(); // 最大方差
    plane_frame_.v_axis = es.eigenvectors().col(1).normalized();
    plane_frame_.n_axis = es.eigenvectors().col(0).normalized(); // 近似法向

    // 统计 (u, v, h) 范围
    plane_frame_.u_min =  std::numeric_limits<double>::infinity();
    plane_frame_.u_max = -std::numeric_limits<double>::infinity();
    plane_frame_.v_min =  std::numeric_limits<double>::infinity();
    plane_frame_.v_max = -std::numeric_limits<double>::infinity();
    double h_sum = 0.0;
    for (int i = 0; i < N; ++i) {
        Eigen::Vector3d d = Q.col(i);
        double u = d.dot(plane_frame_.u_axis);
        double v = d.dot(plane_frame_.v_axis);
        double h = d.dot(plane_frame_.n_axis);
        plane_frame_.u_min = std::min(plane_frame_.u_min, u);
        plane_frame_.u_max = std::max(plane_frame_.u_max, u);
        plane_frame_.v_min = std::min(plane_frame_.v_min, v);
        plane_frame_.v_max = std::max(plane_frame_.v_max, v);
        h_sum += h;
    }
    plane_frame_.h_avg = h_sum / static_cast<double>(N);
    plane_frame_.valid = true;
}

void BSplineSurface::initControlPointPCA(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                         vector<Vector3d>& controlPs, int num_u, int num_v,
                                         double margin_ratio) {
    if (!cloud || cloud->points.size() < 3 || num_u < 2 || num_v < 2) return;

    // ----- 1. PCA 平面坐标系 -----
    computePlaneFrame(cloud);
    if (!plane_frame_.valid) return;

    // ----- 2. 全局 AABB 仍要保存, 供 isPointValid / buildRangeGrid 使用 -----
    Eigen::Vector4f min_pt_4f, max_pt_4f;
    pcl::getMinMax3D(*cloud, min_pt_4f, max_pt_4f);
    max_x = max_pt_4f[0]; min_x = min_pt_4f[0];
    max_y = max_pt_4f[1]; min_y = min_pt_4f[1];
    max_z = max_pt_4f[2]; min_z = min_pt_4f[2];

    // ----- 3. KD-tree (3D) 用于每个控制点找最近邻 -> 估计法向偏移 h -----
    //input_kdtree_.setInputCloud(cloud);

    // ----- 4. (u, v) 矩形 + margin -----
    const double u_range = plane_frame_.u_max - plane_frame_.u_min;
    const double v_range = plane_frame_.v_max - plane_frame_.v_min;
    const double u_lo = plane_frame_.u_min - margin_ratio * u_range;
    const double u_hi = plane_frame_.u_max + margin_ratio * u_range;
    const double v_lo = plane_frame_.v_min - margin_ratio * v_range;
    const double v_hi = plane_frame_.v_max + margin_ratio * v_range;
    plane_frame_.u_lo = u_lo;
    plane_frame_.u_hi = u_hi;
    plane_frame_.v_lo = v_lo;
    plane_frame_.v_hi = v_hi;

    const double u_step = (u_hi - u_lo) / (num_u - 1);
    const double v_step = (v_hi - v_lo) / (num_v - 1);

    controlPs.assign(num_u * num_v, Vector3d::Zero());

    // ----- 5. 在 (u, v) 上均匀取点 -> 3D KD-tree 找最近邻 -> 用最近邻在 n 方向的投影做高度 -----
    for (int i = 0; i < num_u; ++i) {
        for (int j = 0; j < num_v; ++j) {
            const double cu = u_lo + i * u_step;
            const double cv = v_lo + j * v_step;

            // // 落在平面上的 3D 查询点 (高度先用 h_avg)
            const Vector3d query_3d = plane_frame_.centroid
                                    + cu * plane_frame_.u_axis
                                    + cv * plane_frame_.v_axis
                                    + plane_frame_.h_avg * plane_frame_.n_axis;

            pcl::PointXYZ q;
            q.x = static_cast<float>(query_3d.x());
            q.y = static_cast<float>(query_3d.y());
            q.z = static_cast<float>(query_3d.z());

            std::vector<int>   idx(1);
            std::vector<float> sqdist(1);

            double h = plane_frame_.h_avg;
            if (input_kdtree_.nearestKSearch(q, 1, idx, sqdist) > 0) {
                const auto& np = cloud->points[idx[0]];
                Vector3d d(np.x - plane_frame_.centroid.x(),
                           np.y - plane_frame_.centroid.y(),
                           np.z - plane_frame_.centroid.z());
                h = d.dot(plane_frame_.n_axis); // 最近邻在 n 方向的投影 -> 控制点高度
            }

            const Vector3d cp = plane_frame_.centroid
                              + cu * plane_frame_.u_axis
                              + cv * plane_frame_.v_axis
                              + h  * plane_frame_.n_axis;
            controlPs[i * num_v + j] = cp;
        }
    }
}
bool BSplineSurface::isPointValid(const Vector3d& p) {
    double global_margin = 0.03;
    if (p.x() < (min_x - global_margin) || p.x() > (max_x + global_margin) ||
        p.y() < (min_y - global_margin) || p.y() > (max_y + global_margin) ||
        p.z() < (min_z - global_margin) || p.z() > (max_z + global_margin)) {
        cn1++;
        return false;
        }
    if (range_grid_.empty()) {
        return true;
    }

    // Find grid cell for this point
    int ix = static_cast<int>((p.x() - grid_origin_x_) / grid_cell_size_x_);
    int iy = static_cast<int>((p.y() - grid_origin_y_) / grid_cell_size_y_);

    // Clamp to valid range
    ix = std::max(0, std::min(ix, grid_res_x_ - 1));
    iy = std::max(0, std::min(iy, grid_res_y_ - 1));

    // Check current cell and neighboring cells (3x3 neighborhood)
    double local_min_x = std::numeric_limits<double>::max();
    double local_max_x = std::numeric_limits<double>::lowest();
    double local_min_y = std::numeric_limits<double>::max();
    double local_max_y = std::numeric_limits<double>::lowest();
    double local_min_z = std::numeric_limits<double>::max();
    double local_max_z = std::numeric_limits<double>::lowest();
    bool found_data = false;

    for (int di = -1; di <= 1; ++di) {
        for (int dj = -1; dj <= 1; ++dj) {
            int ni = ix + di;
            int nj = iy + dj;
            if (ni >= 0 && ni < grid_res_x_ && nj >= 0 && nj < grid_res_y_) {
                const LocalRange& cell = range_grid_[ni][nj];
                if (cell.has_data) {
                    found_data = true;
                    local_min_x = std::min(local_min_x, cell.min_x);
                    local_max_x = std::max(local_max_x, cell.max_x);
                    local_min_y = std::min(local_min_y, cell.min_y);
                    local_max_y = std::max(local_max_y, cell.max_y);
                    local_min_z = std::min(local_min_z, cell.min_z);
                    local_max_z = std::max(local_max_z, cell.max_z);
                }
            }
        }
    }

    // If no data in neighborhood, reject the point (it's in an empty region)
    if (!found_data) {
        cn2++;
        return false;
    }

    // Check if point is within local range + margin
    double local_margin = 0.02;
    if (p.x() < (local_min_x - local_margin) || p.x() > (local_max_x + local_margin) ||
        p.y() < (local_min_y - local_margin) || p.y() > (local_max_y + local_margin) ||
        p.z() < (local_min_z - local_margin) || p.z() > (local_max_z + local_margin)) {
        cn3++;
        return false;
    }


    return true;
}

void BSplineSurface::setNewControl(const vector<Vector3d> &controlPs, int num_u, int num_v,bool isCut) {
    clear();
    controls = controlPs;
    controls_num_u = num_u;
    controls_num_v = num_v;

    const int start_u = 3;
    const int end_u   = controls_num_u;
    const int start_v = 3;
    const int end_v   = controls_num_v;

    // Pre-allocate span index grid: span i maps to index (i - 3)
    const int nu_spans = std::max(0, controls_num_u - 3);
    const int nv_spans = std::max(0, controls_num_v - 3);
    if (nu_spans > 0 && nv_spans > 0)
        span_sample_index_.assign(nu_spans, vector<vector<int>>(nv_spans));

    int pos_idx = 0;
    for (int i = start_u; i <= end_u; ++i) {
        double dt_u = knots_u[i + 1] - knots_u[i];
        if (dt_u <= 1e-6) continue;
        const int si = i - 3;
        for (int j = start_v; j <= end_v; ++j) {
            double dt_v = knots_v[j + 1] - knots_v[j];
            if (dt_v <= 1e-6) continue;
            const int sj = j - 3;

            for (double fu = 0.0; fu <= 1.0; fu += interal_) {
                for (double fv = 0.0; fv <= 1.0; fv += interal_) {
                    double global_u = knots_u[i] + fu * dt_u;
                    double global_v = knots_v[j] + fv * dt_v;

                    Parameter paraU(i, global_u);
                    Parameter paraV(j, global_v);

                    Vector3d p = getPos(paraU, paraV, knots_u, knots_v, controls, controls_num_v);
                    positions.push_back(p);
                    sampling_paras_.push_back({paraU, paraV});

                    if (si >= 0 && si < nu_spans && sj >= 0 && sj < nv_spans)
                        span_sample_index_[si][sj].push_back(pos_idx);
                    ++pos_idx;
                }
            }
        }
    }
}

void BSplineSurface::setKnotParams(int num_cp_u,int num_cp_v) {
    knots_u.resize(num_cp_u + 4);
    knots_v.resize(num_cp_v + 4);
    double denom_u = (double)(num_cp_u - 3);
    for (int i = 0; i < knots_u.size(); ++i) {
        if (i <= 3) {
            knots_u[i] = 0.0;
        }
        else if (i >= num_cp_u) {
            knots_u[i] = 1.0;
        }
        else {
            knots_u[i] = double(i - 3) / denom_u;
        }
    }
    double denom_v = (double)(num_cp_v - 3);

    for (int i = 0; i < knots_v.size(); ++i) {
        if (i <= 3) {
            knots_v[i] = 0.0;
        }
        else if (i >= num_cp_v) {
            knots_v[i] = 1.0;
        }
        else {
            knots_v[i] = double(i - 3) / denom_v;
        }
    }

}


Eigen::Matrix4d BSplineSurface::ComputeNonUniformBsplineMatrix(int i, const vector<double> &knots) {
    Eigen::Matrix4d M = Eigen::Matrix4d::Zero();
    double t_i   = knots[i];
    double t_im1 = knots[i - 1]; // i-1
    double t_im2 = knots[i - 2]; // i-2
    double t_ip1 = knots[i + 1]; // i+1
    double t_ip2 = knots[i + 2]; // i+2
    double t_ip3 = knots[i + 3]; // i+3

    double dt_i_im1   = t_i - t_im1;
    double dt_ip1_i   = t_ip1 - t_i;
    double dt_ip1_im1 = t_ip1 - t_im1;
    double dt_ip1_im2 = t_ip1 - t_im2;
    double dt_ip2_im1 = t_ip2 - t_im1;
    double dt_ip2_i   = t_ip2 - t_i;
    double dt_ip3_i   = t_ip3 - t_i;

    double m00 = (dt_ip1_i * dt_ip1_i) / (dt_ip1_im1 * dt_ip1_im2);
    double m02 = (dt_i_im1 * dt_i_im1) / (dt_ip2_im1 * dt_ip1_im1);
    double m22 = 3.0 * (dt_ip1_i * dt_ip1_i) / (dt_ip2_im1 * dt_ip1_im1);
    double m33 = (dt_ip1_i * dt_ip1_i) / (dt_ip3_i * dt_ip2_i);
    double m12 = 3.0 * dt_ip1_i * dt_i_im1 / (dt_ip2_im1 * dt_ip1_im1);

    // Row 0
    M(0, 0) = m00;
    M(0, 1) = 1.0 - m00 - m02; // m01 = 1 - m00 - m02
    M(0, 2) = m02;
    M(0, 3) = 0.0;

    // Row 1
    M(1, 0) = -3.0 * m00;
    M(1, 1) = 3.0 * m00 - m12;
    M(1, 2) = m12;
    M(1, 3) = 0.0;

    // Row 2
    M(2, 0) = 3.0 * m00;
    M(2, 1) = -3.0 * m00 - m22;
    M(2, 2) = m22;
    M(2, 3) = 0.0;

    // Row 3
    M(3, 0) = -m00;
    double term_extra = (dt_ip1_i * dt_ip1_i) / (dt_ip2_i * dt_ip2_im1);
    M(3, 2) = -m22 / 3.0 - m33 - term_extra;
    M(3, 1) = m00 - M(3, 2) - m33; // m31 = m00 - m32 - m33
    M(3, 3) = m33;

    return M;
}


double BSplineSurface::apply(
        pcl::PointCloud<pcl::PointXYZ>::Ptr& points,
        int maxIterNum,
        double alpha,
        double gama,
        double eplison )
{

    auto ms_since = [](std::chrono::high_resolution_clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
    };
    auto t0 = std::chrono::high_resolution_clock::now();

    this->input_cloud_ = points;

    //std::cout<<"point size<<"<<points->size()<<std::endl;
    this->input_kdtree_.setInputCloud(points);
    double t_kdtree = ms_since(t0); t0 = std::chrono::high_resolution_clock::now();
    vector<Vector3d> controlPs;

    const int expected = controls_num_u * controls_num_v;
    if (!ext_init_controls_.empty() && (int)ext_init_controls_.size() == expected) {
        // 使用外部提供的初始控制点（range-image 行列采样），跳过 PCA 初始化
        controlPs = ext_init_controls_;
        // AABB 仍需更新，供 isPointValid / buildRangeGrid 使用
        Eigen::Vector4f min_pt_4f, max_pt_4f;
        pcl::getMinMax3D(*points, min_pt_4f, max_pt_4f);
        max_x = max_pt_4f[0]; min_x = min_pt_4f[0];
        max_y = max_pt_4f[1]; min_y = min_pt_4f[1];
        max_z = max_pt_4f[2]; min_z = min_pt_4f[2];
    } else {
        controlPs.resize(expected);
        initControlPoint(points, controlPs, controls_num_u, controls_num_v);
    }
    double t_init = ms_since(t0); t0 = std::chrono::high_resolution_clock::now();

    buildRangeGrid(points, 50);
    double t_grid = ms_since(t0); t0 = std::chrono::high_resolution_clock::now();
    setKnotParams(controls_num_u, controls_num_v);
    setNewControl(controlPs,controls_num_u,controls_num_v);
    double t_set = ms_since(t0); t0 = std::chrono::high_resolution_clock::now();

    // update the control point
    // compute P"(t)
    // MatrixXd pm = spline_surface->getSIntegralSq();
    // MatrixXd sm = spline_surface->getFIntegralSq();
    // end test

    // find the foot print, will result in error
    double total_error = 1e9;
    double last_error = 1e9;
    vector<Vector3d> givepoints;
    pclToEigenVector(points, givepoints);
    double t_vec = ms_since(t0);
    // std::cout << std::fixed << std::setprecision(2)
    //       << "[apply init] kdtree=" << t_kdtree << "ms initCP=" << t_init
    //       << "ms grid=" << t_grid
    //       << "ms pcl2eigen=" << t_vec << "ms"
    //       << " pts=" << givepoints.size() << std::endl;

    int point_num = givepoints.size();bool stop_flag = false;
    double sum_fp = 0.0;
    double sum_pre = 0.0;
    double sum_data_res = 0.0;
    double sum_smooth = 0.0;
    double sum_bound = 0.0;
    double sum_solve = 0.0;
    double sum_set = 0.0;
    for(int iter = 0; iter < maxIterNum; ++iter) {
        ceres::Problem problem;
        vector<pair<Parameter, Parameter>> parameters;
        vector<double> point_dists;
        auto t1 = std::chrono::high_resolution_clock::now();
        double current_sq_dist = findFootPrint(givepoints, parameters,point_dists);
        sum_fp += ms_since(t1);
        t1 = std::chrono::high_resolution_clock::now();
        double diff = last_error - current_sq_dist;
        double relative_decrease = std::abs(diff) / (last_error + 1e-10); // 防止除0
        double rmse = (current_sq_dist / point_num); // 均方根误差(平均距离)

        // 打印详细调试信息，方便观察收敛情况
        // std::cout << "Iter: " << iter
        //           << " | Total Err: " << current_sq_dist
        //           << " | RMSE: " << rmse
        //           << " | Rel Decr: " << relative_decrease * 100.0 << "%" << std::endl;

        // --- 新的终止策略 ---

        // 策略1: 相对下降率极小 (收敛平台期)
        // 例如：如果你传入的 eplison 是 1e-3 (0.1%)，当提升小于这个比例时停止
        // iter > 0 是为了防止第一次 last_error 为初始值时的误判
        if (iter > 0 && relative_decrease < eplison) {
            //std::cout << ">> Converged by Relative Decrease (" << relative_decrease << " < " << eplison << ")" << std::endl;
            vector<Vector3d> controls_copy = controls; // <--- ✅ 先克隆一份

            if (!stop_flag) {
                stop_flag = true;
                //std::cout<<"ready to stop "<<std::endl;
            }
            else {
                setNewControl(controls_copy, controls_num_u, controls_num_v,true);
                break;
            }
        }

        // 策略2: 绝对精度满足要求 (RMSE)
        // 这里的 1e-3 代表平均误差小于 0.001 (假设单位是米，即1mm)
        // 你可以根据你的点云尺度调整这个值
        if (rmse < 1e-2) {
            //std::cout << ">> Converged by RMSE (" << rmse << " < 1e-3)" << std::endl;
            vector<Vector3d> controls_copy = controls; // <--- ✅ 先克隆一份
            if (!stop_flag) {
                stop_flag = true;
                setNewControl(controls_copy, controls_num_u, controls_num_v,true);
            }
            else {
                setNewControl(controls_copy, controls_num_u, controls_num_v,true);
                break;
            }
        }

        last_error = current_sq_dist;
        auto computeMAD = [](const vector<double>& v){
            vector<double> tmp = v;
            std::nth_element(tmp.begin(), tmp.begin() + tmp.size()/2, tmp.end());
            double med = tmp[tmp.size()/2];
            for (auto& x : tmp) x = std::abs(x - med);
            std::nth_element(tmp.begin(), tmp.begin() + tmp.size()/2, tmp.end());
            double mad = tmp[tmp.size()/2];
            return std::make_pair(med, mad);
        };

        auto [med, mad] = computeMAD(point_dists);
        double sigma = 1.4826 * mad + 1e-6;
        double inlier_thresh = med + 2.5 * sigma;
        sum_pre += ms_since(t1);
        t1 = std::chrono::high_resolution_clock::now();
        std::vector<double> active_weights;
        std::vector<double*> active_cp_pointers;
        active_weights.reserve(16);
        active_cp_pointers.reserve(16);
        for(int i = 0; i< parameters.size(); i++)
        {
            active_weights.clear();
            active_cp_pointers.clear();
            if (point_dists[i] > inlier_thresh) {
                continue; // 直接当噪声，跳过
            }
            Parameter paraU = parameters[i].first, paraV = parameters[i].second;
            SurfaceCurvature surf_info = getCurvature(paraU, paraV, knots_u, knots_v, controls, controls_num_v);

            surf_info.point = getPos(paraU, paraV, knots_u, knots_v,controls,controls_num_v);
            int span_u = paraU.first;
            int span_v = paraV.first;

            Matrix4d mat_coeff_u = ComputeNonUniformBsplineMatrix(span_u, knots_u);
            Matrix4d mat_coeff_v = ComputeNonUniformBsplineMatrix(span_v, knots_v);
            double dt_u = knots_u[span_u + 1] - knots_u[span_u];
            double u = (dt_u > 1e-9) ? (paraU.second - knots_u[span_u]) / dt_u : 0.0;
            Vector4d U_vec;
            U_vec << 1.0, u, u * u, u * u * u;
            RowVector4d w_u = U_vec.transpose() * mat_coeff_u;
            double dt_v = knots_v[span_v + 1] - knots_v[span_v];
            double v = (dt_v > 1e-9) ? (paraV.second - knots_v[span_v]) / dt_v : 0.0;
            Vector4d V_vec;
            V_vec << 1.0, v, v * v, v * v * v;
            RowVector4d w_v = V_vec.transpose() * mat_coeff_v;
            for (int l = 0; l < 4; ++l) {
                for (int m = 0; m < 4; ++m)
                {
                    int flat_index = (span_u -3 + l) * controls_num_v + span_v - 3 + m;
                    if (flat_index < 0 || flat_index >= controls.size()) {
                        continue;
                    }
                    active_cp_pointers.push_back(controls[flat_index].data());
                    active_weights.push_back(w_u(l)*w_v(m));
                }
            }

            ceres::CostFunction* cost_func = new BSplineSDMErr(givepoints[i], surf_info, active_weights);
            ceres::LossFunction* loss = new ceres::HuberLoss(0.05);
            problem.AddResidualBlock(cost_func, loss, active_cp_pointers);
            // 1. U 方向平滑 (行约束)
            // 遍历每一行，对中间的点加约束
        }
        sum_data_res += ms_since(t1);
        t1 = std::chrono::high_resolution_clock::now();
        double smooth_weight = 0.03;
        for (int i = 0; i < controls_num_u; ++i) {
            for (int j = 1; j < controls_num_v - 1; ++j) {
                // 获取连续三个点的索引
                int idx_prev = i * controls_num_v + (j - 1);
                int idx_curr = i * controls_num_v + j;
                int idx_next = i * controls_num_v + (j + 1);

                problem.AddResidualBlock(
                    BSplineSmoothnessErr::Create(smooth_weight),
                    nullptr, // 核函数 (nullptr 表示不开鲁棒核)
                    controls[idx_prev].data(), // P(i, j-1)
                    controls[idx_curr].data(), // P(i, j)
                    controls[idx_next].data()  // P(i, j+1)
                );
            }
        }

        // 2. V 方向平滑 (列约束)
        // 遍历每一列，对中间的点加约束
        for (int j = 0; j < controls_num_v; ++j) {
            for (int i = 1; i < controls_num_u - 1; ++i) {
                // 获取连续三个点的索引 (跨行取点)
                int idx_prev = (i - 1) * controls_num_v + j;
                int idx_curr = i * controls_num_v + j;
                int idx_next = (i + 1) * controls_num_v + j;

                problem.AddResidualBlock(
                    BSplineSmoothnessErr::Create(smooth_weight),
                    nullptr,
                    controls[idx_prev].data(), // P(i-1, j)
                    controls[idx_curr].data(), // P(i, j)
                    controls[idx_next].data()  // P(i+1, j)
                );
            }
        }

        sum_smooth += ms_since(t1);

        t1 = std::chrono::high_resolution_clock::now();

        Vector3d min_p(min_x, min_y, min_z);
        Vector3d max_p(max_x, max_y, max_z);
        double bound_w = 10.0; // 可调
        for (int i = 0; i < controls.size(); ++i) {
            ceres::CostFunction* b = new ceres::AutoDiffCostFunction<BoundaryPenalty,3,3>(
                new BoundaryPenalty(min_p, max_p, bound_w)
            );
            problem.AddResidualBlock(b, nullptr, controls[i].data());
        }

        sum_bound += ms_since(t1);
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        options.num_threads = 4;
        options.max_num_iterations = 1; // 关键点！
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        t1 = std::chrono::high_resolution_clock::now();
        ceres::Solve(options, &problem, &summary);
        sum_solve += ms_since(t1);
        t1 = std::chrono::high_resolution_clock::now();
        //std::cout << summary.FullReport() << std::endl;
        vector<Vector3d> controls_copy = controls; // <--- ✅ 先克隆一份
        setNewControl(controls_copy, controls_num_u, controls_num_v);
        sum_set += ms_since(t1);
    }
    // std::cout << std::fixed << std::setprecision(2)
    //       << "sum_fp=" << sum_fp
    //       << "ms sum_pre=" << sum_pre
    //       << "ms sum_data_res=" << sum_data_res
    //       << "ms sum_smooth=" << sum_smooth
    //       << "ms sum_bound=" << sum_bound
    //       << "ms sum_solve=" << sum_solve
    //       << "ms sum_set=" << sum_set
    //       << "ms" << std::endl;
    // 在 apply 函数的 return last_error; 之前加入：


    return 1.0;
}

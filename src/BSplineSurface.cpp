//
// Created by albus on 2026/1/18.
//
#include <fstream>
#include <ANN/ANN.h>

#include "BSpline.h"
#include "BSplineSDMErr.h"

using namespace Eigen;

namespace {

constexpr int kBsplineMaxCp = 15;

Eigen::Matrix4d g_bspline_mat_cache[16][16];

static void makeClampedUniformKnots(int num_cp, std::vector<double>& knots) {
    knots.resize(num_cp + 4);
    const double denom = static_cast<double>(num_cp - 3);
    for (int i = 0; i < static_cast<int>(knots.size()); ++i) {
        if (i <= 3) {
            knots[i] = 0.0;
        } else if (i >= num_cp) {
            knots[i] = 1.0;
        } else {
            knots[i] = static_cast<double>(i - 3) / denom;
        }
    }
}

static Eigen::Matrix4d computeBsplineMatrix(int i, const std::vector<double>& knots) {
    Eigen::Matrix4d M = Eigen::Matrix4d::Zero();
    double t_i   = knots[i];
    double t_im1 = knots[i - 1];
    double t_ip1 = knots[i + 1];
    double t_ip2 = knots[i + 2];
    double t_ip3 = knots[i + 3];

    double dt_i_im1   = t_i - t_im1;
    double dt_ip1_i   = t_ip1 - t_i;
    double dt_ip1_im1 = t_ip1 - t_im1;
    double dt_ip1_im2 = t_ip1 - knots[i - 2];
    double dt_ip2_im1 = t_ip2 - t_im1;
    double dt_ip2_i   = t_ip2 - t_i;
    double dt_ip3_i   = t_ip3 - t_i;

    double m00 = (dt_ip1_i * dt_ip1_i) / (dt_ip1_im1 * dt_ip1_im2);
    double m02 = (dt_i_im1 * dt_i_im1) / (dt_ip2_im1 * dt_ip1_im1);
    double m22 = 3.0 * (dt_ip1_i * dt_ip1_i) / (dt_ip2_im1 * dt_ip1_im1);
    double m33 = (dt_ip1_i * dt_ip1_i) / (dt_ip3_i * dt_ip2_i);
    double m12 = 3.0 * dt_ip1_i * dt_i_im1 / (dt_ip2_im1 * dt_ip1_im1);

    M(0, 0) = m00;
    M(0, 1) = 1.0 - m00 - m02;
    M(0, 2) = m02;
    M(0, 3) = 0.0;

    M(1, 0) = -3.0 * m00;
    M(1, 1) = 3.0 * m00 - m12;
    M(1, 2) = m12;
    M(1, 3) = 0.0;

    M(2, 0) = 3.0 * m00;
    M(2, 1) = -3.0 * m00 - m22;
    M(2, 2) = m22;
    M(2, 3) = 0.0;

    M(3, 0) = -m00;
    double term_extra = (dt_ip1_i * dt_ip1_i) / (dt_ip2_i * dt_ip2_im1);
    M(3, 2) = -m22 / 3.0 - m33 - term_extra;
    M(3, 1) = m00 - M(3, 2) - m33;
    M(3, 3) = m33;

    return M;
}

static void initBsplineMatrixCache() {
    for (int num_cp = 4; num_cp <= kBsplineMaxCp; ++num_cp) {
        std::vector<double> knots;
        makeClampedUniformKnots(num_cp, knots);
        for (int span = 3; span < num_cp; ++span) {
            g_bspline_mat_cache[num_cp][span] = computeBsplineMatrix(span, knots);
        }
    }
}

struct BsplineMatrixCacheInit {
    BsplineMatrixCacheInit() { initBsplineMatrixCache(); }
};
static BsplineMatrixCacheInit g_bspline_mat_cache_init;

static const Eigen::Matrix4d& cachedBsplineMatrix(int num_cp, int span) {
    if (num_cp >= 4 && num_cp <= kBsplineMaxCp && span >= 3 && span < num_cp) {
        return g_bspline_mat_cache[num_cp][span];
    }
    static thread_local Eigen::Matrix4d fallback;
    std::vector<double> knots;
    makeClampedUniformKnots(num_cp, knots);
    fallback = computeBsplineMatrix(span, knots);
    return fallback;
}

static int numCpFromKnots(const std::vector<double>& knots) {
    return static_cast<int>(knots.size()) - 4;
}

}  // namespace

SurfaceEval BSplineSurface::evaluateSurface(
        const Parameter& paraU, const Parameter& paraV,
        const vector<double>& knotsU, const vector<double>& knotsV,
        const std::vector<Vector3d>& controls, int num_cp_v) const
{
    SurfaceEval eval;

    const int ki_u = paraU.first;
    const int ki_v = paraV.first;
    const Matrix4d& matU = cachedBsplineMatrix(numCpFromKnots(knotsU), ki_u);
    const Matrix4d& matV = cachedBsplineMatrix(numCpFromKnots(knotsV), ki_v);

    const double dt_u = knotsU[ki_u + 1] - knotsU[ki_u];
    const double u = (dt_u > 1e-9) ? (paraU.second - knotsU[ki_u]) / dt_u : 0.0;
    const double dt_v = knotsV[ki_v + 1] - knotsV[ki_v];
    const double v = (dt_v > 1e-9) ? (paraV.second - knotsV[ki_v]) / dt_v : 0.0;

    const double u2 = u * u, u3 = u2 * u;
    const double v2 = v * v, v3 = v2 * v;

    Vector4d B0_u, B1_u, B2_u, B0_v, B1_v, B2_v;
    B0_u << 1.0, u, u2, u3;
    B1_u << 0.0, 1.0, 2.0 * u, 3.0 * u2;
    B2_u << 0.0, 0.0, 2.0, 6.0 * u;
    B0_v << 1.0, v, v2, v3;
    B1_v << 0.0, 1.0, 2.0 * v, 3.0 * v2;
    B2_v << 0.0, 0.0, 2.0, 6.0 * v;

    const double inv_dt_u  = (dt_u > 1e-9) ? (1.0 / dt_u) : 0.0;
    const double inv_dt2_u = inv_dt_u * inv_dt_u;
    const double inv_dt_v  = (dt_v > 1e-9) ? (1.0 / dt_v) : 0.0;
    const double inv_dt2_v = inv_dt_v * inv_dt_v;

    eval.w_pos_u = B0_u.transpose() * matU;
    const RowVector4d w_du_u  = (B1_u.transpose() * matU) * inv_dt_u;
    const RowVector4d w_duu_u = (B2_u.transpose() * matU) * inv_dt2_u;
    eval.w_pos_v = B0_v.transpose() * matV;
    const RowVector4d w_dv_v  = (B1_v.transpose() * matV) * inv_dt_v;
    const RowVector4d w_dvv_v = (B2_v.transpose() * matV) * inv_dt2_v;

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const int flat_idx = (ki_u - 3 + i) * num_cp_v + (ki_v - 3 + j);
            if (flat_idx < 0 || flat_idx >= static_cast<int>(controls.size())) {
                continue;
            }
            const Vector3d& cp = controls[flat_idx];
            const double w_pos = eval.w_pos_u(i) * eval.w_pos_v(j);
            eval.pos += w_pos * cp;
            eval.Su  += (w_du_u(i)  * eval.w_pos_v(j)) * cp;
            eval.Sv  += (eval.w_pos_u(i) * w_dv_v(j)) * cp;
            eval.Suu += (w_duu_u(i) * eval.w_pos_v(j)) * cp;
            eval.Svv += (eval.w_pos_u(i) * w_dvv_v(j)) * cp;
            eval.Suv += (w_du_u(i)  * w_dv_v(j)) * cp;
        }
    }
    return eval;
}

SurfaceCurvature BSplineSurface::curvatureFromEval(const SurfaceEval& eval) const
{
    SurfaceCurvature result;
    const Vector3d& Su = eval.Su;
    const Vector3d& Sv = eval.Sv;
    const Vector3d& Suu = eval.Suu;
    const Vector3d& Svv = eval.Svv;
    const Vector3d& Suv = eval.Suv;

    Vector3d normal_raw = Su.cross(Sv);
    double area = normal_raw.norm();

    if (area < 1e-9) {
        result.point = eval.pos;
        result.normal = Vector3d::UnitZ();
        result.tangent1 = Vector3d::UnitX();
        result.tangent2 = Vector3d::UnitY();
        result.E = 1; result.G = 1; result.F = 0;
        result.L = 0; result.M = 0; result.N = 0;
        result.k1 = 0; result.k2 = 0; result.K = 0; result.H = 0;
        return result;
    }
    result.normal = normal_raw / area;

    result.E = Su.dot(Su);
    result.F = Su.dot(Sv);
    result.G = Sv.dot(Sv);

    result.L = Suu.dot(result.normal);
    result.M = Suv.dot(result.normal);
    result.N = Svv.dot(result.normal);

    double det_I = result.E * result.G - result.F * result.F;
    if (std::abs(det_I) < 1e-9) {
        result.point = eval.pos;
        result.K = 0; result.H = 0; result.k1 = 0; result.k2 = 0;
        return result;
    }

    double det_II = result.L * result.N - result.M * result.M;

    result.K = det_II / det_I;
    result.H = (result.E * result.N + result.G * result.L - 2 * result.F * result.M) / (2 * det_I);

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

    Vector3d T2 = result.normal.cross(T1).normalized();

    result.tangent1 = T1;
    result.tangent2 = T2;
    result.point = eval.pos;
    return result;
}

Vector3d BSplineSurface::getPos(const Parameter& paraU, const Parameter& paraV, const vector<double>& knotsU, const vector<double>& knotsV, const std::vector<Vector3d>& controls, int num_cp_v) {
    return evaluateSurface(paraU, paraV, knotsU, knotsV, controls, num_cp_v).pos;
}

Vector3d BSplineSurface::getFirstDiff(const Parameter& paraU, const Parameter& paraV, const vector<double>& knotsU, const vector<double>& knotsV, const std::vector<Vector3d>& controls,int num_cp_v, bool is_diff_u)
{
    const SurfaceEval eval = evaluateSurface(paraU, paraV, knotsU, knotsV, controls, num_cp_v);
    return is_diff_u ? eval.Su : eval.Sv;
}

Vector3d BSplineSurface::getSecondDiff(const Parameter& paraU, const Parameter& paraV, const vector<double>& knotsU, const vector<double>& knotsV, const std::vector<Vector3d>& controls,int num_cp_v, int type) {
    const SurfaceEval eval = evaluateSurface(paraU, paraV, knotsU, knotsV, controls, num_cp_v);
    if (type == 0) return eval.Suu;
    if (type == 1) return eval.Svv;
    return eval.Suv;
}


SurfaceCurvature BSplineSurface::getCurvature(const Parameter& paraU, const Parameter& paraV, const vector<double>& knotsU, const vector<double>& knotsV, const std::vector<Vector3d>& controls,int num_cp_v) {
    return curvatureFromEval(evaluateSurface(paraU, paraV, knotsU, knotsV, controls, num_cp_v));
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
    // 用 PCA 投影冷启动 + Newton 精化，不再依赖 positions[] 采样点扫描
    footPrints.clear();
    coldInitUVFromPCA(givepoints, footPrints);
    return findFootPrintWarm(givepoints, footPrints, point_dists, /*newton_steps=*/6);
}

std::pair<BSplineSurface::Parameter, BSplineSurface::Parameter> BSplineSurface:: getPara(int index) {
    if (index < 0 || index >= (int)sampling_paras_.size()) {
        return {Parameter(0, 0.0), Parameter(0, 0.0)};
    }
    return sampling_paras_[index];
}

// ─── UV 工具：给定全局参数 t ∈ [0,1]，找所在 knot span ───────────────────────
static int findSpanGlobal(double t, const std::vector<double>& knots, int num_cp) {
    if (t >= 1.0 - 1e-9) return num_cp - 1;
    for (int k = 3; k < num_cp; ++k)
        if (knots[k] <= t && t < knots[k + 1]) return k;
    return 3;
}

// ─── PCA 冷启动：把 givepoints 投影到 PCA 平面，映射到 [0,1] BSpline 参数 ───
void BSplineSurface::coldInitUVFromPCA(const vector<Vector3d>& givepoints,
                                       vector<pair<Parameter,Parameter>>& uv_out) const
{
    const int n = static_cast<int>(givepoints.size());
    uv_out.resize(n);

    // 若 plane_frame_ 无效，用域中心作为 fallback
    const double u_fallback = 0.5, v_fallback = 0.5;
    const int span_u_fb = findSpanGlobal(u_fallback, knots_u, controls_num_u);
    const int span_v_fb = findSpanGlobal(v_fallback, knots_v, controls_num_v);
    const Parameter para_u_fb(span_u_fb, u_fallback);
    const Parameter para_v_fb(span_v_fb, v_fallback);

    if (!plane_frame_.valid
        || (plane_frame_.u_hi - plane_frame_.u_lo) < 1e-9
        || (plane_frame_.v_hi - plane_frame_.v_lo) < 1e-9)
    {
        for (auto& uv : uv_out) uv = {para_u_fb, para_v_fb};
        return;
    }

    const double inv_u = 1.0 / (plane_frame_.u_hi - plane_frame_.u_lo);
    const double inv_v = 1.0 / (plane_frame_.v_hi - plane_frame_.v_lo);

    for (int i = 0; i < n; ++i) {
        Vector3d diff = givepoints[i] - plane_frame_.centroid;
        double u_proj = diff.dot(plane_frame_.u_axis);
        double v_proj = diff.dot(plane_frame_.v_axis);
        double u_norm = std::max(0.0, std::min(1.0, (u_proj - plane_frame_.u_lo) * inv_u));
        double v_norm = std::max(0.0, std::min(1.0, (v_proj - plane_frame_.v_lo) * inv_v));
        int span_u = findSpanGlobal(u_norm, knots_u, controls_num_u);
        int span_v = findSpanGlobal(v_norm, knots_v, controls_num_v);
        uv_out[i] = {Parameter(span_u, u_norm), Parameter(span_v, v_norm)};
    }
}

// ─── warm-start findFootPrint：从 uv_state 出发做 Newton 精化 ────────────────
double BSplineSurface::findFootPrintWarm(const vector<Vector3d>& givepoints,
                                         vector<pair<Parameter,Parameter>>& uv_state,
                                         vector<double>& point_dists,
                                         int newton_steps)
{
    if (uv_state.empty()) coldInitUVFromPCA(givepoints, uv_state);
    const int n = static_cast<int>(givepoints.size());
    point_dists.resize(n, 0.0);
    double squareSum = 0.0;

    for (int i = 0; i < n; ++i) {
        const Vector3d& p = givepoints[i];
        auto [paraU, paraV] = uv_state[i];

        for (int nr = 0; nr < newton_steps; ++nr) {
            const SurfaceEval eval = evaluateSurface(paraU, paraV, knots_u, knots_v, controls, controls_num_v);
            Vector3d r = eval.pos - p;

            double a00 = eval.Su.dot(eval.Su), a01 = eval.Su.dot(eval.Sv), a11 = eval.Sv.dot(eval.Sv);
            double b0  = -r.dot(eval.Su), b1  = -r.dot(eval.Sv);
            double det = a00 * a11 - a01 * a01;
            if (std::abs(det) < 1e-10) break;

            double du = (b0 * a11 - b1 * a01) / det;
            double dv = (a00 * b1 - a01 * b0) / det;

            double new_tf_u = std::max(0.0, std::min(1.0, paraU.second + du));
            double new_tf_v = std::max(0.0, std::min(1.0, paraV.second + dv));
            paraU = {findSpanGlobal(new_tf_u, knots_u, controls_num_u), new_tf_u};
            paraV = {findSpanGlobal(new_tf_v, knots_v, controls_num_v), new_tf_v};

            if (std::abs(du) < 1e-6 && std::abs(dv) < 1e-6) break;
        }

        uv_state[i] = {paraU, paraV};
        const SurfaceEval final_eval = evaluateSurface(paraU, paraV, knots_u, knots_v, controls, controls_num_v);
        double dist_sq = (final_eval.pos - p).squaredNorm();
        point_dists[i] = dist_sq;
        squareSum += std::sqrt(dist_sq);
    }
    return squareSum;
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

    // ----- 3. (u, v) 矩形 + margin -----
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

    // ----- 4. 预计算所有输入点在 PCA 空间的 (u, v, h) 投影 -----
    const int N_pts = static_cast<int>(cloud->points.size());
    vector<double> pts_u(N_pts), pts_v(N_pts), pts_h(N_pts);
    for (int k = 0; k < N_pts; ++k) {
        Vector3d d(cloud->points[k].x - plane_frame_.centroid.x(),
                   cloud->points[k].y - plane_frame_.centroid.y(),
                   cloud->points[k].z - plane_frame_.centroid.z());
        pts_u[k] = d.dot(plane_frame_.u_axis);
        pts_v[k] = d.dot(plane_frame_.v_axis);
        pts_h[k] = d.dot(plane_frame_.n_axis);
    }

    // ----- 5. 每个控制点网格位置 -> PCA 2D 最近邻 -> 用其法向投影高度 -----
    for (int i = 0; i < num_u; ++i) {
        for (int j = 0; j < num_v; ++j) {
            const double cu = u_lo + i * u_step;
            const double cv = v_lo + j * v_step;
            double best_d2 = std::numeric_limits<double>::max();
            double h = plane_frame_.h_avg;
            for (int k = 0; k < N_pts; ++k) {
                const double du = pts_u[k] - cu, dv = pts_v[k] - cv;
                const double d2 = du * du + dv * dv;
                if (d2 < best_d2) { best_d2 = d2; h = pts_h[k]; }
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
    controls = controlPs;
    controls_num_u = num_u;
    controls_num_v = num_v;
    positions.clear();
    sampling_paras_.clear();
    span_sample_index_.clear();
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
    return computeBsplineMatrix(i, knots);
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
        // 为 UV warm-start 补充 PCA frame（地面 grid cell 走 ext_init 路径）
        computePlaneFrame(points);
        if (plane_frame_.valid) {
            const double mr = 0.15;
            const double ur = plane_frame_.u_max - plane_frame_.u_min;
            const double vr = plane_frame_.v_max - plane_frame_.v_min;
            plane_frame_.u_lo = plane_frame_.u_min - mr * ur;
            plane_frame_.u_hi = plane_frame_.u_max + mr * ur;
            plane_frame_.v_lo = plane_frame_.v_min - mr * vr;
            plane_frame_.v_hi = plane_frame_.v_max + mr * vr;
        }
    } else {
        controlPs.resize(expected);
        initControlPoint(points, controlPs, controls_num_u, controls_num_v);
    }
    double t_init = ms_since(t0); t0 = std::chrono::high_resolution_clock::now();

    buildRangeGrid(points, 50);
    double t_grid = ms_since(t0); t0 = std::chrono::high_resolution_clock::now();
    setKnotParams(controls_num_u, controls_num_v);
    setNewControl(controlPs, controls_num_u, controls_num_v);
    double t_set = ms_since(t0); t0 = std::chrono::high_resolution_clock::now();

    double last_error = 1e9;
    vector<Vector3d> givepoints;
    pclToEigenVector(points, givepoints);

    // UV warm-start：第一次迭代用 PCA 投影冷启动，后续复用上次精化结果
    vector<pair<Parameter, Parameter>> uv_cache;
    coldInitUVFromPCA(givepoints, uv_cache);

    int point_num = static_cast<int>(givepoints.size());
    bool stop_flag = false;
    double sum_fp = 0.0, sum_pre = 0.0, sum_data_res = 0.0;
    double sum_smooth = 0.0, sum_bound = 0.0, sum_solve = 0.0;
    for(int iter = 0; iter < maxIterNum; ++iter) {
        ceres::Problem problem;
        vector<double> point_dists;
        auto t1 = std::chrono::high_resolution_clock::now();
        // uv_cache 既是输入（warm start）也是输出（精化后结果），下次迭代自动复用
        double current_sq_dist = findFootPrintWarm(givepoints, uv_cache, point_dists, /*newton_steps=*/5);
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
            if (!stop_flag) { stop_flag = true; }
            else { break; }
        }

        // 策略2: 绝对精度满足要求 (RMSE < 1cm)
        if (rmse < 1e-2) {
            if (!stop_flag) { stop_flag = true; }
            else { break; }
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
        for(int i = 0; i < (int)uv_cache.size(); i++)
        {
            active_weights.clear();
            active_cp_pointers.clear();
            if (point_dists[i] > inlier_thresh) {
                continue; // 直接当噪声，跳过
            }
            Parameter paraU = uv_cache[i].first, paraV = uv_cache[i].second;
            const SurfaceEval eval = evaluateSurface(paraU, paraV, knots_u, knots_v, controls, controls_num_v);
            SurfaceCurvature surf_info = curvatureFromEval(eval);

            const int span_u = paraU.first;
            const int span_v = paraV.first;
            for (int l = 0; l < 4; ++l) {
                for (int m = 0; m < 4; ++m)
                {
                    int flat_index = (span_u -3 + l) * controls_num_v + span_v - 3 + m;
                    if (flat_index < 0 || flat_index >= controls.size()) {
                        continue;
                    }
                    active_cp_pointers.push_back(controls[flat_index].data());
                    active_weights.push_back(eval.w_pos_u(l) * eval.w_pos_v(m));
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
        // Ceres 直接在 controls[].data() 上修改，无需再调 setNewControl 重建 positions[]
    }

    // 把最终控制点提交（同时清空过时的 positions[]）
    setNewControl(controls, controls_num_u, controls_num_v);
    return 1.0;
}

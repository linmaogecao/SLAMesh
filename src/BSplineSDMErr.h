//
// Created by albus on 2026/1/26.
//

#ifndef SPLINE_FITTING_BSPLINESDMERR_H
#define SPLINE_FITTING_BSPLINESDMERR_H
#include <ceres/ceres.h>
#include <Eigen/Core>
#include "BSpline.h"
class BSplineSDMErr : public ceres::CostFunction {
public:
    BSplineSDMErr(const Eigen::Vector3d& target_point,
                  const SurfaceCurvature& frame,
                  const std::vector<double>& basis_weights)
        : target_point_(target_point), frame_(frame), weights_(basis_weights)
    {
        // 1. 设置残差维度：3维 (T1方向, T2方向, N方向)
        set_num_residuals(3);

        // 2. 设置参数块维度：根据传入的权重数量决定有多少个控制点
        // 每个控制点都是 3维 (x,y,z)
        for (size_t i = 0; i < weights_.size(); ++i) {
            mutable_parameter_block_sizes()->push_back(3);
        }

        // 3. 预计算公式 (7) 的权重系数 (跟之前一样)
        double dist = (target_point - frame.point).dot(frame.normal);
        double eps = 1e-4;

        auto calc_coeff = [&](double rho) -> double {

            double denom = dist - rho;
            if (std::abs(denom) < eps) return 0.0;
            double val = dist / denom;
            if (val > 100.0) return 10.0;
            return (val > 0) ? std::sqrt(val) : 0.0;
        };

        coeff_t1_ = calc_coeff(1.0/frame.k1);
        coeff_t2_ = calc_coeff(1.0/frame.k2);
        coeff_n_  = 1.0;
    }

    // 核心计算函数
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const override {

        size_t num_cps = weights_.size();

        // --- 1. 计算 P_plus (当前控制点下的曲面点) ---
        Eigen::Vector3d P_plus = Eigen::Vector3d::Zero();
        for (size_t i = 0; i < num_cps; ++i) {
            Eigen::Vector3d cp(parameters[i][0], parameters[i][1], parameters[i][2]);
            P_plus += weights_[i] * cp;
        }
        // --- 2. 计算残差 ---
        Eigen::Vector3d diff = P_plus - target_point_;

        // 投影到局部坐标系
        double x1 = diff.dot(frame_.tangent1);
        double x2 = diff.dot(frame_.tangent2);
        double x3 = diff.dot(frame_.normal);

        residuals[0] = coeff_t1_ * x1;
        residuals[1] = coeff_t2_ * x2;
        residuals[2] = coeff_n_  * x3;
        // --- 3. 计算雅可比 ---
        if (jacobians) {
            for (size_t i = 0; i < num_cps; ++i) {
                if (jacobians[i]) {
                    double w = weights_[i];
                    jacobians[i][0] = coeff_t1_ * frame_.tangent1.x() * w;
                    jacobians[i][1] = coeff_t1_ * frame_.tangent1.y() * w;
                    jacobians[i][2] = coeff_t1_ * frame_.tangent1.z() * w;

                    // Row 1: 对 Residual[1] (T2方向) 求导
                    jacobians[i][3] = coeff_t2_ * frame_.tangent2.x() * w;
                    jacobians[i][4] = coeff_t2_ * frame_.tangent2.y() * w;
                    jacobians[i][5] = coeff_t2_ * frame_.tangent2.z() * w;

                    // Row 2: 对 Residual[2] (N方向) 求导
                    jacobians[i][6] = coeff_n_ * frame_.normal.x() * w;
                    jacobians[i][7] = coeff_n_ * frame_.normal.y() * w;
                    jacobians[i][8] = coeff_n_ * frame_.normal.z() * w;
                }
            }
        }

        return true;
    }

private:
    Eigen::Vector3d target_point_;
    SurfaceCurvature frame_;
    std::vector<double> weights_;
    double coeff_t1_, coeff_t2_, coeff_n_;
};

struct BSplineSmoothnessErr {
    BSplineSmoothnessErr(double weight) : weight_(weight) {}

    template <typename T>
    bool operator()(const T* const p_prev, // 前一个点 P(i-1)
                    const T* const p_curr, // 当前点   P(i)
                    const T* const p_next, // 后一个点 P(i+1)
                    T* residuals) const {
        // 公式： weight * (P_prev - 2*P_curr + P_next)
        // 也就是离散化的二阶导数
        residuals[0] = T(weight_) * (p_prev[0] - T(2.0) * p_curr[0] + p_next[0]);
        residuals[1] = T(weight_) * (p_prev[1] - T(2.0) * p_curr[1] + p_next[1]);
        residuals[2] = T(weight_) * (p_prev[2] - T(2.0) * p_curr[2] + p_next[2]);
        return true;
    }

    // 工厂函数：告诉 Ceres 这是一个 3维残差，输入是 3个 3维向量
    static ceres::CostFunction* Create(double weight) {
        return new ceres::AutoDiffCostFunction<BSplineSmoothnessErr, 3, 3, 3, 3>(
            new BSplineSmoothnessErr(weight));
    }

    double weight_;
};
struct BSplineFirstOrderErr {
    BSplineFirstOrderErr(double weight) : weight_(weight) {}

    template <typename T>
    bool operator()(const T* const p_curr, // 当前点 P(i)
                    const T* const p_next, // 下一点 P(i+1)
                    T* residuals) const {
        // 公式： weight * (P_next - P_curr)
        // 也就是离散化的一阶导数（差分）
        // 物理意义：最小化控制点之间的距离，产生“张力”
        residuals[0] = T(weight_) * (p_next[0] - p_curr[0]);
        residuals[1] = T(weight_) * (p_next[1] - p_curr[1]);
        residuals[2] = T(weight_) * (p_next[2] - p_curr[2]);
        return true;
    }

    // 工厂函数：输入是 2个 3维向量
    static ceres::CostFunction* Create(double weight) {
        return new ceres::AutoDiffCostFunction<BSplineFirstOrderErr, 3, 3, 3>(
            new BSplineFirstOrderErr(weight));
    }

    double weight_;
};

struct BoundaryPenalty {
    BoundaryPenalty(const Eigen::Vector3d& min_p, const Eigen::Vector3d& max_p, double w)
        : min_p_(min_p), max_p_(max_p), w_(w) {}

    template <typename T>
    bool operator()(const T* const cp, T* residual) const {
        for (int k = 0; k < 3; ++k) {
            T v = cp[k];
            T minv = T(min_p_[k]);
            T maxv = T(max_p_[k]);
            T r = T(0);
            if (v < minv) r = minv - v;
            else if (v > maxv) r = v - maxv;
            residual[k] = T(w_) * r;
        }
        return true;
    }

    Eigen::Vector3d min_p_, max_p_;
    double w_;
};


// -----------------------------------------------------------------------
// 配准用: 曲率加权点到曲面残差，优化变量为 SE(3) 位姿 [qx,qy,qz,qw, tx,ty,tz]
// 残差 3 维: (coeff_t1 * T1^T, coeff_t2 * T2^T, N^T) * (R*p_local + t - S*)
// -----------------------------------------------------------------------
class SDMRegistrationCostFunction : public ceres::SizedCostFunction<3, 7> {
public:
    // curr_point      : 传感器坐标系下的点 (用于 Evaluate 中的 q*p+t)
    // curr_point_world: 当前 T_curr 下变换到世界系的点 (只用于 dist 预计算)
    // frame           : 曲面最近点处的曲率帧 (世界系)
    SDMRegistrationCostFunction(const Eigen::Vector3d& curr_point,
                                const Eigen::Vector3d& curr_point_world,
                                const SurfaceCurvature& frame)
        : curr_point_(curr_point), surface_point_(frame.point),
          tangent1_(frame.tangent1), tangent2_(frame.tangent2), normal_(frame.normal)
    {
        // dist 用世界坐标系下的点到曲面的有符号距离
        double dist = (curr_point_world - frame.point).dot(frame.normal);
        double eps = 1e-4;

        auto calc_coeff = [&](double k) -> double {
            if(std::abs(k) < eps) return 0.0;
            double rho = 1.0 / k;
            double denom = dist - rho;
            if(std::abs(denom) < eps) return 0.0;
            double val = dist / denom;
            if(val > 100.0) return 10.0;
            return (val > 0) ? std::sqrt(val) : 0.0;
        };

        coeff_t1_ = calc_coeff(frame.k1);
        coeff_t2_ = calc_coeff(frame.k2);
        coeff_n_  = 1.0;
    }

    virtual ~SDMRegistrationCostFunction() {}

    virtual bool Evaluate(double const *const *parameters,
                          double *residuals,
                          double **jacobians) const override {
        Eigen::Map<const Eigen::Quaterniond> q(parameters[0]);
        Eigen::Map<const Eigen::Vector3d>    t(parameters[0] + 4);

        Eigen::Vector3d point_w = q * curr_point_ + t;
        Eigen::Vector3d diff = point_w - surface_point_;

        residuals[0] = coeff_t1_ * diff.dot(tangent1_);
        residuals[1] = coeff_t2_ * diff.dot(tangent2_);
        residuals[2] = coeff_n_  * diff.dot(normal_);

        if(jacobians && jacobians[0]){
            // dp_w / d_xi = [-[p_w]_x, I]  (3x6, 李代数左扰动)
            Eigen::Matrix3d skew_pw;
            skew_pw <<        0, -point_w.z(),  point_w.y(),
                     point_w.z(),          0, -point_w.x(),
                    -point_w.y(),  point_w.x(),          0;

            Eigen::Matrix<double, 3, 6> dp_by_xi;
            dp_by_xi.block<3,3>(0,0) = -skew_pw;
            dp_by_xi.block<3,3>(0,3) = Eigen::Matrix3d::Identity();

            // A = [coeff_t1 * T1^T; coeff_t2 * T2^T; coeff_n * N^T]  (3x3)
            Eigen::Matrix<double, 3, 3> A;
            A.row(0) = coeff_t1_ * tangent1_.transpose();
            A.row(1) = coeff_t2_ * tangent2_.transpose();
            A.row(2) = coeff_n_  * normal_.transpose();

            // J = A * dp_by_xi  (3x6), 最后一列补0到 3x7
            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> J(jacobians[0]);
            J.setZero();
            J.block<3,6>(0,0) = A * dp_by_xi;
        }

        return true;
    }

private:
    Eigen::Vector3d curr_point_;     // 雷达局部坐标系下的点
    Eigen::Vector3d surface_point_;  // 曲面最近点 (世界系)
    Eigen::Vector3d tangent1_;       // 曲面主方向 T1 (世界系)
    Eigen::Vector3d tangent2_;       // 曲面主方向 T2 (世界系)
    Eigen::Vector3d normal_;         // 曲面法向 (世界系)
    double coeff_t1_, coeff_t2_, coeff_n_;
};

// -----------------------------------------------------------------------
// 地面匹配专用：大法向权重（约束 roll/pitch/高度），小切向权重（不干扰 x/y/yaw）
// w_n : 法向残差权重（建议 5.0）
// w_t : 切向残差权重（建议 0.1，近似关掉切向贡献）
// -----------------------------------------------------------------------
class GroundSDMRegistrationCostFunction : public ceres::SizedCostFunction<3, 7> {
public:
    GroundSDMRegistrationCostFunction(const Eigen::Vector3d& curr_point,
                                      const Eigen::Vector3d& curr_point_world,
                                      const SurfaceCurvature& frame,
                                      double w_n = 5.0,
                                      double w_t = 0.1)
        : curr_point_(curr_point), surface_point_(frame.point),
          tangent1_(frame.tangent1), tangent2_(frame.tangent2), normal_(frame.normal),
          w_n_(w_n), w_t_(w_t) {}

    virtual ~GroundSDMRegistrationCostFunction() {}

    virtual bool Evaluate(double const *const *parameters,
                          double *residuals,
                          double **jacobians) const override {
        Eigen::Map<const Eigen::Quaterniond> q(parameters[0]);
        Eigen::Map<const Eigen::Vector3d>    t(parameters[0] + 4);

        Eigen::Vector3d point_w = q * curr_point_ + t;
        Eigen::Vector3d diff = point_w - surface_point_;

        residuals[0] = w_t_ * diff.dot(tangent1_);
        residuals[1] = w_t_ * diff.dot(tangent2_);
        residuals[2] = w_n_ * diff.dot(normal_);

        if (jacobians && jacobians[0]) {
            Eigen::Matrix3d skew_pw;
            skew_pw <<        0, -point_w.z(),  point_w.y(),
                     point_w.z(),          0, -point_w.x(),
                    -point_w.y(),  point_w.x(),          0;

            Eigen::Matrix<double, 3, 6> dp_by_xi;
            dp_by_xi.block<3,3>(0,0) = -skew_pw;
            dp_by_xi.block<3,3>(0,3) = Eigen::Matrix3d::Identity();

            Eigen::Matrix<double, 3, 3> A;
            A.row(0) = w_t_ * tangent1_.transpose();
            A.row(1) = w_t_ * tangent2_.transpose();
            A.row(2) = w_n_ * normal_.transpose();

            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> J(jacobians[0]);
            J.setZero();
            J.block<3,6>(0,0) = A * dp_by_xi;
        }
        return true;
    }

private:
    Eigen::Vector3d curr_point_;
    Eigen::Vector3d surface_point_;
    Eigen::Vector3d tangent1_;
    Eigen::Vector3d tangent2_;
    Eigen::Vector3d normal_;
    double w_n_, w_t_;
};

// -----------------------------------------------------------------------
// 地面约束（雷达系表述）: 3 个残差，优化变量为 SE(3) 位姿 [qx,qy,qz,qw, tx,ty,tz]
//
// 物理假设：车贴在当地路面上 → 地面法向在雷达系里应竖直向上，地面高度 z_l ≈ z_ground_ref。
// 上坡时世界系地面法向/高度会变，但雷达系下这两个量仍应稳定。
//
// 在当前 Ceres 步内，用线性化锚定位姿 (R_a, t_a) 把 PCA 地面法向/质心变到世界系并固定：
//   n_w_a = R_a * n_l,   c_w_a = R_a * c_l + t_a
// 优化 (R, t) 时，变回雷达系检查：
//   n_lidar = R^T * n_w_a
//   c_lidar = R^T * (c_w_a - t)
//
//   r[0] = w_rp * n_lidar.x   →  当地地面法向在雷达系水平 (roll/pitch)
//   r[1] = w_rp * n_lidar.y
//   r[2] = w_z  * (c_lidar.z - z_ground_ref)  →  雷达系地面高度
//
// n_local     : 传感器系 PCA 地面法向 (z>0)
// c_local     : 传感器系 PCA 地面质心
// n_world_a   : 锚定位姿下的 n_w_a = R_a * n_l
// c_world_a   : 锚定位姿下的 c_w_a = R_a * c_l + t_a
// z_ground_ref: 雷达系地面参考高度 (第一帧 z_l 均值, 约 -1.73m)
// -----------------------------------------------------------------------
class GroundPlaneConstraint : public ceres::SizedCostFunction<3, 7> {
public:
    GroundPlaneConstraint(const Eigen::Vector3d& n_local,
                          const Eigen::Vector3d& c_local,
                          double z_ground_ref,
                          const Eigen::Vector3d& n_world_a,
                          const Eigen::Vector3d& c_world_a,
                          double w_rp = 1.0,
                          double w_z  = 1.0)
        : n_local_(n_local), c_local_(c_local),
          z_ground_ref_(z_ground_ref),
          n_world_a_(n_world_a), c_world_a_(c_world_a),
          w_rp_(w_rp), w_z_(w_z) {}

    virtual ~GroundPlaneConstraint() {}

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const override {
        Eigen::Map<const Eigen::Quaterniond> q(parameters[0]);
        Eigen::Map<const Eigen::Vector3d>    t(parameters[0] + 4);

        // 世界系锚定地面 → 当前雷达系
        const Eigen::Vector3d n_lidar = q.conjugate() * n_world_a_;
        const Eigen::Vector3d c_lidar = q.conjugate() * (c_world_a_ - t);

        residuals[0] = w_rp_ * n_lidar.x();
        residuals[1] = w_rp_ * n_lidar.y();
        residuals[2] = w_z_  * (c_lidar.z() - z_ground_ref_);

        if (jacobians && jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> J(jacobians[0]);
            J.setZero();

            const Eigen::Matrix3d R = q.normalized().toRotationMatrix();

            // n_lidar = R^T n_w_a,  d(n_lidar)/dω = [n_lidar]_×
            J(0, 0) = 0;
            J(0, 1) = w_rp_ *  n_lidar.z();
            J(0, 2) = w_rp_ * -n_lidar.y();

            J(1, 0) = w_rp_ * -n_lidar.z();
            J(1, 1) = 0;
            J(1, 2) = w_rp_ *  n_lidar.x();

            // c_lidar = R^T(c_w_a - t)
            // d(c_lidar.z)/dω = e_z^T [c_lidar]_× = [-c_y, c_x, 0]
            J(2, 0) = w_z_ * (-c_lidar.y());
            J(2, 1) = w_z_ * ( c_lidar.x());
            J(2, 2) = 0;
            // d(c_lidar.z)/d t = -e_z^T R^T  →  -R(j,2) for t_j
            J(2, 3) = w_z_ * (-R(0, 2));
            J(2, 4) = w_z_ * (-R(1, 2));
            J(2, 5) = w_z_ * (-R(2, 2));
        }
        return true;
    }

private:
    Eigen::Vector3d n_local_;
    Eigen::Vector3d c_local_;
    double z_ground_ref_;
    Eigen::Vector3d n_world_a_;
    Eigen::Vector3d c_world_a_;
    double w_rp_, w_z_;
};

#endif //SPLINE_FITTING_BSPLINESDMERR_H
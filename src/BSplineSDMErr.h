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
    SDMRegistrationCostFunction(const Eigen::Vector3d& curr_point,
                                const Eigen::Vector3d& surface_point,
                                const SurfaceCurvature& frame)
        : curr_point_(curr_point), surface_point_(frame.point),
          tangent1_(frame.tangent1), tangent2_(frame.tangent2), normal_(frame.normal)
    {
        double dist = (curr_point - frame.point).dot(frame.normal);
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

#endif //SPLINE_FITTING_BSPLINESDMERR_H
#pragma once


#include <Eigen/Core>
#include <vector>
#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl/io/pcd_io.h>
//#include <tool.h>
#include <map>
#include <chrono>
#include <iomanip>
//#include <readWrite.h>

using namespace std;


struct SurfaceCurvature {
    Eigen::Vector3d point;
    Eigen::Vector3d normal;
    double E, F, G;
    double L, M, N;
    double K;     // Gaussian
    double H;     // Mean
    Eigen::Vector3d tangent1; // 主方向 T1
    Eigen::Vector3d tangent2; // 主方向 T2
    double k1, k2;
};

// 同一 (u,v) 下一次求值：position 基函数权重 + pos/各阶导
struct SurfaceEval {
    Eigen::RowVector4d w_pos_u, w_pos_v;
    Eigen::Vector3d pos = Eigen::Vector3d::Zero();
    Eigen::Vector3d Su  = Eigen::Vector3d::Zero();
    Eigen::Vector3d Sv  = Eigen::Vector3d::Zero();
    Eigen::Vector3d Suu = Eigen::Vector3d::Zero();
    Eigen::Vector3d Svv = Eigen::Vector3d::Zero();
    Eigen::Vector3d Suv = Eigen::Vector3d::Zero();
};

class BSplineSurface {
public:
    typedef std::pair<int, double> Parameter;

    // PCA 平面坐标系: 用于在斜/竖墙面上做参数化, 替代硬编码的 xy/xz/yz 平面
    //   centroid   : 点云质心 (PCA 原点)
    //   u_axis     : 最大方差方向 (参数 u)
    //   v_axis     : 次大方差方向 (参数 v)
    //   n_axis     : 最小方差方向 (近似法向)
    //   u_min/max  : 所有点在 u_axis 上投影的范围
    //   v_min/max  : 所有点在 v_axis 上投影的范围
    //   h_avg      : 所有点在 n_axis 上投影的平均值 (一般 ~0, 但保留以备旋转/平移不规整的输入)
    struct PlaneFrame {
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        Eigen::Vector3d u_axis   = Eigen::Vector3d::UnitX();
        Eigen::Vector3d v_axis   = Eigen::Vector3d::UnitY();
        Eigen::Vector3d n_axis   = Eigen::Vector3d::UnitZ();
        double u_min = 0.0, u_max = 0.0;   // raw point projection range (no margin)
        double v_min = 0.0, v_max = 0.0;
        double h_avg = 0.0;
        double u_lo = 0.0, u_hi = 0.0;    // BSpline parameterization range (with margin)
        double v_lo = 0.0, v_hi = 0.0;
        bool   valid = false;
    };
    BSplineSurface(int deg_u,int deg_v,int control_num_u,int control_num_v):
        Deg_u(deg_u),Deg_v(deg_v),controls_num_u(control_num_u),controls_num_v(control_num_v)
    {
        knots_u.resize(deg_u+control_num_u+1);
        knots_v.resize(deg_v+control_num_v+1);
        for (int i = 0; i <= deg_u+control_num_u; i++) {
            if (i <= deg_u)
                knots_u[i] = 0;
            else if (i >= control_num_u)
                knots_u[i] = 1;
            else
                knots_u[i] = (double)(i-deg_u)/(double)(control_num_u-deg_u);
        }
        for (int i = 0; i <= deg_v+control_num_v; i++) {
            if (i <= deg_v)
                knots_v[i] = 0;
            else if (i >= control_num_v)
                knots_v[i] = 1;
            else
                knots_v[i] = (double)(i-deg_v)/(double)(control_num_v-deg_v);
        }
    }

    ~BSplineSurface(){
        clear();
    }

    // size_t nb_control_u() const {return controls_u.size();}
    // size_t nb_control_v() const {return controls_v.size();}

    /**
     *  calculate the position of current parameter
     *  ref to <<General Matrix Representations for B-Splines>> remap the tf into [ti,ti+1]
     * @param para
     * @param knots intervals
     * @param controls control points set
     * @return
     */
    Eigen::Vector3d getPos(const Parameter& paraU, const Parameter& paraV, const vector<double>& knotsU, const vector<double>& knotsV, const std::vector<Eigen::Vector3d>& controls,int num_cp_v);
    SurfaceCurvature getCurvature(const Parameter& paraU, const Parameter& paraV, const vector<double>& knotsU, const vector<double>& knotsV, const std::vector<Eigen::Vector3d>& controls,int num_cp_v);
    double findFootPrint(const vector<Eigen::Vector3d>& givepoints,vector<pair<Parameter, Parameter>>& footPrints, vector<double> &point_dists);
    // UV warm-start 版本: uv_state 作为输入初始值并被更新为精化后的 (u,v)。
    // 首次调用时若 uv_state 为空，自动从 PCA 平面投影做冷启动。
    // out_evals 非空时写入每点末次完整 SurfaceEval（含二阶导），供 apply 复用曲率。
    double findFootPrintWarm(const vector<Eigen::Vector3d>& givepoints,
                             vector<pair<Parameter,Parameter>>& uv_state,
                             vector<double>& point_dists,
                             int newton_steps = 5,
                             vector<SurfaceEval>* out_evals = nullptr);
    void initControlPoint(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,vector<Eigen::Vector3d>& controlPs,int num_u,int num_v);
    void initControlPointPCA(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                             vector<Eigen::Vector3d>& controlPs, int num_u, int num_v,
                             double margin_ratio = 0.15);

    // 在 apply() 之前调用，提供外部算好的初始控制点（如 range-image 行列采样结果）。
    // 大小须等于 num_u * num_v；若不匹配则回退到内部 initControlPoint。
    void setExternalInitControls(const std::vector<Eigen::Vector3d>& init_cp) {
        ext_init_controls_ = init_cp;
    }
    const PlaneFrame& getPlaneFrame() const { return plane_frame_; }
    void setNewControl(const vector<Eigen::Vector3d> &controlPs, int num_u, int num_v,bool isCut = false);
    void setKnotParams(int num_cp_u,int num_cp_v);
    void pclToEigenVector(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, std::vector<Eigen::Vector3d>& out_vec);
    bool apply(pcl::PointCloud<pcl::PointXYZ>::Ptr& points,int maxIterNum,double alpha,double gama,double eplison);
    // 非空时 apply() 每次结束打印并追加一行 profile（用于第一帧建图诊断）
    static void setApplyProfileLogPath(const std::string& path);
    static void setApplyProfileLabel(const std::string& label);
    static void clearApplyProfileLog();
    const vector<Eigen::Vector3d>& getControls() const{return controls;}
    const vector<double>& getKnotsU() const{return knots_u;}
    const vector<double>& getKnotsV() const{return knots_v;}
    int getNumCpV() const{return controls_num_v;}
    int getNumCpU() const{return controls_num_u;}
    // apply() 收尾时的平均点到面距离（米，非 RMSE）。<0 表示未拟合过。
    // 这是该片自身的几何可信度：墙面能到 2~3 cm，植被/灌木/动车是几十 cm，
    // 而配准里两者的残差权重目前完全相同。
    double getFitMeanDist() const { return fit_mean_dist_; }
    // 同上但乘了 sqrt(n/(n-p)) 无偏化因子，抵消过参数化对样本内残差的低估
    double getFitMeanDistAdj() const { return fit_mean_dist_adj_; }
private:

    void clear(){
        controls.clear();
    }

    // 把 givepoints 投影到 PCA 平面，映射到 [0,1] B-spline 参数空间，作为 UV 冷启动。
    void coldInitUVFromPCA(const vector<Eigen::Vector3d>& givepoints,
                           vector<pair<Parameter,Parameter>>& uv_out) const;

    /** follow <<General Matrix Representations for B-Splines>> calculate BSpline coeff matrix
     *
     * @param i in which param interal  [ti,ti+1]
     * @param knots knots intervals
     * @return
     */
    Eigen::Matrix4d ComputeNonUniformBsplineMatrix(int i, const vector<double>& knots);

    SurfaceEval evaluateSurface(const Parameter& paraU, const Parameter& paraV,
                                const vector<double>& knotsU, const vector<double>& knotsV,
                                const std::vector<Eigen::Vector3d>& controls, int num_cp_v,
                                bool second_order = true) const;
    SurfaceCurvature curvatureFromEval(const SurfaceEval& eval) const;

    void computePlaneFrame(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
public:
    int controls_num_u;
    int controls_num_v;
private:
    std::vector<Eigen::Vector3d> controls;
    std::vector<double> knots_u;
    std::vector<double> knots_v;
    int Deg_u;
    int Deg_v;

    // 输入点云的 AABB，apply() 里的控制点边界约束用
    double max_x,max_y,max_z;
    double min_x,min_y,min_z;
    PlaneFrame plane_frame_;
    std::vector<Eigen::Vector3d> ext_init_controls_;
    double fit_mean_dist_ = -1.0;
    double fit_mean_dist_adj_ = -1.0;
};

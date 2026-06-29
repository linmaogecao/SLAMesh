#pragma once


#include <Eigen/Core>
#include <vector>
#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/centroid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/surface/concave_hull.h>
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

struct CurvatureCenters {
    Eigen::Vector3d center1;
    Eigen::Vector3d center2;
    bool is_planar;
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
    struct LocalRange {
        double min_x, max_x;
        double min_y, max_y;
        double min_z, max_z;
        bool has_data = false;
    };

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
    BSplineSurface(int deg_u,int deg_v,int control_num_u,int control_num_v,double interal=0.01):
        interal_u(interal),interal_v(interal),Deg_u(deg_u),Deg_v(deg_v),controls_num_u(control_num_u),controls_num_v(control_num_v)
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
        interal_ = interal;
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
    Eigen::Vector3d getFirstDiff(const Parameter& paraU, const Parameter& paraV, const vector<double>& knotsU, const vector<double>& knotsV, const std::vector<Eigen::Vector3d>& controls,int num_cp_v, bool is_diff_u);
    Eigen::Vector3d getSecondDiff(const Parameter& paraU, const Parameter& paraV, const vector<double>& knotsU, const vector<double>& knotsV, const std::vector<Eigen::Vector3d>& controls,int num_cp_v, int type);
    SurfaceCurvature getCurvature(const Parameter& paraU, const Parameter& paraV, const vector<double>& knotsU, const vector<double>& knotsV, const std::vector<Eigen::Vector3d>& controls,int num_cp_v);
    Eigen::Vector3d getTangent(const Parameter& para, const vector<double> &knots,const std::vector<Eigen::Vector3d> &controls);
    Eigen::Vector3d getNormal(const Parameter& para, const vector<double> &knots,const std::vector<Eigen::Vector3d> &controls);
    Eigen::Vector3d getCurvCenter(const Parameter& para, const vector<double> &knots,const std::vector<Eigen::Vector3d> &controls);
    double findFootPrint(const vector<Eigen::Vector3d>& givepoints,vector<pair<Parameter, Parameter>>& footPrints, vector<double> &point_dists);
    // UV warm-start 版本: uv_state 作为输入初始值并被更新为精化后的 (u,v)。
    // 首次调用时若 uv_state 为空，自动从 PCA 平面投影做冷启动。
    double findFootPrintWarm(const vector<Eigen::Vector3d>& givepoints,
                             vector<pair<Parameter,Parameter>>& uv_state,
                             vector<double>& point_dists,
                             int newton_steps = 5);
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
    pair<Parameter, Parameter> getPara(int index);
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
    bool isPointValid(const Eigen::Vector3d& p);
    const vector<Eigen::Vector3d>& getSamples() const{return positions;}
    void buildRangeGrid(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, int grid_res);
private:

    void clear(){
        controls.clear();
        positions.clear();
        sampling_paras_.clear();
        span_sample_index_.clear();
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
                                const std::vector<Eigen::Vector3d>& controls, int num_cp_v) const;
    SurfaceCurvature curvatureFromEval(const SurfaceEval& eval) const;

    void computePlaneFrame(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
public:
    int controls_num_u;
    int controls_num_v;
private:
    double interal_u;
    double interal_v;
    std::vector<Eigen::Vector3d> controls;
    std::vector<Eigen::Vector3d> positions;
    std::vector<double> knots_u;
    std::vector<double> knots_v;
    int Deg_u;
    int Deg_v;

    double interal_;
    double max_x,max_y,max_z;
    double min_x,min_y,min_z;
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud_;
    vector<pair<Parameter, Parameter>> sampling_paras_;
    std::vector<std::vector<LocalRange>> range_grid_;
    int grid_res_x_ = 0;
    int grid_res_y_ = 0;
    double grid_cell_size_x_ = 0;
    double grid_cell_size_y_ = 0;
    double grid_origin_x_ = 0;
    double grid_origin_y_ = 0;
    int cn1 = 0;
    int cn2 = 0;
    int cn3 = 0;
    PlaneFrame plane_frame_;
    // Span-indexed sample lookup built in setNewControl.
    // span_sample_index_[si][sj] holds position indices for span (si+3, sj+3).
    std::vector<std::vector<std::vector<int>>> span_sample_index_;
    std::vector<Eigen::Vector3d> ext_init_controls_;
};

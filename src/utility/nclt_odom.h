#pragma once
#ifndef NCLT_ODOM_H_
#define NCLT_ODOM_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>

// 读 NCLT 的 odometry_mu_100hz.csv："utime,x,y,z,roll,pitch,yaw"，约 100Hz，
// 从每段 session 起点算起的绝对位姿，欧拉序 ZYX。
//
// 该里程计是 NCLT 用 EKF 融合轮式编码器、KVH 光纤陀螺与 GX3 IMU 得到的，
// 所以不需要再单独读 kvh.csv / ms25.csv，那些数据已经在里面。
//
// 三个必须注意的性质（都在 dataset 上实测确认过）：
//   1. z 列恒等于 0——NCLT 明确说 Segway 的传感器无法观测高度。调用方不要用
//      增量的 z 分量。
//   2. yaw 是累积值不缠绕（实测跨度 -40.6 ~ +1.7 rad），与 kvh.csv 的
//      [0,2π) 缠绕角不同，读进来不需要解缠。
//   3. 采样间隔不均匀（实测 1000 ~ 60883 us），必须按时间戳插值，不能按
//      索引偏移取邻居。
//
// 位姿存的是 NCLT body 系（x 前、y 右、z 下），而 SLAMesh 用 x 前、y 左、
// z 上。readKitti 已经把点云翻到后者，位姿必须做相应的相似变换才能匹配：
// 记 F = diag(1,-1,-1,1)（绕 x 轴 180°，F 自逆），则 T_slamesh = F * T_nclt * F。
// 只在一侧乘会让旋转与平移失配。PIN-SLAM 处理 NCLT groundtruth 用的是同一形式。
//
// 用法上只取两次查询之间的相对变换：绝对值在 5.5km 的 session 上会漂到不可用，
// 但 100ms 的帧间增量很准，正好适合当配准初值。
class NcltOdometry {
public:
    // 失败时返回 false 并保持 empty()，调用方应回退到匀速外推
    bool load(const std::string& path);

    // utime 落在数据覆盖区间外返回 false。T 已经是 SLAMesh 系。
    bool poseAt(int64_t utime, Eigen::Matrix4d& T) const;

    bool   empty() const { return samples_.empty(); }
    size_t size()  const { return samples_.size(); }
    int64_t frontUtime() const { return samples_.empty() ? 0 : samples_.front().utime; }
    int64_t backUtime()  const { return samples_.empty() ? 0 : samples_.back().utime; }

private:
    struct Sample {
        int64_t utime;
        double  x, y, z, roll, pitch, yaw;
    };
    std::vector<Sample> samples_;
};

#endif  // NCLT_ODOM_H_

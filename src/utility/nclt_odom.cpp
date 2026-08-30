#include "nclt_odom.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

namespace {

// F = diag(1,-1,-1,1)：NCLT body(x前 y右 z下) <-> SLAMesh(x前 y左 z上)
const Eigen::Matrix4d& flipMatrix() {
    static const Eigen::Matrix4d F = Eigen::Vector4d(1.0, -1.0, -1.0, 1.0).asDiagonal();
    return F;
}

Eigen::Matrix3d rpyToMatrix(double roll, double pitch, double yaw) {
    // ZYX，与 tf::Matrix3x3::setRPY 一致（节点其余部分用的就是它）
    return (Eigen::AngleAxisd(yaw,   Eigen::Vector3d::UnitZ())
          * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())
          * Eigen::AngleAxisd(roll,  Eigen::Vector3d::UnitX())).toRotationMatrix();
}

}  // namespace

bool NcltOdometry::load(const std::string& path) {
    samples_.clear();
    std::ifstream file(path);
    if (!file.good()) {
        std::cerr << "[NCLT] cannot open odometry file: " << path << std::endl;
        return false;
    }

    std::string line;
    size_t rejected = 0;
    samples_.reserve(700000);
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        const char* p = line.c_str();
        char* end = nullptr;
        const long long ut = std::strtoll(p, &end, 10);
        if (end == p) { ++rejected; continue; }
        p = end;

        double v[6];
        bool ok = true;
        for (double& val : v) {
            while (*p == ',' || *p == ' ') ++p;
            val = std::strtod(p, &end);
            if (end == p || !std::isfinite(val)) { ok = false; break; }
            p = end;
        }
        if (!ok) { ++rejected; continue; }

        // 二分查询要求严格递增
        if (!samples_.empty() && ut <= samples_.back().utime) { ++rejected; continue; }

        Sample s;
        s.utime = (int64_t)ut;
        s.x = v[0]; s.y = v[1]; s.z = v[2];
        s.roll = v[3]; s.pitch = v[4]; s.yaw = v[5];
        samples_.push_back(s);
    }

    if (samples_.size() < 2) {
        std::cerr << "[NCLT] odometry file has too few usable rows: " << path << std::endl;
        samples_.clear();
        return false;
    }
    std::cout << "[NCLT] odometry loaded: " << samples_.size() << " samples, "
              << (samples_.back().utime - samples_.front().utime) / 1e6 << " s"
              << (rejected ? (" (" + std::to_string(rejected) + " rows rejected)") : "")
              << std::endl;
    return true;
}

bool NcltOdometry::poseAt(int64_t utime, Eigen::Matrix4d& T) const {
    if (samples_.size() < 2) return false;
    if (utime < samples_.front().utime || utime > samples_.back().utime) return false;

    const auto it = std::upper_bound(samples_.begin(), samples_.end(), utime,
                                     [](int64_t t, const Sample& s) { return t < s.utime; });
    size_t hi = (size_t)(it - samples_.begin());
    if (hi == 0) hi = 1;
    if (hi >= samples_.size()) hi = samples_.size() - 1;

    const Sample& a = samples_[hi - 1];
    const Sample& b = samples_[hi];
    const double span = (double)(b.utime - a.utime);
    const double w = (span > 0.0) ? (double)(utime - a.utime) / span : 0.0;

    const Eigen::Quaterniond qa(rpyToMatrix(a.roll, a.pitch, a.yaw));
    const Eigen::Quaterniond qb(rpyToMatrix(b.roll, b.pitch, b.yaw));

    Eigen::Matrix4d T_nclt = Eigen::Matrix4d::Identity();
    T_nclt.block<3, 3>(0, 0) = qa.slerp(w, qb).toRotationMatrix();
    T_nclt(0, 3) = a.x + w * (b.x - a.x);
    T_nclt(1, 3) = a.y + w * (b.y - a.y);
    T_nclt(2, 3) = a.z + w * (b.z - a.z);

    T = flipMatrix() * T_nclt * flipMatrix();
    return T.allFinite();
}

#ifndef SLAMESH_GROUND_MATCH_POLICY_H
#define SLAMESH_GROUND_MATCH_POLICY_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ground_match_policy {

// 8 档 r_signed 距离 bin：后方延伸到 -50m，前远仍保留但默认可 cap=0
// [-50,-30) [-30,-20) [-20,-10) [-10,0) | [0,5) [5,10) [10,20) [20,40]
constexpr int kDistanceBinCount = 8;

struct Candidate {
    int scan_idx;
    double x;   // r_signed，用于 distanceBin
    double y;   // 雷达系 y，左右分桶
    double bx = 0.0;  // 雷达系 x，方位去重
    double by = 0.0;  // 雷达系 y（与 y 相同，便于显式传 body 坐标）
};

inline int distanceBin(double x)
{
    if (x >= -50.0 && x < -30.0) return 0;  // 后极远
    if (x >= -30.0 && x < -20.0) return 1;  // 后远
    if (x >= -20.0 && x < -10.0) return 2;  // 后中
    if (x >= -10.0 && x < 0.0) return 3;    // 后近
    if (x >= 0.0 && x < 5.0) return 4;      // 前近
    if (x >= 5.0 && x < 10.0) return 5;     // 前中近
    if (x >= 10.0 && x < 20.0) return 6;    // 前中
    if (x >= 20.0 && x <= 40.0) return 7;   // 前远
    return -1;
}

// 方位扇区 [0, n_bins)：同一扇区视为同一条 scan 线邻域
inline int azimuthBin(double bx, double by, int n_bins)
{
    if (n_bins <= 0) return 0;
    const double a = std::atan2(by, bx);  // [-pi, pi]
    int k = static_cast<int>(std::floor((a + M_PI) / (2.0 * M_PI) * n_bins));
    if (k >= n_bins) k -= n_bins;
    if (k < 0) k += n_bins;
    return k;
}

// 每个方位扇区只留第一个点（候选大致按点云顺序，等价于同环连续点只留一个）
inline std::vector<Candidate> dedupeByAzimuth(const std::vector<Candidate>& in, int n_bins)
{
    if (n_bins <= 0 || in.empty()) return in;
    std::vector<char> seen(static_cast<size_t>(n_bins), 0);
    std::vector<Candidate> out;
    out.reserve(std::min(in.size(), static_cast<size_t>(n_bins)));
    for (const auto& c : in) {
        const int k = azimuthBin(c.bx, c.by, n_bins);
        if (seen[static_cast<size_t>(k)]) continue;
        seen[static_cast<size_t>(k)] = 1;
        out.push_back(c);
    }
    return out;
}

inline void appendEvenly(const std::vector<Candidate>& source,
                         int count,
                         std::vector<Candidate>& output)
{
    if (count <= 0 || source.empty()) return;
    if (count >= static_cast<int>(source.size())) {
        output.insert(output.end(), source.begin(), source.end());
        return;
    }
    for (int i = 0; i < count; ++i) {
        const std::size_t pos =
            static_cast<std::size_t>((i * static_cast<long long>(source.size())) / count);
        output.push_back(source[pos]);
    }
}

// azim_bins>0：每个距离 bin×左右桶内先按方位去重，再按 cap 均匀抽样
inline std::vector<Candidate> selectStratified(
    const std::vector<Candidate>& candidates,
    const std::array<int, kDistanceBinCount>& caps,
    int azim_bins = 360)
{
    std::array<std::array<std::vector<Candidate>, 2>, kDistanceBinCount> buckets;
    for (const auto& candidate : candidates) {
        const int bin = distanceBin(candidate.x);
        if (bin < 0) continue;
        buckets[bin][candidate.y < 0.0 ? 0 : 1].push_back(candidate);
    }

    std::vector<Candidate> selected;
    for (int bin = 0; bin < kDistanceBinCount; ++bin) {
        const int cap = std::max(0, caps[bin]);
        const auto left_u  = dedupeByAzimuth(buckets[bin][0], azim_bins);
        const auto right_u = dedupeByAzimuth(buckets[bin][1], azim_bins);

        const int half = cap / 2;
        int left_count = std::min(half, static_cast<int>(left_u.size()));
        int right_count = std::min(half, static_cast<int>(right_u.size()));
        int remaining = cap - left_count - right_count;

        const int left_available = static_cast<int>(left_u.size()) - left_count;
        const int add_left = std::min(remaining, left_available);
        left_count += add_left;
        remaining -= add_left;

        const int right_available = static_cast<int>(right_u.size()) - right_count;
        right_count += std::min(remaining, right_available);

        appendEvenly(left_u, left_count, selected);
        appendEvenly(right_u, right_count, selected);
    }
    return selected;
}

inline double normalizedPointWeight(
    int bin,
    const std::array<int, kDistanceBinCount>& match_counts)
{
    // 后极远 > 后远 > 后中 > 后近；前中 ≥ 前中近 > 前近；前远(无图)=0
    static constexpr std::array<double, kDistanceBinCount> kBinPriors{
        {0.18, 0.16, 0.14, 0.12, 0.11, 0.14, 0.15, 0.0}};
    static constexpr double kRefPrior = 0.12;
    if (bin < 0 || bin >= kDistanceBinCount) return 1.0;
    if (match_counts[bin] <= 0) return 1.0;
    if (kBinPriors[bin] <= 0.0) return 0.0;

    int active_bins = 0;
    int total_matches = 0;
    for (int i = 0; i < kDistanceBinCount; ++i) {
        if (match_counts[i] <= 0 || kBinPriors[i] <= 0.0) continue;
        ++active_bins;
        total_matches += match_counts[i];
    }
    if (active_bins <= 0 || total_matches <= 0) return 1.0;

    const double prior = kBinPriors[bin] / kRefPrior;
    const double avg_per_bin =
        static_cast<double>(total_matches) / static_cast<double>(active_bins);
    const double sparse_boost = std::min(
        1.5, std::max(1.0, avg_per_bin / static_cast<double>(match_counts[bin])));
    const double weight = prior * sparse_boost;
    return std::max(0.5, std::min(weight, 2.5));
}

}  // namespace ground_match_policy

#endif  // SLAMESH_GROUND_MATCH_POLICY_H

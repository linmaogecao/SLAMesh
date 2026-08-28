#!/usr/bin/env python3
"""从 SLAMesh 导出的控制点离线重建全局 B-spline 地图（障碍 + 地面 + 轨迹）。

求值逻辑逐行对齐 src/BSplineSurface.cpp，输出可直接拖进 CloudCompare / MeshLab：

    map_mesh.ply    障碍 + 地面三角网格 + 轨迹管道，一个文件看全貌
    map_cloud.ply   障碍 + 地面曲面采样点云
    traj.ply        轨迹管道单独一份

输入（--build-dir 下，即 C++ 里的 kBsplineBuildDir）：

    controls/<sid>.txt               障碍控制点，无表头，n*n 行 "x y z"
    gnd_controls/<层>_<ix>_<iy>.txt  地面控制点，首行 "n_u n_v"，其后 "x y z"
    bspline_traj_xyz.txt             每帧 "x y z"

用法：

    python render_bspline_map.py --build-dir ~/catkin_ws/src/SLAMesh/build
    python render_bspline_map.py --build-dir build --res 16 --color type
"""

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

DEG = 3  # 三次 B-spline，全工程唯一取值（BSplineSurface(3,3,n,n)）


# ─── B-spline 求值：对齐 src/BSplineSurface.cpp ──────────────────────────────

def clamped_uniform_knots(n_cp):
    """makeClampedUniformKnots（BSplineSurface.cpp:21）：长度 n_cp+4，前后各 4 个 0/1。"""
    knots = np.empty(n_cp + 4)
    denom = float(n_cp - DEG)
    for i in range(n_cp + 4):
        if i <= DEG:
            knots[i] = 0.0
        elif i >= n_cp:
            knots[i] = 1.0
        else:
            knots[i] = (i - DEG) / denom
    return knots


def bspline_matrix(i, knots):
    """computeBsplineMatrix（BSplineSurface.cpp:35）：第 i 段的 4x4 基矩阵。"""
    t_im2, t_im1, t_i = knots[i - 2], knots[i - 1], knots[i]
    t_ip1, t_ip2, t_ip3 = knots[i + 1], knots[i + 2], knots[i + 3]

    dt_i_im1 = t_i - t_im1
    dt_ip1_i = t_ip1 - t_i
    dt_ip1_im1 = t_ip1 - t_im1
    dt_ip1_im2 = t_ip1 - t_im2
    dt_ip2_im1 = t_ip2 - t_im1
    dt_ip2_i = t_ip2 - t_i
    dt_ip3_i = t_ip3 - t_i

    m00 = (dt_ip1_i ** 2) / (dt_ip1_im1 * dt_ip1_im2)
    m02 = (dt_i_im1 ** 2) / (dt_ip2_im1 * dt_ip1_im1)
    m22 = 3.0 * (dt_ip1_i ** 2) / (dt_ip2_im1 * dt_ip1_im1)
    m33 = (dt_ip1_i ** 2) / (dt_ip3_i * dt_ip2_i)
    m12 = 3.0 * dt_ip1_i * dt_i_im1 / (dt_ip2_im1 * dt_ip1_im1)

    M = np.zeros((4, 4))
    M[0] = (m00, 1.0 - m00 - m02, m02, 0.0)
    M[1] = (-3.0 * m00, 3.0 * m00 - m12, m12, 0.0)
    M[2] = (3.0 * m00, -3.0 * m00 - m22, m22, 0.0)
    term_extra = (dt_ip1_i ** 2) / (dt_ip2_i * dt_ip2_im1)
    M[3, 0] = -m00
    M[3, 2] = -m22 / 3.0 - m33 - term_extra
    M[3, 1] = m00 - M[3, 2] - m33
    M[3, 3] = m33
    return M


def find_span(t, knots, n_cp):
    """findSpanGlobal（BSplineSurface.cpp:310）。"""
    if t >= 1.0 - 1e-9:
        return n_cp - 1
    for k in range(DEG, n_cp):
        if knots[k] <= t < knots[k + 1]:
            return k
    return DEG


def weight_matrix(n_cp, res):
    """(res, n_cp) 稠密权重阵：第 a 行是 t=a/(res-1) 处各控制点的系数。

    C++ 里每次求值只累加 4 个控制点，这里摊成整行的零填充稠密阵，方便把同尺寸的
    曲面堆成一个批次做 einsum，比逐点循环快两个量级。
    """
    knots = clamped_uniform_knots(n_cp)
    ts = np.linspace(0.0, 1.0, res)
    W = np.zeros((res, n_cp))
    mats = {}
    for a, t in enumerate(ts):
        span = find_span(t, knots, n_cp)
        if span not in mats:
            mats[span] = bspline_matrix(span, knots)
        dt = knots[span + 1] - knots[span]
        u = (t - knots[span]) / dt if dt > 1e-9 else 0.0
        W[a, span - DEG:span + 1] = np.array([1.0, u, u * u, u ** 3]) @ mats[span]
    return W


def eval_batch(grids, n_u, n_v, res):
    """grids (K, n_u, n_v, 3) → 采样点 (K, res, res, 3)。控制点行主序，见 cpp:206。"""
    Wu = weight_matrix(n_u, res)
    Wv = Wu if n_v == n_u else weight_matrix(n_v, res)
    tmp = np.einsum("au,kuvc->kavc", Wu, grids, optimize=True)
    return np.einsum("bv,kavc->kabc", Wv, tmp, optimize=True)


# ─── 读控制点 ───────────────────────────────────────────────────────────────

def _tokens(path):
    """整文件读进来再切词。np.loadtxt 是逐行 Python 循环，这里快约 1.7 倍。"""
    with open(path, "rb") as fp:
        return fp.read().split()


def load_obs(path):
    """障碍面无表头，靠控制点总数反推 n：全工程构造均为 (3,3,n,n)。"""
    vals = np.array(_tokens(path), dtype=np.float64)
    if vals.size % 3:
        raise ValueError(f"{vals.size} 个数值不是 3 的倍数")
    cps = vals.reshape(-1, 3)
    n = int(round(np.sqrt(len(cps))))
    if n * n != len(cps):
        raise ValueError(f"{len(cps)} 个控制点不是完全平方数，无法反推 n_u/n_v")
    return cps, n, n


def load_gnd(path):
    tok = _tokens(path)
    if len(tok) < 2:
        raise ValueError("文件为空或缺少 n_u/n_v 表头")
    n_u, n_v = int(tok[0]), int(tok[1])
    vals = np.array(tok[2:], dtype=np.float64)
    if vals.size != n_u * n_v * 3:
        raise ValueError(f"表头 {n_u}x{n_v} 与 {vals.size // 3} 个控制点不符")
    return vals.reshape(-1, 3), n_u, n_v


def collect(pattern, loader, tag, limit=0, workers=8):
    """读一批控制点文件，按 (n_u, n_v) 分组返回 {(n_u,n_v): (K,n_u,n_v,3)}。

    上万个几百字节的小文件，冷缓存下几乎全是磁盘寻道延迟（实测 13 ms/文件）。
    read() 期间会释放 GIL，所以线程池能把这些延迟叠起来。
    """
    paths = sorted(glob.glob(pattern))
    if limit > 0:
        paths = paths[:limit]
    if not paths:
        print(f"  {tag}: 0 张曲面")
        return {}

    def safe(path):
        try:
            return path, loader(path), None
        except Exception as exc:  # 单个坏文件不该毁掉整次重建
            return path, None, exc

    if workers > 1 and len(paths) > 64:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(safe, paths))
    else:
        results = [safe(p) for p in paths]

    groups, bad = {}, []
    for path, loaded, exc in results:
        if exc is not None:
            bad.append(f"{os.path.basename(path)}: {exc}")
            continue
        cps, n_u, n_v = loaded
        if n_u < 4 or n_v < 4:
            bad.append(f"{os.path.basename(path)}: 控制点数 {n_u}x{n_v} 小于 4")
            continue
        groups.setdefault((n_u, n_v), []).append(cps.reshape(n_u, n_v, 3))

    n_ok = sum(len(v) for v in groups.values())
    print(f"  {tag}: {n_ok} 张曲面"
          + (f"，跳过 {len(bad)} 个" if bad else "")
          + (f"（尺寸分布 {sorted((k[0], len(v)) for k, v in groups.items())}）" if groups else ""))
    for msg in bad[:5]:
        print(f"    ! {msg}")
    if len(bad) > 5:
        print(f"    ! ...另有 {len(bad) - 5} 个同类问题")

    return {k: np.asarray(v) for k, v in groups.items()}


def tessellate(groups, res):
    """把分好组的控制网格求值成顶点 + 三角面（每片一个 res×res 网格）。"""
    if not groups:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)

    verts, faces, base = [], [], 0
    # 单片的面索引模板，平移 base 即可复用
    cell = np.arange(res * res).reshape(res, res)
    tl, tr = cell[:-1, :-1].ravel(), cell[:-1, 1:].ravel()
    br, bl = cell[1:, 1:].ravel(), cell[1:, :-1].ravel()
    tmpl = np.concatenate([np.stack([tl, tr, br], 1), np.stack([tl, br, bl], 1)])

    for (n_u, n_v), grids in sorted(groups.items()):
        pts = eval_batch(grids, n_u, n_v, res)          # (K, res, res, 3)
        k = len(grids)
        verts.append(pts.reshape(-1, 3))
        offs = base + np.arange(k) * (res * res)
        faces.append((tmpl[None, :, :] + offs[:, None, None]).reshape(-1, 3))
        base += k * res * res

    return np.concatenate(verts), np.concatenate(faces)


# ─── 轨迹管道 ───────────────────────────────────────────────────────────────

def traj_tube(P, radius, sides=8):
    """把折线扫成三角管。单纯写点的话轨迹在网格里几乎看不见。"""
    keep = np.ones(len(P), bool)
    keep[1:] = np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-6
    P = P[keep]
    if len(P) < 2:
        return None, None

    T = np.empty_like(P)
    T[1:-1] = P[2:] - P[:-2]
    T[0], T[-1] = P[1] - P[0], P[-1] - P[-2]
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12

    # 平行传输参考向量，避免管道在转弯处扭转
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(T[0] @ ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    N = np.empty_like(P)
    n = ref - (ref @ T[0]) * T[0]
    N[0] = n / np.linalg.norm(n)
    for i in range(1, len(P)):
        n = N[i - 1] - (N[i - 1] @ T[i]) * T[i]
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            n = ref - (ref @ T[i]) * T[i]
            norm = np.linalg.norm(n)
        N[i] = n / norm
    B = np.cross(T, N)

    ang = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ring = (np.cos(ang)[None, :, None] * N[:, None, :]
            + np.sin(ang)[None, :, None] * B[:, None, :])
    V = (P[:, None, :] + radius * ring).reshape(-1, 3)

    idx = np.arange(len(P) * sides).reshape(len(P), sides)
    nxt = np.roll(idx, -1, axis=1)
    a, b = idx[:-1].ravel(), nxt[:-1].ravel()
    c, d = nxt[1:].ravel(), idx[1:].ravel()
    F = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    return V, F


# ─── 着色与 PLY 输出 ────────────────────────────────────────────────────────

_VIRIDIS = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140],
                     [94, 201, 98], [253, 231, 37]], float)


def height_colors(z, lo=None, hi=None):
    """按高度上 viridis。用 2/98 分位而非 min/max，免得几个离群面吃掉整个色阶。"""
    if lo is None or hi is None:
        lo, hi = np.percentile(z, [2.0, 98.0])
    t = np.clip((z - lo) / max(hi - lo, 1e-9), 0.0, 1.0) * (len(_VIRIDIS) - 1)
    i = np.clip(t.astype(int), 0, len(_VIRIDIS) - 2)
    f = (t - i)[:, None]
    return (_VIRIDIS[i] * (1.0 - f) + _VIRIDIS[i + 1] * f).astype(np.uint8)


def flat_colors(n, rgb):
    return np.tile(np.asarray(rgb, np.uint8), (n, 1))


def write_ply(path, verts, colors, faces=None):
    n_v = len(verts)
    n_f = 0 if faces is None else len(faces)
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n_v}",
              "property float x", "property float y", "property float z",
              "property uchar red", "property uchar green", "property uchar blue"]
    if faces is not None:
        header += [f"element face {n_f}", "property list uchar int vertex_indices"]
    header.append("end_header")

    with open(path, "wb") as fp:
        fp.write(("\n".join(header) + "\n").encode())
        rec = np.empty(n_v, dtype=[("p", "<f4", 3), ("c", "u1", 3)])
        rec["p"], rec["c"] = verts, colors
        fp.write(rec.tobytes())
        if faces is not None:
            rec = np.empty(n_f, dtype=[("n", "u1"), ("i", "<i4", 3)])
            rec["n"], rec["i"] = 3, faces
            fp.write(rec.tobytes())

    size_mb = os.path.getsize(path) / 1e6
    print(f"  -> {path}  ({n_v} 顶点, {n_f} 面, {size_mb:.1f} MB)")


# ─── 主流程 ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="从 SLAMesh 控制点重建全局 B-spline 地图",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--build-dir", required=True, help="C++ 的 kBsplineBuildDir")
    ap.add_argument("--out-dir", default=None, help="输出目录，默认同 build-dir")
    ap.add_argument("--res", type=int, default=10, help="每片 u/v 方向采样数")
    ap.add_argument("--color", choices=["height", "type"], default="height",
                    help="height=按高度上色；type=障碍/地面分色，便于核对分割")
    ap.add_argument("--traj-radius", type=float, default=0.3, help="轨迹管道半径 (m)")
    ap.add_argument("--no-obs", action="store_true", help="跳过障碍面")
    ap.add_argument("--no-gnd", action="store_true", help="跳过地面")
    ap.add_argument("--no-traj", action="store_true", help="跳过轨迹")
    ap.add_argument("--no-cloud", action="store_true", help="不输出采样点云")
    ap.add_argument("--max-surfaces", type=int, default=0,
                    help="每类最多读多少张，>0 时用于快速预览")
    ap.add_argument("--workers", type=int, default=8,
                    help="读控制点的线程数，冷缓存下能显著缩短耗时")
    args = ap.parse_args()

    build = args.build_dir
    out_dir = args.out_dir or build
    if not os.path.isdir(build):
        sys.exit(f"目录不存在: {build}")
    os.makedirs(out_dir, exist_ok=True)
    if args.res < 2:
        sys.exit("--res 至少为 2")

    t0 = time.time()
    print(f"读控制点 ({build}) ...")
    obs = {} if args.no_obs else collect(
        os.path.join(build, "controls", "*.txt"), load_obs, "障碍",
        args.max_surfaces, args.workers)
    gnd = {} if args.no_gnd else collect(
        os.path.join(build, "gnd_controls", "*.txt"), load_gnd, "地面",
        args.max_surfaces, args.workers)
    if not obs and not gnd:
        sys.exit(f"{build} 下没读到任何控制点，确认 controls/ 与 gnd_controls/ 是否存在")

    print(f"求值曲面 (每片 {args.res}x{args.res}) ...")
    V_obs, F_obs = tessellate(obs, args.res)
    V_gnd, F_gnd = tessellate(gnd, args.res)
    print(f"  障碍 {len(V_obs)} 顶点 / {len(F_obs)} 面，"
          f"地面 {len(V_gnd)} 顶点 / {len(F_gnd)} 面")

    # 色阶用障碍+地面的整体高度统一标定，两者才可比
    V_map = np.concatenate([V_obs, V_gnd]) if len(V_gnd) else V_obs
    if args.color == "height":
        lo, hi = np.percentile(V_map[:, 2], [2.0, 98.0])
        C_obs = height_colors(V_obs[:, 2], lo, hi)
        C_gnd = height_colors(V_gnd[:, 2], lo, hi) if len(V_gnd) else np.zeros((0, 3), np.uint8)
    else:
        C_obs = flat_colors(len(V_obs), (230, 160, 60))    # 橙：障碍
        C_gnd = flat_colors(len(V_gnd), (90, 140, 190))    # 蓝：地面
    C_map = np.concatenate([C_obs, C_gnd]) if len(V_gnd) else C_obs

    parts_v = [V_obs, V_gnd]
    parts_c = [C_obs, C_gnd]
    parts_f = [F_obs, F_gnd + len(V_obs)]
    base = len(V_obs) + len(V_gnd)

    traj_path = os.path.join(build, "bspline_traj_xyz.txt")
    if not args.no_traj and os.path.exists(traj_path):
        P = np.loadtxt(traj_path, dtype=np.float64).reshape(-1, 3)
        V_tr, F_tr = traj_tube(P, args.traj_radius)
        if V_tr is None:
            print(f"  轨迹只有 {len(P)} 个有效点，跳过管道")
        else:
            print(f"  轨迹 {len(P)} 帧 -> 管道 {len(V_tr)} 顶点")
            C_tr = flat_colors(len(V_tr), (220, 30, 30))
            parts_v.append(V_tr)
            parts_c.append(C_tr)
            parts_f.append(F_tr + base)
            write_ply(os.path.join(out_dir, "traj.ply"), V_tr, C_tr, F_tr)
    elif not args.no_traj:
        print(f"  未找到 {traj_path}，跳过轨迹")

    print("写出 ...")
    write_ply(os.path.join(out_dir, "map_mesh.ply"),
              np.concatenate(parts_v), np.concatenate(parts_c), np.concatenate(parts_f))
    if not args.no_cloud:
        write_ply(os.path.join(out_dir, "map_cloud.ply"), V_map, C_map)

    print(f"完成，耗时 {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()

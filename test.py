import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_points(filename):
    """
    读取三维点文件
    每行格式:
    x y z
    """
    return np.loadtxt(filename)


# 读取数据
pts1 = load_points("bspline_traj_xyz.txt")
pts2 = load_points("slamesh_r00_traj_xyz.txt")

# 创建3D图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 第一条轨迹
ax.plot(
    pts1[:, 0],
    pts1[:, 1],
    pts1[:, 2],
    '-o',
    linewidth=2,
    markersize=4,
    label='Trajectory 1'
)

# 第二条轨迹
ax.plot(
    pts2[:, 0],
    pts2[:, 1],
    pts2[:, 2],
    '-o',
    linewidth=2,
    markersize=4,
    label='Trajectory 2'
)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.legend()
ax.set_title('3D Trajectories')

plt.tight_layout()
plt.show()

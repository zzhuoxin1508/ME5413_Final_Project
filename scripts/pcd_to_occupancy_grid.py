#!/usr/bin/env python3
"""
将 PCD 点云转为 2D 占据栅格，供 map_server 使用（AMCL + move_base 等 2D 导航）。
支持一体化处理：PCD 清理 + 2D 栅格化 + 栅格去噪。
默认同时输出清理后的 3D 点云（.pcd）和 2D 地图（.pgm + .yaml）。
用法: python3 pcd_to_occupancy_grid.py <input.pcd> [-o output_basename] [-r resolution]
依赖: pip install open3d numpy
"""
import argparse
import os
from collections import deque


def filter_point_cloud(
    pcd,
    max_range=30.0,
    z_min=None,
    z_max=None,
    outlier=False,
    outlier_k=20,
    outlier_std=2.0,
    voxel=None,
):
    import numpy as np
    import open3d as o3d

    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return pcd

    # 1) 距离过滤：去掉远于 max_range 的点（消除放射状直线）
    if max_range is not None and max_range > 0:
        dist = np.linalg.norm(pts, axis=1)
        mask = dist < max_range
        pts = pts[mask]
        pcd.points = o3d.utility.Vector3dVector(pts)

    # 2) Z 轴裁剪（优先按楼层范围保留）
    if z_min is not None or z_max is not None:
        pts = np.asarray(pcd.points)
        if len(pts) > 0:
            z = pts[:, 2]
            mask = np.ones(len(pts), dtype=bool)
            if z_min is not None:
                mask &= z >= z_min
            if z_max is not None:
                mask &= z <= z_max
            pts = pts[mask]
            pcd.points = o3d.utility.Vector3dVector(pts)

    # 3) 离群点去除
    pts = np.asarray(pcd.points)
    if outlier and len(pts) > outlier_k:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=outlier_k,
            std_ratio=outlier_std,
        )

    # 4) 体素下采样（NDT 更稳、文件更小）
    pts = np.asarray(pcd.points)
    if voxel is not None and voxel > 0 and len(pts) > 0:
        pcd = pcd.voxel_down_sample(voxel)

    return pcd


def remove_small_obstacle_components(occ_mask, min_component_px):
    """
    删除过小障碍连通域，抑制孤立黑点噪声。
    occ_mask: True=障碍，False=可通行
    """
    import numpy as np

    if min_component_px <= 1:
        return occ_mask

    rows, cols = occ_mask.shape
    visited = np.zeros_like(occ_mask, dtype=bool)
    out = occ_mask.copy()
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for r in range(rows):
        for c in range(cols):
            if not occ_mask[r, c] or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            component = [(r, c)]
            while q:
                cr, cc = q.popleft()
                for dr, dc in neighbors:
                    nr, nc = cr + dr, cc + dc
                    if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                        continue
                    if visited[nr, nc] or not occ_mask[nr, nc]:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))
                    component.append((nr, nc))

            if len(component) < min_component_px:
                for pr, pc in component:
                    out[pr, pc] = False

    return out


def remove_isolated_obstacles(occ_mask, min_neighbors):
    """
    删除邻域中障碍像素过少的点，进一步去散点。
    """
    import numpy as np

    if min_neighbors <= 0:
        return occ_mask

    rows, cols = occ_mask.shape
    out = occ_mask.copy()
    for r in range(rows):
        r0 = max(0, r - 1)
        r1 = min(rows, r + 2)
        for c in range(cols):
            if not occ_mask[r, c]:
                continue
            c0 = max(0, c - 1)
            c1 = min(cols, c + 2)
            # 邻域计数包含自身，阈值建议 2~4
            n_occ = int(np.count_nonzero(occ_mask[r0:r1, c0:c1]))
            if n_occ < min_neighbors:
                out[r, c] = False
    return out

def main():
    parser = argparse.ArgumentParser(description="PCD -> 2D occupancy grid (.pgm + .yaml)")
    parser.add_argument("pcd", help="Input PCD file path")
    parser.add_argument("--output", default="my_map", help="Output base name (no extension)")
    parser.add_argument("--resolution", type=float, default=0.05, help="Grid resolution (m/pixel)")
    parser.add_argument("--z-min", type=float, default=None, help="Points below this Z are ignored")
    parser.add_argument("--z-max", type=float, default=None, help="Points above this Z are ignored")
    parser.add_argument("--max-range", type=float, default=35.0, help="保留距离原点小于此值的点（米），<=0 表示不做距离过滤")
    parser.add_argument("--outlier", action="store_true", help="做统计离群点去除（去孤立噪点）")
    parser.add_argument("--outlier-k", type=int, default=20, help="离群点邻域点数")
    parser.add_argument("--outlier-std", type=float, default=2.0, help="离群点标准差倍数")
    parser.add_argument("--voxel", type=float, default=None, help="体素下采样边长（米），如 0.1")
    parser.add_argument("--clean-pcd", default="", help="清理后 3D 点云路径（默认: <output>_clean.pcd）")
    parser.add_argument("--floor-z", type=float, default=None, help="可选：显式指定地面高度")
    parser.add_argument("--floor-percentile", type=float, default=5.0, help="自动估计地面时使用的分位数")
    parser.add_argument("--height-thresh", type=float, default=0.15, help="Min height above floor to count as obstacle (m)")
    parser.add_argument("--min-component-px", type=int, default=6, help="删除小于该像素数的障碍连通域")
    parser.add_argument("--min-neighbors", type=int, default=2, help="删除邻域障碍像素数小于该阈值的散点")
    args = parser.parse_args()

    try:
        import open3d as o3d
        import numpy as np
    except ImportError:
        print("请安装依赖: pip install open3d numpy")
        return 1

    if not os.path.isfile(args.pcd):
        print("文件不存在:", args.pcd)
        return 1

    # 读取并清理点云（统一与 filter_pcd.py 保持一致）
    pcd = o3d.io.read_point_cloud(args.pcd)
    pcd = filter_point_cloud(
        pcd,
        max_range=args.max_range,
        z_min=args.z_min,
        z_max=args.z_max,
        outlier=args.outlier,
        outlier_k=args.outlier_k,
        outlier_std=args.outlier_std,
        voxel=args.voxel,
    )

    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        print("清理后点云为空，请检查过滤参数")
        return 1

    # 统一输出路径：始终输出清理后的 3D 点云 + 2D 地图
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    base = args.output
    if base.endswith(".pgm") or base.endswith(".yaml") or base.endswith(".pcd"):
        base = os.path.splitext(base)[0]
    pgm_path = base + ".pgm"
    yaml_path = base + ".yaml"
    clean_pcd_path = args.clean_pcd if args.clean_pcd else (base + "_clean.pcd")
    clean_out_dir = os.path.dirname(clean_pcd_path)
    if clean_out_dir and not os.path.isdir(clean_out_dir):
        os.makedirs(clean_out_dir)
    o3d.io.write_point_cloud(clean_pcd_path, pcd)
    print("已保存清理后点云:", clean_pcd_path)

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_max = z.max()

    # 地面高度：取较低分位数，障碍物为高于地面 height_thresh 的点
    if args.floor_z is not None:
        floor_z = float(args.floor_z)
    else:
        floor_z = float(np.percentile(z, args.floor_percentile))
    obs_z_min = floor_z + args.height_thresh

    res = args.resolution
    cols = int(np.ceil((x_max - x_min) / res)) + 1
    rows = int(np.ceil((y_max - y_min) / res)) + 1

    # 栅格：0 = 障碍, 255 = 可通行 (map_server 约定)
    grid = np.full((rows, cols), 255, dtype=np.uint8)

    # 将点落入格子，高于 obs_z_min 的记为障碍
    xi = ((x - x_min) / res).astype(int)
    yi = ((y_max - y) / res).astype(int)  # 图像 y 从上到下
    xi = np.clip(xi, 0, cols - 1)
    yi = np.clip(yi, 0, rows - 1)
    mask = (z >= obs_z_min) & (z <= z_max)
    occ_mask = np.zeros((rows, cols), dtype=bool)
    occ_mask[yi[mask], xi[mask]] = True

    # 2D 去噪：删除小连通域 + 去孤立散点，让地图更清晰干净
    occ_mask = remove_small_obstacle_components(occ_mask, args.min_component_px)
    occ_mask = remove_isolated_obstacles(occ_mask, args.min_neighbors)
    grid[occ_mask] = 0

    # 写入 PGM（0=障碍，255=可通行）
    with open(pgm_path, "wb") as f:
        f.write(b"P5\n%d %d\n255\n" % (cols, rows))
        f.write(grid.tobytes())

    # 写入 YAML（map_server 格式）
    # origin 取地图左下角（最小 x, 最小 y）
    origin_x = x_min
    origin_y = y_min
    with open(yaml_path, "w") as f:
        f.write("image: %s\n" % os.path.basename(pgm_path))
        f.write("resolution: %.4f\n" % res)
        f.write("origin: [%.4f, %.4f, 0.0]\n" % (origin_x, origin_y))
        f.write("negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")

    print("已生成:", pgm_path, yaml_path)
    print("  resolution:", res, "  size:", cols, "x", rows)
    print("  floor_z:", round(float(floor_z), 4), " obs_z_min:", round(float(obs_z_min), 4))
    print("  occ_ratio:", round(float(np.count_nonzero(grid == 0) / grid.size), 6))
    return 0

if __name__ == "__main__":
    exit(main())

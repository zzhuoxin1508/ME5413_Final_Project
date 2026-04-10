import open3d as o3d
import numpy as np

def crop_pcd(input_path, output_path, min_bound, max_bound):
    """
    input_path: 原始 .pcd 文件路径
    output_path: 裁剪后的保存路径
    min_bound: 最小坐标范围 [x_min, y_min, z_min]
    max_bound: 最大坐标范围 [x_max, y_max, z_max]
    """
    # 1. 加载点云
    pcd = o3d.io.read_point_cloud(input_path)
    print(f"原始点云数量: {len(pcd.points)}")

    # 2. 定义裁剪范围
    # 注意：这里的坐标需要根据你点云的实际坐标系来设定
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=np.array(min_bound), 
        max_bound=np.array(max_bound)
    )

    # 3. 执行裁剪
    # crop 方法会返回 bbox 内部的所有点
    cropped_pcd = pcd.crop(bbox)
    
    # 4. 检查结果并保存
    if not cropped_pcd.is_empty():
        print(f"裁剪后点云数量: {len(cropped_pcd.points)}")
        o3d.io.write_point_cloud(output_path, cropped_pcd)
        # 可选：查看裁剪效果
        # o3d.visualization.draw_geometries([cropped_pcd])
    else:
        print("警告：裁剪范围内没有发现点，请检查坐标范围。")

# --- 使用示例 ---
# 假设你想保留min_coords = [xmin, ymin, zmin] max_coords = [xmax, ymax, zmax]
min_coords = [-10.0, -8.5, -1.0]
max_coords = [45.0, 50.0, 10.0]

crop_pcd("2795228000.pcd", "map_cropped2795228000.pcd", min_coords, max_coords)

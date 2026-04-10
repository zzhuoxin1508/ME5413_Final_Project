# 课程作业 + PCD 地图 + 本工作空间 整合说明

**两个工作空间：**
- 课程作业：`/home/yun/me5413/ME5413_Final_Project`
- NDT 项目：`/home/yun/catkin_ntd`

---

## 把两个工作空间“整理到一起”有没有必要？

**不一定必要，取决于你怎么用：**

| 情况 | 建议 |
|------|------|
| 课程作业**不需要**用 NDT（例如只用 AMCL + 2D 地图） | **不必**整合。两个工作空间各用各的即可。 |
| 课程作业**要用** NDT 做点云定位（例如用 PCD 地图替代/补充 AMCL） | **建议**整合，这样只在一个工作空间里编译、启动，交作业也方便。 |

**整合方式推荐：** 把 `ndt_localizer` 复制到课程作业工作空间里，这样“一个工作空间、一次 source、一套 launch”，为课程作业所用。下面按此方式写具体步骤。

---

## 如何整理到一起（让 NDT 为课程作业 ME5413_Final_Project 所用）

### 步骤 1：把 ndt_localizer 拷到课程作业工作空间

在终端执行：

```bash
cp -r /home/yun/catkin_ntd/src/ndt_localizer /home/yun/me5413/ME5413_Final_Project/src/
```

这样课程工作空间里会有：`ME5413_Final_Project/src/ndt_localizer/`。

### 步骤 2：把 PCD 地图放进课程工作空间

- 把你的 **.pcd** 地图放到：  
  **`/home/yun/me5413/ME5413_Final_Project/src/ndt_localizer/map/`**
- 若命名为 **court_yard_map.pcd**，则无需改 launch；否则编辑  
  `ME5413_Final_Project/src/ndt_localizer/launch/map_loader.launch`  
  第 29 行，把 `pcd_path` 改成你的文件名，例如：  
  `default="$(find ndt_localizer)/map/你的地图名.pcd"`

### 步骤 3：在课程工作空间里编译

```bash
cd /home/yun/me5413/ME5413_Final_Project
catkin_make
source devel/setup.bash
```

### 步骤 4：在课程作业里使用 NDT

以后**只在课程工作空间**操作即可：

- 启动你的世界/仿真（如 `world.launch` 等）。
- 需要 NDT 时再启动：
  ```bash
  roslaunch ndt_localizer ndt_localizer.launch
  ```

这样 NDT 和地图都在课程工作空间内，**catkin_ntd 可以不再打开**，所有东西都“整理到一起”为课程作业所用。

### 课程作业里的传感器话题（重要）

你的仿真里点云话题是 **`/mid/points`**（PointCloud2）。NDT 的体素滤波默认订阅的是 **`/os_cloud_node/points`**，需要改成你的话题。

在课程工作空间里编辑  
**`src/ndt_localizer/launch/points_downsample.launch`**，把第 4 行改为：

```xml
<arg name="points_topic" default="/mid/points" />
```

这样 NDT 才能收到 Jackal 的当前帧点云。

---

## 一、把 PCD 地图放进本工作空间（若只在 catkin_ntd 里用）

### 1. 放地图文件

把你的 **.pcd** 地图放到 NDT 包里的 `map` 目录：

```
catkin_ntd/
  src/ndt_localizer/
    map/
      court_yard_map.pcd   ← 把你这张 PCD 放这里，并命名为 court_yard_map.pcd
      或
      my_map.pcd          ← 或用你自己的名字（见下一步）
```

- **方式 A**：直接复制/移动你的 PCD 到  
  `catkin_ntd/src/ndt_localizer/map/`  
  并改名为 **court_yard_map.pcd**（这样不用改 launch，默认就会用这张图）。

- **方式 B**：保持你的文件名，例如 `my_map.pcd`，然后改 launch 里的路径（见下）。

### 2. 若 PCD 文件名不是 court_yard_map.pcd

编辑 **map_loader.launch**，改 `pcd_path` 的默认值：

文件：`src/ndt_localizer/launch/map_loader.launch`

```xml
<!-- 把这行的 court_yard_map.pcd 改成你的 PCD 文件名 -->
<arg name="pcd_path" default="$(find ndt_localizer)/map/你的地图名.pcd"/>
```

保存后，用下面的命令启动即可用你的地图。

---

## 二、只在本工作空间里跑（不混课程作业工作空间）

在 **catkin_ntd** 里编译并启动：

```bash
cd /home/yun/catkin_ntd
catkin_make
source devel/setup.bash
roslaunch ndt_localizer ndt_localizer.launch
```

这样会：

- 从 `map/` 里加载你放的 PCD（或你在 launch 里指定的路径）
- 发布点云地图、跑 NDT 定位、开 RViz 等

**总结**：PCD 放进 `ndt_localizer/map/`（或改 launch 指向你的路径），然后在本工作空间 `roslaunch ndt_localizer ndt_localizer.launch` 即可。

---

## 三、和“课程作业工作空间”放一起的两种做法

### 做法 1：课程作业工作空间只当“上层”，用本工作空间的地图与 NDT

- 课程作业工作空间：放你的作业代码、仿真、机器人描述等。
- 本工作空间（catkin_ntd）：只负责地图 + NDT。

步骤：

1. 把 PCD 放到 `catkin_ntd/src/ndt_localizer/map/`（或按上面改 launch 路径）。
2. 先 source 课程工作空间，再 source 本工作空间（让本工作空间的 ndt_localizer 生效）：
   ```bash
   source /path/to/你的课程工作空间/devel/setup.bash
   source /home/yun/catkin_ntd/devel/setup.bash
   ```
3. 在课程工作空间里启动你的机器人/仿真。
4. 再开一个终端，只 source 本工作空间后启动 NDT 与地图：
   ```bash
   source /home/yun/catkin_ntd/devel/setup.bash
   roslaunch ndt_localizer ndt_localizer.launch
   ```

这样：**课程作业**和**这张 PCD 地图 + 本工作空间**就“放一起”用了——话题、TF 对上即可（见下文“注意”）。

### 做法 2：把 ndt_localizer 和地图都拷到课程作业工作空间

1. 复制整个包与地图：
   ```bash
   cp -r /home/yun/catkin_ntd/src/ndt_localizer /path/to/你的课程工作空间/src/
   ```
2. 把你的 PCD 放到课程工作空间里的  
   `src/ndt_localizer/map/`  
   （若不用默认名，同样要改 `map_loader.launch` 里的 `pcd_path`）。
3. 在课程工作空间里编译、启动：
   ```bash
   cd /path/to/你的课程工作空间
   catkin_make
   source devel/setup.bash
   roslaunch ndt_localizer ndt_localizer.launch
   ```

这样所有东西都在“课程作业工作空间”里，本工作空间可以不再单独打开。

---

## 四、注意（话题与 TF）

- NDT 需要：
  - **地图点云**：`/points_map`（由 map_loader 发布）
  - **当前帧点云**：`/filtered_points`（由 points_downsample 发布，一般来自你的激光/雷达话题）
- 若你的课程作业里激光话题不叫 `points_raw`，需要在 launch 里做 remap，或改  
  `launch/points_downsample.launch` 里订阅的话题，使其订阅你的激光话题。
- 若用仿真时间：本仓库的 `ndt_localizer.launch` 里已有 `<param name="/use_sim_time" value="true" />`，和 Gazebo 等一起用时要保证仿真时间一致。

---

## 五、快速检查清单

- [ ] PCD 已放到 `catkin_ntd/src/ndt_localizer/map/`（或你改过的路径）
- [ ] 若文件名不是 `court_yard_map.pcd`，已改 `map_loader.launch` 里的 `pcd_path`
- [ ] 本工作空间已 `catkin_make` 且 `source devel/setup.bash`
- [ ] 若和课程工作空间一起用：source 顺序、话题名、TF 已对上

按上面做完，**课程作业工作空间、一张 PCD 地图、本工作空间**就都“放一起”了；若你告诉我课程工作空间路径和激光话题名，我可以按你的实际路径和话题写一份改好的 launch 片段。

---

## 六、用 EVO 评估定位/建图性能

课程要求可用 [EVO](https://github.com/MichaelGrupp/evo) 定量评估 SLAM/定位效果。本工作空间里：

- **真值轨迹**：`/gazebo/ground_truth/state`（`nav_msgs/Odometry`，Gazebo 仿真真值）
- **估计轨迹**：`/ndt_pose`（NDT 输出的 `nav_msgs/Odometry`）

### 1. 安装 EVO

```bash
pip install evo --upgrade
```

### 2. 录包（仿真 + NDT 运行时）

**终端 1**：启动仿真与 NDT（保证已 `source devel/setup.bash`）

```bash
roslaunch me5413_world world.launch
# 再开一个终端：
roslaunch ndt_localizer ndt_localizer.launch
```

**终端 2**：在 RViz 里给好 2D Pose Estimate 后，开始录包（用仿真时间）

```bash
source /home/yun/me5413/ME5413_Final_Project/devel/setup.bash
rosbag record -O evo_eval.bag /gazebo/ground_truth/state /ndt_pose /clock
```

开车/走一段轨迹后 Ctrl+C 结束录包，得到 `evo_eval.bag`。

### 3. 用 EVO 算轨迹误差

在录包所在目录下执行（ATE：绝对轨迹误差，仅平移；`--align` 做 SE3 对齐）：

```bash
evo_ape bag evo_eval.bag /gazebo/ground_truth/state /ndt_pose --pose_relation trans_part -va --align
```

只看平移误差时用 `trans_part`；若要看旋转可改用 `angle_deg` 或 `angle_rad`。  
RPE（相对位姿误差，看局部漂移）：

```bash
evo_rpe bag evo_eval.bag /gazebo/ground_truth/state /ndt_pose --pose_relation trans_part -va --align
```

画轨迹对比图：

```bash
evo_traj bag evo_eval.bag /gazebo/ground_truth/state /ndt_pose -p
```

脚本用法见项目根目录下 `scripts/evo_eval.sh`（可选）。

---

## 七、用 2D 栅格地图做导航（PCD → grid map）

课程里 2D 导航（`navigation.launch`：AMCL + move_base）用的是 **map_server**，需要 **2D 占据栅格**：一张图（`.pgm`）+ 描述文件（`.yaml`），不是 3D 的 PCD。所以若你手上只有 PCD，要做一步 **PCD → 2D grid map**。

### 两种做法

| 做法 | 说明 |
|------|------|
| **做法 A：PCD 转 2D 栅格** | 把现有 PCD 俯视投影成栅格，得到 `.pgm` + `.yaml`，供 map_server 用。 |
| **做法 B：用 Gmapping 重新建 2D 图** | 不依赖 PCD：启动 `world.launch` + `mapping.launch`，键盘开车跑一圈，用 `map_saver` 保存 2D 图。 |

### 做法 A：PCD → 2D 栅格

1. 用项目里的脚本（需 Python3 + open3d）：
   ```bash
   pip install open3d numpy  # 若未装
   python3 scripts/pcd_to_occupancy_grid.py src/ndt_localizer/map/scans.pcd -o src/me5413_world/maps/my_map -r 0.05
   ```
   会生成 `my_map.pgm` 和 `my_map.yaml`。  
2. 把生成的文件放到 `me5413_world/maps/`，`navigation.launch` 里已默认使用 `$(find me5413_world)/maps/my_map.yaml`，无需改 launch。  
3. 若 PCD 坐标系或朝向与仿真不一致，可改脚本里的 `origin` 或对图像做旋转/平移后再用。

### 做法 B：Gmapping 建 2D 图（不用 PCD）

1. 启动仿真与 Gmapping：
   ```bash
   roslaunch me5413_world world.launch
   # 另一终端：
   roslaunch me5413_world mapping.launch
   ```
2. 键盘控制机器人跑遍可通行区域。
3. 保存地图（再开一终端）：
   ```bash
   mkdir -p src/me5413_world/maps
   rosrun map_server map_saver -f src/me5413_world/maps/my_map
   ```
4. 用 2D 导航时启动：
   ```bash
   roslaunch me5413_world navigation.launch
   ```

**小结**：要用 2D 地图导航，就需要 2D 栅格（grid map）。要么把现有 PCD 转成栅格（PCD → grid），要么用 Gmapping 直接建 2D 图；两种都能得到 `my_map.pgm` + `my_map.yaml` 供 map_server 使用。

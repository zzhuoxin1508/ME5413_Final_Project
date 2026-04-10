/* goal_publisher_node.cpp

 * Copyright (C) 2023 SS47816

 * ROS Node for publishing goal poses

 deprecated for 2526
 
**/

#include "me5413_world/goal_publisher_node.hpp"

namespace me5413_world 
{

GoalPublisherNode::GoalPublisherNode() : tf2_listener_(tf2_buffer_)
{
  this->pub_goal_ = nh_.advertise<geometry_msgs::PoseStamped>("/move_base_simple/goal", 1);
  this->pub_absolute_position_error_ = nh_.advertise<std_msgs::Float32>("/me5413_world/absolute/position_error", 1);
  this->pub_absolute_heading_error_ = nh_.advertise<std_msgs::Float32>("/me5413_world/absolute/heading_error", 1);
  this->pub_relative_position_error_ = nh_.advertise<std_msgs::Float32>("/me5413_world/relative/position_error", 1);
  this->pub_relative_heading_error_ = nh_.advertise<std_msgs::Float32>("/me5413_world/relative/heading_error", 1);

  this->timer_ = nh_.createTimer(ros::Duration(0.2), &GoalPublisherNode::timerCallback, this);
  this->sub_robot_odom_ = nh_.subscribe("/gazebo/ground_truth/state", 1, &GoalPublisherNode::robotOdomCallback, this);
  this->sub_goal_name_ = nh_.subscribe("/rviz_panel/goal_name", 1, &GoalPublisherNode::goalNameCallback, this);
  this->sub_goal_pose_ = nh_.subscribe("/move_base_simple/goal", 1, &GoalPublisherNode::goalPoseCallback, this);
  this->sub_box_markers_ = nh_.subscribe("/gazebo/ground_truth/box_markers", 1, &GoalPublisherNode::boxMarkersCallback, this);
  
  // Initialization
  this->robot_frame_ = "base_link"; // 机器人自生坐标系，控制用
  this->map_frame_ = "map"; // 与world不完全对齐，会随着定位修正。ndt_pose / amcl_pose都是map坐标系，所有导航用这个
  this->world_frame_ = "world"; // Gazebo 仿真世界原点，不岁机器人和校正变化。/gazebo/ground_truth/state是world坐标系
  this->absolute_position_error_.data = 0.0; // 初始化机器人和目标位置姿态的误差
  this->absolute_heading_error_.data = 0.0;
  this->relative_position_error_.data = 0.0;
  this->relative_heading_error_.data = 0.0;
};

//定时器回调，0.2s执行一次timerCallback
void GoalPublisherNode::timerCallback(const ros::TimerEvent&)
{ 
  // 每次定时都把机器人 world 位姿同步到 map，避免 relative 误差一直使用旧值/未初始化值
  try
  {
    geometry_msgs::TransformStamped transform_map_world =
      this->tf2_buffer_.lookupTransform(this->map_frame_, this->world_frame_, ros::Time(0));
    tf2::doTransform(this->pose_world_robot_, this->pose_map_robot_, transform_map_world);
  }
  catch (tf2::TransformException& ex)
  {
    ROS_WARN_THROTTLE(2.0, "timerCallback map<-world transform failed: %s", ex.what());
  }

  // 计算绝对误差：world 系
  const std::pair<double, double> error_absolute = calculatePoseError(this->pose_world_robot_, this->pose_world_goal_);
  // 计算相对误差：map 系
  const std::pair<double, double> error_relative = calculatePoseError(this->pose_map_robot_, this->pose_map_goal_);
  
  //将误差写入消息
  this->absolute_position_error_.data = error_absolute.first;
  this->absolute_heading_error_.data = error_absolute.second;
  this->relative_position_error_.data = error_relative.first;
  this->relative_heading_error_.data = error_relative.second;

  if (this->goal_type_ == "box") //如果当前目标是box将航向误差设置为0，box 目标通常只在乎“走到箱子位置”，不在乎朝向
  {
    this->absolute_heading_error_.data = 0.0;
    this->relative_heading_error_.data = 0.0;
  }

  // 发布误差
  this->pub_absolute_position_error_.publish(this->absolute_position_error_);
  this->pub_absolute_heading_error_.publish(this->absolute_heading_error_);
  this->pub_relative_position_error_.publish(this->relative_position_error_);
  this->pub_relative_heading_error_.publish(this->relative_heading_error_);

  return;
};

//里程计回调，每次收到 /gazebo/ground_truth/state 时执行（这个在构造函数里面初始化的）
void GoalPublisherNode::robotOdomCallback(const nav_msgs::Odometry::ConstPtr& odom)
{
  this->world_frame_ = odom->header.frame_id; // 把 world frame 名字更新为 odom 消息里的 header.frame_id
  this->robot_frame_ = odom->child_frame_id;
  this->pose_world_robot_ = odom->pose.pose; // 把机器人在 world 坐标系下的 pose 存起来

  const tf2::Transform T_world_robot = convertPoseToTransform(this->pose_world_robot_); //T_world_robot 表示 robot 在 world 下的位姿变换
  const tf2::Transform T_robot_world = T_world_robot.inverse(); //T__robot_world 表示 world 在 robot 下的位姿变换

  geometry_msgs::TransformStamped transformStamped; // 创建一个 TransformStamped 消息，用来广播 TF
  transformStamped.header.stamp = ros::Time::now();
  transformStamped.header.frame_id = this->robot_frame_;
  transformStamped.child_frame_id = this->world_frame_;
  transformStamped.transform.translation.x = T_robot_world.getOrigin().getX(); // 设置平移旋转和四元数变换
  transformStamped.transform.translation.y = T_robot_world.getOrigin().getY();
  transformStamped.transform.translation.z = 0.0;
  transformStamped.transform.rotation.x = T_robot_world.getRotation().getX();
  transformStamped.transform.rotation.y = T_robot_world.getRotation().getY();
  transformStamped.transform.rotation.z = T_robot_world.getRotation().getZ();
  transformStamped.transform.rotation.w = T_robot_world.getRotation().getW();
  
  this->tf2_bcaster_.sendTransform(transformStamped); //广播这条 TF 变换

  return;
};

//目标名字回调（把“任务目标（名字）”变成“导航目标（map坐标）”）
void GoalPublisherNode::goalNameCallback(const std_msgs::String::ConstPtr& name)
{ 
  const std::string goal_name = name->data; //从 ROS 字符串消息中取出实际字符串
  const int end = goal_name.find_last_of("_");
  this->goal_type_ = goal_name.substr(1, end-1);
  const int goal_box_id = stoi(goal_name.substr(end+1, 1));

  geometry_msgs::PoseStamped P_world_goal;
  if (this->goal_type_ == "box")
  {
    if (box_poses_.empty())
    {
      ROS_ERROR_STREAM("Box poses unknown, please spawn boxes first!");
      return;
    }
    else if (goal_box_id >= box_poses_.size())
    {
      ROS_ERROR_STREAM("Box id is outside the available range, please select a smaller id!");
      return;
    }
    
    P_world_goal = box_poses_[goal_box_id - 1];
  }
  else
  {
    // Get the Pose of the goal in world frame
    P_world_goal = getGoalPoseFromConfig(goal_name);
  }

  this->pose_world_goal_ = P_world_goal.pose; //把 world 系下的目标 pose 保存到成员变量
  // Get the Transform from world to map from the tf_listener
  geometry_msgs::TransformStamped transform_map_world;
  try
  {
    transform_map_world = this->tf2_buffer_.lookupTransform(this->map_frame_, this->world_frame_, ros::Time(0));
  }
  catch (tf2::TransformException& ex)
  {
    ROS_WARN("%s", ex.what());
    return;
  }

  // Transform the goal pose to map frame
  geometry_msgs::PoseStamped P_map_goal;
  tf2::doTransform(P_world_goal, P_map_goal, transform_map_world);
  P_map_goal.header.stamp = ros::Time::now();
  P_map_goal.header.frame_id = map_frame_;

  // Transform the robot pose to map frame
  tf2::doTransform(this->pose_world_robot_, this->pose_map_robot_, transform_map_world);

  // Publish goal pose in map frame 
  if (this->goal_type_ != "box")
  {
    this->pub_goal_.publish(P_map_goal);
  }

  return;
};

//目标姿态回调
void GoalPublisherNode::goalPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& goal_pose)
{
  this->pose_map_goal_ = goal_pose->pose; //把收到的 goal pose 保存成 map 坐标系下的目标 pose

  // 关键修复：手动 RViz 设目标时，同时更新 world 系目标，否则 AHE 会长期使用默认目标导致 NaN
  geometry_msgs::PoseStamped goal_map = *goal_pose;
  if (goal_map.header.frame_id.empty())
  {
    goal_map.header.frame_id = this->map_frame_;
  }

  try
  {
    geometry_msgs::TransformStamped transform_world_goal =
      this->tf2_buffer_.lookupTransform(this->world_frame_, goal_map.header.frame_id, ros::Time(0));
    geometry_msgs::PoseStamped goal_world;
    tf2::doTransform(goal_map, goal_world, transform_world_goal);
    this->pose_world_goal_ = goal_world.pose;
  }
  catch (tf2::TransformException& ex)
  {
    ROS_WARN_THROTTLE(2.0, "goalPoseCallback world transform failed: %s", ex.what());
  }
}

tf2::Transform GoalPublisherNode::convertPoseToTransform(const geometry_msgs::Pose& pose)
{
  tf2::Transform T;
  T.setOrigin(tf2::Vector3(pose.position.x, pose.position.y, 0));
  tf2::Quaternion q;
  q.setValue(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w);
  T.setRotation(q);

  return T;
};

// 定义 box marker 数组的回调函数
void GoalPublisherNode::boxMarkersCallback(const visualization_msgs::MarkerArray::ConstPtr& box_markers)
{
  this->box_poses_.clear();
  for (const auto& box : box_markers->markers)
  {
    geometry_msgs::PoseStamped pose;
    pose.pose = box.pose;
    this->box_poses_.emplace_back(pose);
  }

  return;
};

//从参数服务器读取目标位姿（参数服务器 = 静态信息 topic = 动态信息）
geometry_msgs::PoseStamped GoalPublisherNode::getGoalPoseFromConfig(const std::string& name)
{
  /** 
   * Get the Transform from goal to world from the file
   */

  double x, y, yaw;
  nh_.getParam("/me5413_world" + name + "/x", x);
  nh_.getParam("/me5413_world" + name + "/y", y);
  nh_.getParam("/me5413_world" + name + "/yaw", yaw);
  nh_.getParam("/me5413_world/frame_id", this->world_frame_);

  tf2::Quaternion q;
  q.setRPY(0, 0, yaw);
  q.normalize();

  geometry_msgs::PoseStamped P_world_goal;
  P_world_goal.pose.position.x = x;
  P_world_goal.pose.position.y = y;
  P_world_goal.pose.orientation = tf2::toMsg(q);

  return P_world_goal;
};

// 定义一个函数，输入机器人 pose 和目标 pose，输出方向和航向误差
std::pair<double, double> GoalPublisherNode::calculatePoseError(const geometry_msgs::Pose& pose_robot, const geometry_msgs::Pose& pose_goal)
{
  // Positional Error（欧氏距离）
  const double position_error = std::sqrt(
    std::pow(pose_robot.position.x - pose_goal.position.x, 2) + 
    std::pow(pose_robot.position.y - pose_goal.position.y, 2)
  );

  // Heading Error
  tf2::Quaternion q_robot, q_goal;
  tf2::fromMsg(pose_robot.orientation, q_robot);
  tf2::fromMsg(pose_goal.orientation, q_goal);
  const tf2::Matrix3x3 m_robot = tf2::Matrix3x3(q_robot);
  const tf2::Matrix3x3 m_goal = tf2::Matrix3x3(q_goal);

  double roll, pitch, yaw_robot, yaw_goal;
  m_robot.getRPY(roll, pitch, yaw_robot);
  m_goal.getRPY(roll, pitch, yaw_goal);

  const double heading_error = (yaw_robot - yaw_goal)/M_PI*180.0; // 计算航向误差，并从弧度转成角度。

  return std::pair<double, double>(position_error, heading_error);
}

} // namespace me5413_world

int main(int argc, char** argv)
{
  ros::init(argc, argv, "goal_publisher_node"); //程序入口初始化ROS节点
  me5413_world::GoalPublisherNode goal_publisher_node; //创建GoalPublisherNode对象
  ros::spin();  // spin the ros node.
  return 0;
}

// 这个文件 = 把任务目标 → 转成导航目标；统一 world / map 坐标；实时计算导航误差
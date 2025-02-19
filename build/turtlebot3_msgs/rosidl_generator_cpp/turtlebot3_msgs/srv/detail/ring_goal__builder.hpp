// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from turtlebot3_msgs:srv/RingGoal.idl
// generated code does not contain a copyright notice

#ifndef TURTLEBOT3_MSGS__SRV__DETAIL__RING_GOAL__BUILDER_HPP_
#define TURTLEBOT3_MSGS__SRV__DETAIL__RING_GOAL__BUILDER_HPP_

#include "turtlebot3_msgs/srv/detail/ring_goal__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace turtlebot3_msgs
{

namespace srv
{

namespace builder
{

class Init_RingGoal_Request_radius
{
public:
  explicit Init_RingGoal_Request_radius(::turtlebot3_msgs::srv::RingGoal_Request & msg)
  : msg_(msg)
  {}
  ::turtlebot3_msgs::srv::RingGoal_Request radius(::turtlebot3_msgs::srv::RingGoal_Request::_radius_type arg)
  {
    msg_.radius = std::move(arg);
    return std::move(msg_);
  }

private:
  ::turtlebot3_msgs::srv::RingGoal_Request msg_;
};

class Init_RingGoal_Request_robot_pose_y
{
public:
  explicit Init_RingGoal_Request_robot_pose_y(::turtlebot3_msgs::srv::RingGoal_Request & msg)
  : msg_(msg)
  {}
  Init_RingGoal_Request_radius robot_pose_y(::turtlebot3_msgs::srv::RingGoal_Request::_robot_pose_y_type arg)
  {
    msg_.robot_pose_y = std::move(arg);
    return Init_RingGoal_Request_radius(msg_);
  }

private:
  ::turtlebot3_msgs::srv::RingGoal_Request msg_;
};

class Init_RingGoal_Request_robot_pose_x
{
public:
  Init_RingGoal_Request_robot_pose_x()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_RingGoal_Request_robot_pose_y robot_pose_x(::turtlebot3_msgs::srv::RingGoal_Request::_robot_pose_x_type arg)
  {
    msg_.robot_pose_x = std::move(arg);
    return Init_RingGoal_Request_robot_pose_y(msg_);
  }

private:
  ::turtlebot3_msgs::srv::RingGoal_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::turtlebot3_msgs::srv::RingGoal_Request>()
{
  return turtlebot3_msgs::srv::builder::Init_RingGoal_Request_robot_pose_x();
}

}  // namespace turtlebot3_msgs


namespace turtlebot3_msgs
{

namespace srv
{


}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::turtlebot3_msgs::srv::RingGoal_Response>()
{
  return ::turtlebot3_msgs::srv::RingGoal_Response(rosidl_runtime_cpp::MessageInitialization::ZERO);
}

}  // namespace turtlebot3_msgs

#endif  // TURTLEBOT3_MSGS__SRV__DETAIL__RING_GOAL__BUILDER_HPP_

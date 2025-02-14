// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from turtlebot3_msgs:srv/Goal.idl
// generated code does not contain a copyright notice

#ifndef TURTLEBOT3_MSGS__SRV__DETAIL__GOAL__BUILDER_HPP_
#define TURTLEBOT3_MSGS__SRV__DETAIL__GOAL__BUILDER_HPP_

#include "turtlebot3_msgs/srv/detail/goal__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace turtlebot3_msgs
{

namespace srv
{

namespace builder
{

class Init_Goal_Request_empty
{
public:
  Init_Goal_Request_empty()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::turtlebot3_msgs::srv::Goal_Request empty(::turtlebot3_msgs::srv::Goal_Request::_empty_type arg)
  {
    msg_.empty = std::move(arg);
    return std::move(msg_);
  }

private:
  ::turtlebot3_msgs::srv::Goal_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::turtlebot3_msgs::srv::Goal_Request>()
{
  return turtlebot3_msgs::srv::builder::Init_Goal_Request_empty();
}

}  // namespace turtlebot3_msgs


namespace turtlebot3_msgs
{

namespace srv
{

namespace builder
{

class Init_Goal_Response_new_goal
{
public:
  Init_Goal_Response_new_goal()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::turtlebot3_msgs::srv::Goal_Response new_goal(::turtlebot3_msgs::srv::Goal_Response::_new_goal_type arg)
  {
    msg_.new_goal = std::move(arg);
    return std::move(msg_);
  }

private:
  ::turtlebot3_msgs::srv::Goal_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::turtlebot3_msgs::srv::Goal_Response>()
{
  return turtlebot3_msgs::srv::builder::Init_Goal_Response_new_goal();
}

}  // namespace turtlebot3_msgs

#endif  // TURTLEBOT3_MSGS__SRV__DETAIL__GOAL__BUILDER_HPP_

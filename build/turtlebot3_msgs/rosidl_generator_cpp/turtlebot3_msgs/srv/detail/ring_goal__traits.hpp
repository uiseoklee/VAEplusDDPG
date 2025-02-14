// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from turtlebot3_msgs:srv/RingGoal.idl
// generated code does not contain a copyright notice

#ifndef TURTLEBOT3_MSGS__SRV__DETAIL__RING_GOAL__TRAITS_HPP_
#define TURTLEBOT3_MSGS__SRV__DETAIL__RING_GOAL__TRAITS_HPP_

#include "turtlebot3_msgs/srv/detail/ring_goal__struct.hpp"
#include <rosidl_runtime_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<turtlebot3_msgs::srv::RingGoal_Request>()
{
  return "turtlebot3_msgs::srv::RingGoal_Request";
}

template<>
inline const char * name<turtlebot3_msgs::srv::RingGoal_Request>()
{
  return "turtlebot3_msgs/srv/RingGoal_Request";
}

template<>
struct has_fixed_size<turtlebot3_msgs::srv::RingGoal_Request>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<turtlebot3_msgs::srv::RingGoal_Request>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<turtlebot3_msgs::srv::RingGoal_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<turtlebot3_msgs::srv::RingGoal_Response>()
{
  return "turtlebot3_msgs::srv::RingGoal_Response";
}

template<>
inline const char * name<turtlebot3_msgs::srv::RingGoal_Response>()
{
  return "turtlebot3_msgs/srv/RingGoal_Response";
}

template<>
struct has_fixed_size<turtlebot3_msgs::srv::RingGoal_Response>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<turtlebot3_msgs::srv::RingGoal_Response>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<turtlebot3_msgs::srv::RingGoal_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<turtlebot3_msgs::srv::RingGoal>()
{
  return "turtlebot3_msgs::srv::RingGoal";
}

template<>
inline const char * name<turtlebot3_msgs::srv::RingGoal>()
{
  return "turtlebot3_msgs/srv/RingGoal";
}

template<>
struct has_fixed_size<turtlebot3_msgs::srv::RingGoal>
  : std::integral_constant<
    bool,
    has_fixed_size<turtlebot3_msgs::srv::RingGoal_Request>::value &&
    has_fixed_size<turtlebot3_msgs::srv::RingGoal_Response>::value
  >
{
};

template<>
struct has_bounded_size<turtlebot3_msgs::srv::RingGoal>
  : std::integral_constant<
    bool,
    has_bounded_size<turtlebot3_msgs::srv::RingGoal_Request>::value &&
    has_bounded_size<turtlebot3_msgs::srv::RingGoal_Response>::value
  >
{
};

template<>
struct is_service<turtlebot3_msgs::srv::RingGoal>
  : std::true_type
{
};

template<>
struct is_service_request<turtlebot3_msgs::srv::RingGoal_Request>
  : std::true_type
{
};

template<>
struct is_service_response<turtlebot3_msgs::srv::RingGoal_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // TURTLEBOT3_MSGS__SRV__DETAIL__RING_GOAL__TRAITS_HPP_

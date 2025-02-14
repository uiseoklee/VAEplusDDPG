// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from turtlebot3_msgs:srv/Goal.idl
// generated code does not contain a copyright notice

#ifndef TURTLEBOT3_MSGS__SRV__DETAIL__GOAL__TRAITS_HPP_
#define TURTLEBOT3_MSGS__SRV__DETAIL__GOAL__TRAITS_HPP_

#include "turtlebot3_msgs/srv/detail/goal__struct.hpp"
#include <rosidl_runtime_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<turtlebot3_msgs::srv::Goal_Request>()
{
  return "turtlebot3_msgs::srv::Goal_Request";
}

template<>
inline const char * name<turtlebot3_msgs::srv::Goal_Request>()
{
  return "turtlebot3_msgs/srv/Goal_Request";
}

template<>
struct has_fixed_size<turtlebot3_msgs::srv::Goal_Request>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<turtlebot3_msgs::srv::Goal_Request>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<turtlebot3_msgs::srv::Goal_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<turtlebot3_msgs::srv::Goal_Response>()
{
  return "turtlebot3_msgs::srv::Goal_Response";
}

template<>
inline const char * name<turtlebot3_msgs::srv::Goal_Response>()
{
  return "turtlebot3_msgs/srv/Goal_Response";
}

template<>
struct has_fixed_size<turtlebot3_msgs::srv::Goal_Response>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<turtlebot3_msgs::srv::Goal_Response>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<turtlebot3_msgs::srv::Goal_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<turtlebot3_msgs::srv::Goal>()
{
  return "turtlebot3_msgs::srv::Goal";
}

template<>
inline const char * name<turtlebot3_msgs::srv::Goal>()
{
  return "turtlebot3_msgs/srv/Goal";
}

template<>
struct has_fixed_size<turtlebot3_msgs::srv::Goal>
  : std::integral_constant<
    bool,
    has_fixed_size<turtlebot3_msgs::srv::Goal_Request>::value &&
    has_fixed_size<turtlebot3_msgs::srv::Goal_Response>::value
  >
{
};

template<>
struct has_bounded_size<turtlebot3_msgs::srv::Goal>
  : std::integral_constant<
    bool,
    has_bounded_size<turtlebot3_msgs::srv::Goal_Request>::value &&
    has_bounded_size<turtlebot3_msgs::srv::Goal_Response>::value
  >
{
};

template<>
struct is_service<turtlebot3_msgs::srv::Goal>
  : std::true_type
{
};

template<>
struct is_service_request<turtlebot3_msgs::srv::Goal_Request>
  : std::true_type
{
};

template<>
struct is_service_response<turtlebot3_msgs::srv::Goal_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // TURTLEBOT3_MSGS__SRV__DETAIL__GOAL__TRAITS_HPP_

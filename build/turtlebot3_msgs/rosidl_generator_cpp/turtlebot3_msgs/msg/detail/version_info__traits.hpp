// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from turtlebot3_msgs:msg/VersionInfo.idl
// generated code does not contain a copyright notice

#ifndef TURTLEBOT3_MSGS__MSG__DETAIL__VERSION_INFO__TRAITS_HPP_
#define TURTLEBOT3_MSGS__MSG__DETAIL__VERSION_INFO__TRAITS_HPP_

#include "turtlebot3_msgs/msg/detail/version_info__struct.hpp"
#include <rosidl_runtime_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<turtlebot3_msgs::msg::VersionInfo>()
{
  return "turtlebot3_msgs::msg::VersionInfo";
}

template<>
inline const char * name<turtlebot3_msgs::msg::VersionInfo>()
{
  return "turtlebot3_msgs/msg/VersionInfo";
}

template<>
struct has_fixed_size<turtlebot3_msgs::msg::VersionInfo>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<turtlebot3_msgs::msg::VersionInfo>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<turtlebot3_msgs::msg::VersionInfo>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // TURTLEBOT3_MSGS__MSG__DETAIL__VERSION_INFO__TRAITS_HPP_

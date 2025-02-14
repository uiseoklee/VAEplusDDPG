// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from turtlebot3_msgs:msg/VersionInfo.idl
// generated code does not contain a copyright notice

#ifndef TURTLEBOT3_MSGS__MSG__DETAIL__VERSION_INFO__STRUCT_H_
#define TURTLEBOT3_MSGS__MSG__DETAIL__VERSION_INFO__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'hardware'
// Member 'firmware'
// Member 'software'
#include "rosidl_runtime_c/string.h"

// Struct defined in msg/VersionInfo in the package turtlebot3_msgs.
typedef struct turtlebot3_msgs__msg__VersionInfo
{
  rosidl_runtime_c__String hardware;
  rosidl_runtime_c__String firmware;
  rosidl_runtime_c__String software;
} turtlebot3_msgs__msg__VersionInfo;

// Struct for a sequence of turtlebot3_msgs__msg__VersionInfo.
typedef struct turtlebot3_msgs__msg__VersionInfo__Sequence
{
  turtlebot3_msgs__msg__VersionInfo * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} turtlebot3_msgs__msg__VersionInfo__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TURTLEBOT3_MSGS__MSG__DETAIL__VERSION_INFO__STRUCT_H_

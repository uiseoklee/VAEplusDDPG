// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from turtlebot3_msgs:srv/Goal.idl
// generated code does not contain a copyright notice

#ifndef TURTLEBOT3_MSGS__SRV__DETAIL__GOAL__STRUCT_H_
#define TURTLEBOT3_MSGS__SRV__DETAIL__GOAL__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Struct defined in srv/Goal in the package turtlebot3_msgs.
typedef struct turtlebot3_msgs__srv__Goal_Request
{
  bool empty;
} turtlebot3_msgs__srv__Goal_Request;

// Struct for a sequence of turtlebot3_msgs__srv__Goal_Request.
typedef struct turtlebot3_msgs__srv__Goal_Request__Sequence
{
  turtlebot3_msgs__srv__Goal_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} turtlebot3_msgs__srv__Goal_Request__Sequence;


// Constants defined in the message

// Struct defined in srv/Goal in the package turtlebot3_msgs.
typedef struct turtlebot3_msgs__srv__Goal_Response
{
  bool new_goal;
} turtlebot3_msgs__srv__Goal_Response;

// Struct for a sequence of turtlebot3_msgs__srv__Goal_Response.
typedef struct turtlebot3_msgs__srv__Goal_Response__Sequence
{
  turtlebot3_msgs__srv__Goal_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} turtlebot3_msgs__srv__Goal_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TURTLEBOT3_MSGS__SRV__DETAIL__GOAL__STRUCT_H_

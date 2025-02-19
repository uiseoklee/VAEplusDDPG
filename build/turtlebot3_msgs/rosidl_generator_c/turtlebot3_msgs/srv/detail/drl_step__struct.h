// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from turtlebot3_msgs:srv/DrlStep.idl
// generated code does not contain a copyright notice

#ifndef TURTLEBOT3_MSGS__SRV__DETAIL__DRL_STEP__STRUCT_H_
#define TURTLEBOT3_MSGS__SRV__DETAIL__DRL_STEP__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'action'
// Member 'previous_action'
#include "rosidl_runtime_c/primitives_sequence.h"

// Struct defined in srv/DrlStep in the package turtlebot3_msgs.
typedef struct turtlebot3_msgs__srv__DrlStep_Request
{
  rosidl_runtime_c__float__Sequence action;
  rosidl_runtime_c__float__Sequence previous_action;
} turtlebot3_msgs__srv__DrlStep_Request;

// Struct for a sequence of turtlebot3_msgs__srv__DrlStep_Request.
typedef struct turtlebot3_msgs__srv__DrlStep_Request__Sequence
{
  turtlebot3_msgs__srv__DrlStep_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} turtlebot3_msgs__srv__DrlStep_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'state'
// already included above
// #include "rosidl_runtime_c/primitives_sequence.h"

// Struct defined in srv/DrlStep in the package turtlebot3_msgs.
typedef struct turtlebot3_msgs__srv__DrlStep_Response
{
  rosidl_runtime_c__float__Sequence state;
  float reward;
  bool done;
  uint32_t success;
  float distance_traveled;
} turtlebot3_msgs__srv__DrlStep_Response;

// Struct for a sequence of turtlebot3_msgs__srv__DrlStep_Response.
typedef struct turtlebot3_msgs__srv__DrlStep_Response__Sequence
{
  turtlebot3_msgs__srv__DrlStep_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} turtlebot3_msgs__srv__DrlStep_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TURTLEBOT3_MSGS__SRV__DETAIL__DRL_STEP__STRUCT_H_

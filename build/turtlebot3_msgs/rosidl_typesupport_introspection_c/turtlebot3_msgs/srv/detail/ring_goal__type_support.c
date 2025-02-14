// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from turtlebot3_msgs:srv/RingGoal.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "turtlebot3_msgs/srv/detail/ring_goal__rosidl_typesupport_introspection_c.h"
#include "turtlebot3_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "turtlebot3_msgs/srv/detail/ring_goal__functions.h"
#include "turtlebot3_msgs/srv/detail/ring_goal__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void RingGoal_Request__rosidl_typesupport_introspection_c__RingGoal_Request_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  turtlebot3_msgs__srv__RingGoal_Request__init(message_memory);
}

void RingGoal_Request__rosidl_typesupport_introspection_c__RingGoal_Request_fini_function(void * message_memory)
{
  turtlebot3_msgs__srv__RingGoal_Request__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember RingGoal_Request__rosidl_typesupport_introspection_c__RingGoal_Request_message_member_array[3] = {
  {
    "robot_pose_x",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(turtlebot3_msgs__srv__RingGoal_Request, robot_pose_x),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "robot_pose_y",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(turtlebot3_msgs__srv__RingGoal_Request, robot_pose_y),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "radius",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(turtlebot3_msgs__srv__RingGoal_Request, radius),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers RingGoal_Request__rosidl_typesupport_introspection_c__RingGoal_Request_message_members = {
  "turtlebot3_msgs__srv",  // message namespace
  "RingGoal_Request",  // message name
  3,  // number of fields
  sizeof(turtlebot3_msgs__srv__RingGoal_Request),
  RingGoal_Request__rosidl_typesupport_introspection_c__RingGoal_Request_message_member_array,  // message members
  RingGoal_Request__rosidl_typesupport_introspection_c__RingGoal_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  RingGoal_Request__rosidl_typesupport_introspection_c__RingGoal_Request_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t RingGoal_Request__rosidl_typesupport_introspection_c__RingGoal_Request_message_type_support_handle = {
  0,
  &RingGoal_Request__rosidl_typesupport_introspection_c__RingGoal_Request_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_turtlebot3_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, turtlebot3_msgs, srv, RingGoal_Request)() {
  if (!RingGoal_Request__rosidl_typesupport_introspection_c__RingGoal_Request_message_type_support_handle.typesupport_identifier) {
    RingGoal_Request__rosidl_typesupport_introspection_c__RingGoal_Request_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &RingGoal_Request__rosidl_typesupport_introspection_c__RingGoal_Request_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "turtlebot3_msgs/srv/detail/ring_goal__rosidl_typesupport_introspection_c.h"
// already included above
// #include "turtlebot3_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "turtlebot3_msgs/srv/detail/ring_goal__functions.h"
// already included above
// #include "turtlebot3_msgs/srv/detail/ring_goal__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void RingGoal_Response__rosidl_typesupport_introspection_c__RingGoal_Response_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  turtlebot3_msgs__srv__RingGoal_Response__init(message_memory);
}

void RingGoal_Response__rosidl_typesupport_introspection_c__RingGoal_Response_fini_function(void * message_memory)
{
  turtlebot3_msgs__srv__RingGoal_Response__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember RingGoal_Response__rosidl_typesupport_introspection_c__RingGoal_Response_message_member_array[1] = {
  {
    "structure_needs_at_least_one_member",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(turtlebot3_msgs__srv__RingGoal_Response, structure_needs_at_least_one_member),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers RingGoal_Response__rosidl_typesupport_introspection_c__RingGoal_Response_message_members = {
  "turtlebot3_msgs__srv",  // message namespace
  "RingGoal_Response",  // message name
  1,  // number of fields
  sizeof(turtlebot3_msgs__srv__RingGoal_Response),
  RingGoal_Response__rosidl_typesupport_introspection_c__RingGoal_Response_message_member_array,  // message members
  RingGoal_Response__rosidl_typesupport_introspection_c__RingGoal_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  RingGoal_Response__rosidl_typesupport_introspection_c__RingGoal_Response_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t RingGoal_Response__rosidl_typesupport_introspection_c__RingGoal_Response_message_type_support_handle = {
  0,
  &RingGoal_Response__rosidl_typesupport_introspection_c__RingGoal_Response_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_turtlebot3_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, turtlebot3_msgs, srv, RingGoal_Response)() {
  if (!RingGoal_Response__rosidl_typesupport_introspection_c__RingGoal_Response_message_type_support_handle.typesupport_identifier) {
    RingGoal_Response__rosidl_typesupport_introspection_c__RingGoal_Response_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &RingGoal_Response__rosidl_typesupport_introspection_c__RingGoal_Response_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "turtlebot3_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "turtlebot3_msgs/srv/detail/ring_goal__rosidl_typesupport_introspection_c.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/service_introspection.h"

// this is intentionally not const to allow initialization later to prevent an initialization race
static rosidl_typesupport_introspection_c__ServiceMembers turtlebot3_msgs__srv__detail__ring_goal__rosidl_typesupport_introspection_c__RingGoal_service_members = {
  "turtlebot3_msgs__srv",  // service namespace
  "RingGoal",  // service name
  // these two fields are initialized below on the first access
  NULL,  // request message
  // turtlebot3_msgs__srv__detail__ring_goal__rosidl_typesupport_introspection_c__RingGoal_Request_message_type_support_handle,
  NULL  // response message
  // turtlebot3_msgs__srv__detail__ring_goal__rosidl_typesupport_introspection_c__RingGoal_Response_message_type_support_handle
};

static rosidl_service_type_support_t turtlebot3_msgs__srv__detail__ring_goal__rosidl_typesupport_introspection_c__RingGoal_service_type_support_handle = {
  0,
  &turtlebot3_msgs__srv__detail__ring_goal__rosidl_typesupport_introspection_c__RingGoal_service_members,
  get_service_typesupport_handle_function,
};

// Forward declaration of request/response type support functions
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, turtlebot3_msgs, srv, RingGoal_Request)();

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, turtlebot3_msgs, srv, RingGoal_Response)();

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_turtlebot3_msgs
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, turtlebot3_msgs, srv, RingGoal)() {
  if (!turtlebot3_msgs__srv__detail__ring_goal__rosidl_typesupport_introspection_c__RingGoal_service_type_support_handle.typesupport_identifier) {
    turtlebot3_msgs__srv__detail__ring_goal__rosidl_typesupport_introspection_c__RingGoal_service_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  rosidl_typesupport_introspection_c__ServiceMembers * service_members =
    (rosidl_typesupport_introspection_c__ServiceMembers *)turtlebot3_msgs__srv__detail__ring_goal__rosidl_typesupport_introspection_c__RingGoal_service_type_support_handle.data;

  if (!service_members->request_members_) {
    service_members->request_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, turtlebot3_msgs, srv, RingGoal_Request)()->data;
  }
  if (!service_members->response_members_) {
    service_members->response_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, turtlebot3_msgs, srv, RingGoal_Response)()->data;
  }

  return &turtlebot3_msgs__srv__detail__ring_goal__rosidl_typesupport_introspection_c__RingGoal_service_type_support_handle;
}

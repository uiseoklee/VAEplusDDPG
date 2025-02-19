// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from turtlebot3_msgs:srv/DrlStep.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "turtlebot3_msgs/srv/detail/drl_step__rosidl_typesupport_introspection_c.h"
#include "turtlebot3_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "turtlebot3_msgs/srv/detail/drl_step__functions.h"
#include "turtlebot3_msgs/srv/detail/drl_step__struct.h"


// Include directives for member types
// Member `action`
// Member `previous_action`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void DrlStep_Request__rosidl_typesupport_introspection_c__DrlStep_Request_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  turtlebot3_msgs__srv__DrlStep_Request__init(message_memory);
}

void DrlStep_Request__rosidl_typesupport_introspection_c__DrlStep_Request_fini_function(void * message_memory)
{
  turtlebot3_msgs__srv__DrlStep_Request__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember DrlStep_Request__rosidl_typesupport_introspection_c__DrlStep_Request_message_member_array[2] = {
  {
    "action",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(turtlebot3_msgs__srv__DrlStep_Request, action),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "previous_action",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(turtlebot3_msgs__srv__DrlStep_Request, previous_action),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers DrlStep_Request__rosidl_typesupport_introspection_c__DrlStep_Request_message_members = {
  "turtlebot3_msgs__srv",  // message namespace
  "DrlStep_Request",  // message name
  2,  // number of fields
  sizeof(turtlebot3_msgs__srv__DrlStep_Request),
  DrlStep_Request__rosidl_typesupport_introspection_c__DrlStep_Request_message_member_array,  // message members
  DrlStep_Request__rosidl_typesupport_introspection_c__DrlStep_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  DrlStep_Request__rosidl_typesupport_introspection_c__DrlStep_Request_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t DrlStep_Request__rosidl_typesupport_introspection_c__DrlStep_Request_message_type_support_handle = {
  0,
  &DrlStep_Request__rosidl_typesupport_introspection_c__DrlStep_Request_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_turtlebot3_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, turtlebot3_msgs, srv, DrlStep_Request)() {
  if (!DrlStep_Request__rosidl_typesupport_introspection_c__DrlStep_Request_message_type_support_handle.typesupport_identifier) {
    DrlStep_Request__rosidl_typesupport_introspection_c__DrlStep_Request_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &DrlStep_Request__rosidl_typesupport_introspection_c__DrlStep_Request_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "turtlebot3_msgs/srv/detail/drl_step__rosidl_typesupport_introspection_c.h"
// already included above
// #include "turtlebot3_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "turtlebot3_msgs/srv/detail/drl_step__functions.h"
// already included above
// #include "turtlebot3_msgs/srv/detail/drl_step__struct.h"


// Include directives for member types
// Member `state`
// already included above
// #include "rosidl_runtime_c/primitives_sequence_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void DrlStep_Response__rosidl_typesupport_introspection_c__DrlStep_Response_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  turtlebot3_msgs__srv__DrlStep_Response__init(message_memory);
}

void DrlStep_Response__rosidl_typesupport_introspection_c__DrlStep_Response_fini_function(void * message_memory)
{
  turtlebot3_msgs__srv__DrlStep_Response__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember DrlStep_Response__rosidl_typesupport_introspection_c__DrlStep_Response_message_member_array[5] = {
  {
    "state",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(turtlebot3_msgs__srv__DrlStep_Response, state),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "reward",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(turtlebot3_msgs__srv__DrlStep_Response, reward),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "done",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(turtlebot3_msgs__srv__DrlStep_Response, done),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "success",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(turtlebot3_msgs__srv__DrlStep_Response, success),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "distance_traveled",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(turtlebot3_msgs__srv__DrlStep_Response, distance_traveled),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers DrlStep_Response__rosidl_typesupport_introspection_c__DrlStep_Response_message_members = {
  "turtlebot3_msgs__srv",  // message namespace
  "DrlStep_Response",  // message name
  5,  // number of fields
  sizeof(turtlebot3_msgs__srv__DrlStep_Response),
  DrlStep_Response__rosidl_typesupport_introspection_c__DrlStep_Response_message_member_array,  // message members
  DrlStep_Response__rosidl_typesupport_introspection_c__DrlStep_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  DrlStep_Response__rosidl_typesupport_introspection_c__DrlStep_Response_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t DrlStep_Response__rosidl_typesupport_introspection_c__DrlStep_Response_message_type_support_handle = {
  0,
  &DrlStep_Response__rosidl_typesupport_introspection_c__DrlStep_Response_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_turtlebot3_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, turtlebot3_msgs, srv, DrlStep_Response)() {
  if (!DrlStep_Response__rosidl_typesupport_introspection_c__DrlStep_Response_message_type_support_handle.typesupport_identifier) {
    DrlStep_Response__rosidl_typesupport_introspection_c__DrlStep_Response_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &DrlStep_Response__rosidl_typesupport_introspection_c__DrlStep_Response_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "turtlebot3_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "turtlebot3_msgs/srv/detail/drl_step__rosidl_typesupport_introspection_c.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/service_introspection.h"

// this is intentionally not const to allow initialization later to prevent an initialization race
static rosidl_typesupport_introspection_c__ServiceMembers turtlebot3_msgs__srv__detail__drl_step__rosidl_typesupport_introspection_c__DrlStep_service_members = {
  "turtlebot3_msgs__srv",  // service namespace
  "DrlStep",  // service name
  // these two fields are initialized below on the first access
  NULL,  // request message
  // turtlebot3_msgs__srv__detail__drl_step__rosidl_typesupport_introspection_c__DrlStep_Request_message_type_support_handle,
  NULL  // response message
  // turtlebot3_msgs__srv__detail__drl_step__rosidl_typesupport_introspection_c__DrlStep_Response_message_type_support_handle
};

static rosidl_service_type_support_t turtlebot3_msgs__srv__detail__drl_step__rosidl_typesupport_introspection_c__DrlStep_service_type_support_handle = {
  0,
  &turtlebot3_msgs__srv__detail__drl_step__rosidl_typesupport_introspection_c__DrlStep_service_members,
  get_service_typesupport_handle_function,
};

// Forward declaration of request/response type support functions
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, turtlebot3_msgs, srv, DrlStep_Request)();

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, turtlebot3_msgs, srv, DrlStep_Response)();

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_turtlebot3_msgs
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, turtlebot3_msgs, srv, DrlStep)() {
  if (!turtlebot3_msgs__srv__detail__drl_step__rosidl_typesupport_introspection_c__DrlStep_service_type_support_handle.typesupport_identifier) {
    turtlebot3_msgs__srv__detail__drl_step__rosidl_typesupport_introspection_c__DrlStep_service_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  rosidl_typesupport_introspection_c__ServiceMembers * service_members =
    (rosidl_typesupport_introspection_c__ServiceMembers *)turtlebot3_msgs__srv__detail__drl_step__rosidl_typesupport_introspection_c__DrlStep_service_type_support_handle.data;

  if (!service_members->request_members_) {
    service_members->request_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, turtlebot3_msgs, srv, DrlStep_Request)()->data;
  }
  if (!service_members->response_members_) {
    service_members->response_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, turtlebot3_msgs, srv, DrlStep_Response)()->data;
  }

  return &turtlebot3_msgs__srv__detail__drl_step__rosidl_typesupport_introspection_c__DrlStep_service_type_support_handle;
}

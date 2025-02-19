// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from turtlebot3_msgs:action/Patrol.idl
// generated code does not contain a copyright notice

#ifndef TURTLEBOT3_MSGS__ACTION__DETAIL__PATROL__TRAITS_HPP_
#define TURTLEBOT3_MSGS__ACTION__DETAIL__PATROL__TRAITS_HPP_

#include "turtlebot3_msgs/action/detail/patrol__struct.hpp"
#include <rosidl_runtime_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<turtlebot3_msgs::action::Patrol_Goal>()
{
  return "turtlebot3_msgs::action::Patrol_Goal";
}

template<>
inline const char * name<turtlebot3_msgs::action::Patrol_Goal>()
{
  return "turtlebot3_msgs/action/Patrol_Goal";
}

template<>
struct has_fixed_size<turtlebot3_msgs::action::Patrol_Goal>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<turtlebot3_msgs::action::Patrol_Goal>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<turtlebot3_msgs::action::Patrol_Goal>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<turtlebot3_msgs::action::Patrol_Result>()
{
  return "turtlebot3_msgs::action::Patrol_Result";
}

template<>
inline const char * name<turtlebot3_msgs::action::Patrol_Result>()
{
  return "turtlebot3_msgs/action/Patrol_Result";
}

template<>
struct has_fixed_size<turtlebot3_msgs::action::Patrol_Result>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<turtlebot3_msgs::action::Patrol_Result>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<turtlebot3_msgs::action::Patrol_Result>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<turtlebot3_msgs::action::Patrol_Feedback>()
{
  return "turtlebot3_msgs::action::Patrol_Feedback";
}

template<>
inline const char * name<turtlebot3_msgs::action::Patrol_Feedback>()
{
  return "turtlebot3_msgs/action/Patrol_Feedback";
}

template<>
struct has_fixed_size<turtlebot3_msgs::action::Patrol_Feedback>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<turtlebot3_msgs::action::Patrol_Feedback>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<turtlebot3_msgs::action::Patrol_Feedback>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'goal_id'
#include "unique_identifier_msgs/msg/detail/uuid__traits.hpp"
// Member 'goal'
#include "turtlebot3_msgs/action/detail/patrol__traits.hpp"

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<turtlebot3_msgs::action::Patrol_SendGoal_Request>()
{
  return "turtlebot3_msgs::action::Patrol_SendGoal_Request";
}

template<>
inline const char * name<turtlebot3_msgs::action::Patrol_SendGoal_Request>()
{
  return "turtlebot3_msgs/action/Patrol_SendGoal_Request";
}

template<>
struct has_fixed_size<turtlebot3_msgs::action::Patrol_SendGoal_Request>
  : std::integral_constant<bool, has_fixed_size<turtlebot3_msgs::action::Patrol_Goal>::value && has_fixed_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct has_bounded_size<turtlebot3_msgs::action::Patrol_SendGoal_Request>
  : std::integral_constant<bool, has_bounded_size<turtlebot3_msgs::action::Patrol_Goal>::value && has_bounded_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct is_message<turtlebot3_msgs::action::Patrol_SendGoal_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__traits.hpp"

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<turtlebot3_msgs::action::Patrol_SendGoal_Response>()
{
  return "turtlebot3_msgs::action::Patrol_SendGoal_Response";
}

template<>
inline const char * name<turtlebot3_msgs::action::Patrol_SendGoal_Response>()
{
  return "turtlebot3_msgs/action/Patrol_SendGoal_Response";
}

template<>
struct has_fixed_size<turtlebot3_msgs::action::Patrol_SendGoal_Response>
  : std::integral_constant<bool, has_fixed_size<builtin_interfaces::msg::Time>::value> {};

template<>
struct has_bounded_size<turtlebot3_msgs::action::Patrol_SendGoal_Response>
  : std::integral_constant<bool, has_bounded_size<builtin_interfaces::msg::Time>::value> {};

template<>
struct is_message<turtlebot3_msgs::action::Patrol_SendGoal_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<turtlebot3_msgs::action::Patrol_SendGoal>()
{
  return "turtlebot3_msgs::action::Patrol_SendGoal";
}

template<>
inline const char * name<turtlebot3_msgs::action::Patrol_SendGoal>()
{
  return "turtlebot3_msgs/action/Patrol_SendGoal";
}

template<>
struct has_fixed_size<turtlebot3_msgs::action::Patrol_SendGoal>
  : std::integral_constant<
    bool,
    has_fixed_size<turtlebot3_msgs::action::Patrol_SendGoal_Request>::value &&
    has_fixed_size<turtlebot3_msgs::action::Patrol_SendGoal_Response>::value
  >
{
};

template<>
struct has_bounded_size<turtlebot3_msgs::action::Patrol_SendGoal>
  : std::integral_constant<
    bool,
    has_bounded_size<turtlebot3_msgs::action::Patrol_SendGoal_Request>::value &&
    has_bounded_size<turtlebot3_msgs::action::Patrol_SendGoal_Response>::value
  >
{
};

template<>
struct is_service<turtlebot3_msgs::action::Patrol_SendGoal>
  : std::true_type
{
};

template<>
struct is_service_request<turtlebot3_msgs::action::Patrol_SendGoal_Request>
  : std::true_type
{
};

template<>
struct is_service_response<turtlebot3_msgs::action::Patrol_SendGoal_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__traits.hpp"

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<turtlebot3_msgs::action::Patrol_GetResult_Request>()
{
  return "turtlebot3_msgs::action::Patrol_GetResult_Request";
}

template<>
inline const char * name<turtlebot3_msgs::action::Patrol_GetResult_Request>()
{
  return "turtlebot3_msgs/action/Patrol_GetResult_Request";
}

template<>
struct has_fixed_size<turtlebot3_msgs::action::Patrol_GetResult_Request>
  : std::integral_constant<bool, has_fixed_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct has_bounded_size<turtlebot3_msgs::action::Patrol_GetResult_Request>
  : std::integral_constant<bool, has_bounded_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct is_message<turtlebot3_msgs::action::Patrol_GetResult_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'result'
// already included above
// #include "turtlebot3_msgs/action/detail/patrol__traits.hpp"

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<turtlebot3_msgs::action::Patrol_GetResult_Response>()
{
  return "turtlebot3_msgs::action::Patrol_GetResult_Response";
}

template<>
inline const char * name<turtlebot3_msgs::action::Patrol_GetResult_Response>()
{
  return "turtlebot3_msgs/action/Patrol_GetResult_Response";
}

template<>
struct has_fixed_size<turtlebot3_msgs::action::Patrol_GetResult_Response>
  : std::integral_constant<bool, has_fixed_size<turtlebot3_msgs::action::Patrol_Result>::value> {};

template<>
struct has_bounded_size<turtlebot3_msgs::action::Patrol_GetResult_Response>
  : std::integral_constant<bool, has_bounded_size<turtlebot3_msgs::action::Patrol_Result>::value> {};

template<>
struct is_message<turtlebot3_msgs::action::Patrol_GetResult_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<turtlebot3_msgs::action::Patrol_GetResult>()
{
  return "turtlebot3_msgs::action::Patrol_GetResult";
}

template<>
inline const char * name<turtlebot3_msgs::action::Patrol_GetResult>()
{
  return "turtlebot3_msgs/action/Patrol_GetResult";
}

template<>
struct has_fixed_size<turtlebot3_msgs::action::Patrol_GetResult>
  : std::integral_constant<
    bool,
    has_fixed_size<turtlebot3_msgs::action::Patrol_GetResult_Request>::value &&
    has_fixed_size<turtlebot3_msgs::action::Patrol_GetResult_Response>::value
  >
{
};

template<>
struct has_bounded_size<turtlebot3_msgs::action::Patrol_GetResult>
  : std::integral_constant<
    bool,
    has_bounded_size<turtlebot3_msgs::action::Patrol_GetResult_Request>::value &&
    has_bounded_size<turtlebot3_msgs::action::Patrol_GetResult_Response>::value
  >
{
};

template<>
struct is_service<turtlebot3_msgs::action::Patrol_GetResult>
  : std::true_type
{
};

template<>
struct is_service_request<turtlebot3_msgs::action::Patrol_GetResult_Request>
  : std::true_type
{
};

template<>
struct is_service_response<turtlebot3_msgs::action::Patrol_GetResult_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__traits.hpp"
// Member 'feedback'
// already included above
// #include "turtlebot3_msgs/action/detail/patrol__traits.hpp"

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<turtlebot3_msgs::action::Patrol_FeedbackMessage>()
{
  return "turtlebot3_msgs::action::Patrol_FeedbackMessage";
}

template<>
inline const char * name<turtlebot3_msgs::action::Patrol_FeedbackMessage>()
{
  return "turtlebot3_msgs/action/Patrol_FeedbackMessage";
}

template<>
struct has_fixed_size<turtlebot3_msgs::action::Patrol_FeedbackMessage>
  : std::integral_constant<bool, has_fixed_size<turtlebot3_msgs::action::Patrol_Feedback>::value && has_fixed_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct has_bounded_size<turtlebot3_msgs::action::Patrol_FeedbackMessage>
  : std::integral_constant<bool, has_bounded_size<turtlebot3_msgs::action::Patrol_Feedback>::value && has_bounded_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct is_message<turtlebot3_msgs::action::Patrol_FeedbackMessage>
  : std::true_type {};

}  // namespace rosidl_generator_traits


namespace rosidl_generator_traits
{

template<>
struct is_action<turtlebot3_msgs::action::Patrol>
  : std::true_type
{
};

template<>
struct is_action_goal<turtlebot3_msgs::action::Patrol_Goal>
  : std::true_type
{
};

template<>
struct is_action_result<turtlebot3_msgs::action::Patrol_Result>
  : std::true_type
{
};

template<>
struct is_action_feedback<turtlebot3_msgs::action::Patrol_Feedback>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits


#endif  // TURTLEBOT3_MSGS__ACTION__DETAIL__PATROL__TRAITS_HPP_

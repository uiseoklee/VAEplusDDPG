// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from turtlebot3_msgs:srv/DrlStep.idl
// generated code does not contain a copyright notice

#ifndef TURTLEBOT3_MSGS__SRV__DETAIL__DRL_STEP__STRUCT_HPP_
#define TURTLEBOT3_MSGS__SRV__DETAIL__DRL_STEP__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


#ifndef _WIN32
# define DEPRECATED__turtlebot3_msgs__srv__DrlStep_Request __attribute__((deprecated))
#else
# define DEPRECATED__turtlebot3_msgs__srv__DrlStep_Request __declspec(deprecated)
#endif

namespace turtlebot3_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct DrlStep_Request_
{
  using Type = DrlStep_Request_<ContainerAllocator>;

  explicit DrlStep_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
  }

  explicit DrlStep_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
    (void)_alloc;
  }

  // field types and members
  using _action_type =
    std::vector<float, typename ContainerAllocator::template rebind<float>::other>;
  _action_type action;
  using _previous_action_type =
    std::vector<float, typename ContainerAllocator::template rebind<float>::other>;
  _previous_action_type previous_action;

  // setters for named parameter idiom
  Type & set__action(
    const std::vector<float, typename ContainerAllocator::template rebind<float>::other> & _arg)
  {
    this->action = _arg;
    return *this;
  }
  Type & set__previous_action(
    const std::vector<float, typename ContainerAllocator::template rebind<float>::other> & _arg)
  {
    this->previous_action = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    turtlebot3_msgs::srv::DrlStep_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const turtlebot3_msgs::srv::DrlStep_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<turtlebot3_msgs::srv::DrlStep_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<turtlebot3_msgs::srv::DrlStep_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      turtlebot3_msgs::srv::DrlStep_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<turtlebot3_msgs::srv::DrlStep_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      turtlebot3_msgs::srv::DrlStep_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<turtlebot3_msgs::srv::DrlStep_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<turtlebot3_msgs::srv::DrlStep_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<turtlebot3_msgs::srv::DrlStep_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__turtlebot3_msgs__srv__DrlStep_Request
    std::shared_ptr<turtlebot3_msgs::srv::DrlStep_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__turtlebot3_msgs__srv__DrlStep_Request
    std::shared_ptr<turtlebot3_msgs::srv::DrlStep_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const DrlStep_Request_ & other) const
  {
    if (this->action != other.action) {
      return false;
    }
    if (this->previous_action != other.previous_action) {
      return false;
    }
    return true;
  }
  bool operator!=(const DrlStep_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct DrlStep_Request_

// alias to use template instance with default allocator
using DrlStep_Request =
  turtlebot3_msgs::srv::DrlStep_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace turtlebot3_msgs


#ifndef _WIN32
# define DEPRECATED__turtlebot3_msgs__srv__DrlStep_Response __attribute__((deprecated))
#else
# define DEPRECATED__turtlebot3_msgs__srv__DrlStep_Response __declspec(deprecated)
#endif

namespace turtlebot3_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct DrlStep_Response_
{
  using Type = DrlStep_Response_<ContainerAllocator>;

  explicit DrlStep_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->reward = 0.0f;
      this->done = false;
      this->success = 0ul;
      this->distance_traveled = 0.0f;
    }
  }

  explicit DrlStep_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->reward = 0.0f;
      this->done = false;
      this->success = 0ul;
      this->distance_traveled = 0.0f;
    }
  }

  // field types and members
  using _state_type =
    std::vector<float, typename ContainerAllocator::template rebind<float>::other>;
  _state_type state;
  using _reward_type =
    float;
  _reward_type reward;
  using _done_type =
    bool;
  _done_type done;
  using _success_type =
    uint32_t;
  _success_type success;
  using _distance_traveled_type =
    float;
  _distance_traveled_type distance_traveled;

  // setters for named parameter idiom
  Type & set__state(
    const std::vector<float, typename ContainerAllocator::template rebind<float>::other> & _arg)
  {
    this->state = _arg;
    return *this;
  }
  Type & set__reward(
    const float & _arg)
  {
    this->reward = _arg;
    return *this;
  }
  Type & set__done(
    const bool & _arg)
  {
    this->done = _arg;
    return *this;
  }
  Type & set__success(
    const uint32_t & _arg)
  {
    this->success = _arg;
    return *this;
  }
  Type & set__distance_traveled(
    const float & _arg)
  {
    this->distance_traveled = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    turtlebot3_msgs::srv::DrlStep_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const turtlebot3_msgs::srv::DrlStep_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<turtlebot3_msgs::srv::DrlStep_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<turtlebot3_msgs::srv::DrlStep_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      turtlebot3_msgs::srv::DrlStep_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<turtlebot3_msgs::srv::DrlStep_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      turtlebot3_msgs::srv::DrlStep_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<turtlebot3_msgs::srv::DrlStep_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<turtlebot3_msgs::srv::DrlStep_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<turtlebot3_msgs::srv::DrlStep_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__turtlebot3_msgs__srv__DrlStep_Response
    std::shared_ptr<turtlebot3_msgs::srv::DrlStep_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__turtlebot3_msgs__srv__DrlStep_Response
    std::shared_ptr<turtlebot3_msgs::srv::DrlStep_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const DrlStep_Response_ & other) const
  {
    if (this->state != other.state) {
      return false;
    }
    if (this->reward != other.reward) {
      return false;
    }
    if (this->done != other.done) {
      return false;
    }
    if (this->success != other.success) {
      return false;
    }
    if (this->distance_traveled != other.distance_traveled) {
      return false;
    }
    return true;
  }
  bool operator!=(const DrlStep_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct DrlStep_Response_

// alias to use template instance with default allocator
using DrlStep_Response =
  turtlebot3_msgs::srv::DrlStep_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace turtlebot3_msgs

namespace turtlebot3_msgs
{

namespace srv
{

struct DrlStep
{
  using Request = turtlebot3_msgs::srv::DrlStep_Request;
  using Response = turtlebot3_msgs::srv::DrlStep_Response;
};

}  // namespace srv

}  // namespace turtlebot3_msgs

#endif  // TURTLEBOT3_MSGS__SRV__DETAIL__DRL_STEP__STRUCT_HPP_

// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from turtlebot3_msgs:srv/Goal.idl
// generated code does not contain a copyright notice

#ifndef TURTLEBOT3_MSGS__SRV__DETAIL__GOAL__STRUCT_HPP_
#define TURTLEBOT3_MSGS__SRV__DETAIL__GOAL__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


#ifndef _WIN32
# define DEPRECATED__turtlebot3_msgs__srv__Goal_Request __attribute__((deprecated))
#else
# define DEPRECATED__turtlebot3_msgs__srv__Goal_Request __declspec(deprecated)
#endif

namespace turtlebot3_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct Goal_Request_
{
  using Type = Goal_Request_<ContainerAllocator>;

  explicit Goal_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->empty = false;
    }
  }

  explicit Goal_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->empty = false;
    }
  }

  // field types and members
  using _empty_type =
    bool;
  _empty_type empty;

  // setters for named parameter idiom
  Type & set__empty(
    const bool & _arg)
  {
    this->empty = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    turtlebot3_msgs::srv::Goal_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const turtlebot3_msgs::srv::Goal_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<turtlebot3_msgs::srv::Goal_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<turtlebot3_msgs::srv::Goal_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      turtlebot3_msgs::srv::Goal_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<turtlebot3_msgs::srv::Goal_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      turtlebot3_msgs::srv::Goal_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<turtlebot3_msgs::srv::Goal_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<turtlebot3_msgs::srv::Goal_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<turtlebot3_msgs::srv::Goal_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__turtlebot3_msgs__srv__Goal_Request
    std::shared_ptr<turtlebot3_msgs::srv::Goal_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__turtlebot3_msgs__srv__Goal_Request
    std::shared_ptr<turtlebot3_msgs::srv::Goal_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Goal_Request_ & other) const
  {
    if (this->empty != other.empty) {
      return false;
    }
    return true;
  }
  bool operator!=(const Goal_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Goal_Request_

// alias to use template instance with default allocator
using Goal_Request =
  turtlebot3_msgs::srv::Goal_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace turtlebot3_msgs


#ifndef _WIN32
# define DEPRECATED__turtlebot3_msgs__srv__Goal_Response __attribute__((deprecated))
#else
# define DEPRECATED__turtlebot3_msgs__srv__Goal_Response __declspec(deprecated)
#endif

namespace turtlebot3_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct Goal_Response_
{
  using Type = Goal_Response_<ContainerAllocator>;

  explicit Goal_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->new_goal = false;
    }
  }

  explicit Goal_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->new_goal = false;
    }
  }

  // field types and members
  using _new_goal_type =
    bool;
  _new_goal_type new_goal;

  // setters for named parameter idiom
  Type & set__new_goal(
    const bool & _arg)
  {
    this->new_goal = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    turtlebot3_msgs::srv::Goal_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const turtlebot3_msgs::srv::Goal_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<turtlebot3_msgs::srv::Goal_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<turtlebot3_msgs::srv::Goal_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      turtlebot3_msgs::srv::Goal_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<turtlebot3_msgs::srv::Goal_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      turtlebot3_msgs::srv::Goal_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<turtlebot3_msgs::srv::Goal_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<turtlebot3_msgs::srv::Goal_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<turtlebot3_msgs::srv::Goal_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__turtlebot3_msgs__srv__Goal_Response
    std::shared_ptr<turtlebot3_msgs::srv::Goal_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__turtlebot3_msgs__srv__Goal_Response
    std::shared_ptr<turtlebot3_msgs::srv::Goal_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Goal_Response_ & other) const
  {
    if (this->new_goal != other.new_goal) {
      return false;
    }
    return true;
  }
  bool operator!=(const Goal_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Goal_Response_

// alias to use template instance with default allocator
using Goal_Response =
  turtlebot3_msgs::srv::Goal_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace turtlebot3_msgs

namespace turtlebot3_msgs
{

namespace srv
{

struct Goal
{
  using Request = turtlebot3_msgs::srv::Goal_Request;
  using Response = turtlebot3_msgs::srv::Goal_Response;
};

}  // namespace srv

}  // namespace turtlebot3_msgs

#endif  // TURTLEBOT3_MSGS__SRV__DETAIL__GOAL__STRUCT_HPP_

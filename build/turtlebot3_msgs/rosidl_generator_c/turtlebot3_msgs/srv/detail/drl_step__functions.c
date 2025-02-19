// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from turtlebot3_msgs:srv/DrlStep.idl
// generated code does not contain a copyright notice
#include "turtlebot3_msgs/srv/detail/drl_step__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"

// Include directives for member types
// Member `action`
// Member `previous_action`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

bool
turtlebot3_msgs__srv__DrlStep_Request__init(turtlebot3_msgs__srv__DrlStep_Request * msg)
{
  if (!msg) {
    return false;
  }
  // action
  if (!rosidl_runtime_c__float__Sequence__init(&msg->action, 0)) {
    turtlebot3_msgs__srv__DrlStep_Request__fini(msg);
    return false;
  }
  // previous_action
  if (!rosidl_runtime_c__float__Sequence__init(&msg->previous_action, 0)) {
    turtlebot3_msgs__srv__DrlStep_Request__fini(msg);
    return false;
  }
  return true;
}

void
turtlebot3_msgs__srv__DrlStep_Request__fini(turtlebot3_msgs__srv__DrlStep_Request * msg)
{
  if (!msg) {
    return;
  }
  // action
  rosidl_runtime_c__float__Sequence__fini(&msg->action);
  // previous_action
  rosidl_runtime_c__float__Sequence__fini(&msg->previous_action);
}

bool
turtlebot3_msgs__srv__DrlStep_Request__are_equal(const turtlebot3_msgs__srv__DrlStep_Request * lhs, const turtlebot3_msgs__srv__DrlStep_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // action
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->action), &(rhs->action)))
  {
    return false;
  }
  // previous_action
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->previous_action), &(rhs->previous_action)))
  {
    return false;
  }
  return true;
}

bool
turtlebot3_msgs__srv__DrlStep_Request__copy(
  const turtlebot3_msgs__srv__DrlStep_Request * input,
  turtlebot3_msgs__srv__DrlStep_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // action
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->action), &(output->action)))
  {
    return false;
  }
  // previous_action
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->previous_action), &(output->previous_action)))
  {
    return false;
  }
  return true;
}

turtlebot3_msgs__srv__DrlStep_Request *
turtlebot3_msgs__srv__DrlStep_Request__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  turtlebot3_msgs__srv__DrlStep_Request * msg = (turtlebot3_msgs__srv__DrlStep_Request *)allocator.allocate(sizeof(turtlebot3_msgs__srv__DrlStep_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(turtlebot3_msgs__srv__DrlStep_Request));
  bool success = turtlebot3_msgs__srv__DrlStep_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
turtlebot3_msgs__srv__DrlStep_Request__destroy(turtlebot3_msgs__srv__DrlStep_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    turtlebot3_msgs__srv__DrlStep_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
turtlebot3_msgs__srv__DrlStep_Request__Sequence__init(turtlebot3_msgs__srv__DrlStep_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  turtlebot3_msgs__srv__DrlStep_Request * data = NULL;

  if (size) {
    data = (turtlebot3_msgs__srv__DrlStep_Request *)allocator.zero_allocate(size, sizeof(turtlebot3_msgs__srv__DrlStep_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = turtlebot3_msgs__srv__DrlStep_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        turtlebot3_msgs__srv__DrlStep_Request__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
turtlebot3_msgs__srv__DrlStep_Request__Sequence__fini(turtlebot3_msgs__srv__DrlStep_Request__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      turtlebot3_msgs__srv__DrlStep_Request__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

turtlebot3_msgs__srv__DrlStep_Request__Sequence *
turtlebot3_msgs__srv__DrlStep_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  turtlebot3_msgs__srv__DrlStep_Request__Sequence * array = (turtlebot3_msgs__srv__DrlStep_Request__Sequence *)allocator.allocate(sizeof(turtlebot3_msgs__srv__DrlStep_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = turtlebot3_msgs__srv__DrlStep_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
turtlebot3_msgs__srv__DrlStep_Request__Sequence__destroy(turtlebot3_msgs__srv__DrlStep_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    turtlebot3_msgs__srv__DrlStep_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
turtlebot3_msgs__srv__DrlStep_Request__Sequence__are_equal(const turtlebot3_msgs__srv__DrlStep_Request__Sequence * lhs, const turtlebot3_msgs__srv__DrlStep_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!turtlebot3_msgs__srv__DrlStep_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
turtlebot3_msgs__srv__DrlStep_Request__Sequence__copy(
  const turtlebot3_msgs__srv__DrlStep_Request__Sequence * input,
  turtlebot3_msgs__srv__DrlStep_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(turtlebot3_msgs__srv__DrlStep_Request);
    turtlebot3_msgs__srv__DrlStep_Request * data =
      (turtlebot3_msgs__srv__DrlStep_Request *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!turtlebot3_msgs__srv__DrlStep_Request__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          turtlebot3_msgs__srv__DrlStep_Request__fini(&data[i]);
        }
        free(data);
        return false;
      }
    }
    output->data = data;
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!turtlebot3_msgs__srv__DrlStep_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `state`
// already included above
// #include "rosidl_runtime_c/primitives_sequence_functions.h"

bool
turtlebot3_msgs__srv__DrlStep_Response__init(turtlebot3_msgs__srv__DrlStep_Response * msg)
{
  if (!msg) {
    return false;
  }
  // state
  if (!rosidl_runtime_c__float__Sequence__init(&msg->state, 0)) {
    turtlebot3_msgs__srv__DrlStep_Response__fini(msg);
    return false;
  }
  // reward
  // done
  // success
  // distance_traveled
  return true;
}

void
turtlebot3_msgs__srv__DrlStep_Response__fini(turtlebot3_msgs__srv__DrlStep_Response * msg)
{
  if (!msg) {
    return;
  }
  // state
  rosidl_runtime_c__float__Sequence__fini(&msg->state);
  // reward
  // done
  // success
  // distance_traveled
}

bool
turtlebot3_msgs__srv__DrlStep_Response__are_equal(const turtlebot3_msgs__srv__DrlStep_Response * lhs, const turtlebot3_msgs__srv__DrlStep_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // state
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->state), &(rhs->state)))
  {
    return false;
  }
  // reward
  if (lhs->reward != rhs->reward) {
    return false;
  }
  // done
  if (lhs->done != rhs->done) {
    return false;
  }
  // success
  if (lhs->success != rhs->success) {
    return false;
  }
  // distance_traveled
  if (lhs->distance_traveled != rhs->distance_traveled) {
    return false;
  }
  return true;
}

bool
turtlebot3_msgs__srv__DrlStep_Response__copy(
  const turtlebot3_msgs__srv__DrlStep_Response * input,
  turtlebot3_msgs__srv__DrlStep_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // state
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->state), &(output->state)))
  {
    return false;
  }
  // reward
  output->reward = input->reward;
  // done
  output->done = input->done;
  // success
  output->success = input->success;
  // distance_traveled
  output->distance_traveled = input->distance_traveled;
  return true;
}

turtlebot3_msgs__srv__DrlStep_Response *
turtlebot3_msgs__srv__DrlStep_Response__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  turtlebot3_msgs__srv__DrlStep_Response * msg = (turtlebot3_msgs__srv__DrlStep_Response *)allocator.allocate(sizeof(turtlebot3_msgs__srv__DrlStep_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(turtlebot3_msgs__srv__DrlStep_Response));
  bool success = turtlebot3_msgs__srv__DrlStep_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
turtlebot3_msgs__srv__DrlStep_Response__destroy(turtlebot3_msgs__srv__DrlStep_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    turtlebot3_msgs__srv__DrlStep_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
turtlebot3_msgs__srv__DrlStep_Response__Sequence__init(turtlebot3_msgs__srv__DrlStep_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  turtlebot3_msgs__srv__DrlStep_Response * data = NULL;

  if (size) {
    data = (turtlebot3_msgs__srv__DrlStep_Response *)allocator.zero_allocate(size, sizeof(turtlebot3_msgs__srv__DrlStep_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = turtlebot3_msgs__srv__DrlStep_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        turtlebot3_msgs__srv__DrlStep_Response__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
turtlebot3_msgs__srv__DrlStep_Response__Sequence__fini(turtlebot3_msgs__srv__DrlStep_Response__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      turtlebot3_msgs__srv__DrlStep_Response__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

turtlebot3_msgs__srv__DrlStep_Response__Sequence *
turtlebot3_msgs__srv__DrlStep_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  turtlebot3_msgs__srv__DrlStep_Response__Sequence * array = (turtlebot3_msgs__srv__DrlStep_Response__Sequence *)allocator.allocate(sizeof(turtlebot3_msgs__srv__DrlStep_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = turtlebot3_msgs__srv__DrlStep_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
turtlebot3_msgs__srv__DrlStep_Response__Sequence__destroy(turtlebot3_msgs__srv__DrlStep_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    turtlebot3_msgs__srv__DrlStep_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
turtlebot3_msgs__srv__DrlStep_Response__Sequence__are_equal(const turtlebot3_msgs__srv__DrlStep_Response__Sequence * lhs, const turtlebot3_msgs__srv__DrlStep_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!turtlebot3_msgs__srv__DrlStep_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
turtlebot3_msgs__srv__DrlStep_Response__Sequence__copy(
  const turtlebot3_msgs__srv__DrlStep_Response__Sequence * input,
  turtlebot3_msgs__srv__DrlStep_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(turtlebot3_msgs__srv__DrlStep_Response);
    turtlebot3_msgs__srv__DrlStep_Response * data =
      (turtlebot3_msgs__srv__DrlStep_Response *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!turtlebot3_msgs__srv__DrlStep_Response__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          turtlebot3_msgs__srv__DrlStep_Response__fini(&data[i]);
        }
        free(data);
        return false;
      }
    }
    output->data = data;
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!turtlebot3_msgs__srv__DrlStep_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}

# generated from rosidl_generator_py/resource/_idl.py.em
# with input from turtlebot3_msgs:srv/RingGoal.idl
# generated code does not contain a copyright notice


# Import statements for member types

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_RingGoal_Request(type):
    """Metaclass of message 'RingGoal_Request'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('turtlebot3_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'turtlebot3_msgs.srv.RingGoal_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__ring_goal__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__ring_goal__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__ring_goal__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__ring_goal__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__ring_goal__request

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class RingGoal_Request(metaclass=Metaclass_RingGoal_Request):
    """Message class 'RingGoal_Request'."""

    __slots__ = [
        '_robot_pose_x',
        '_robot_pose_y',
        '_radius',
    ]

    _fields_and_field_types = {
        'robot_pose_x': 'float',
        'robot_pose_y': 'float',
        'radius': 'float',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.robot_pose_x = kwargs.get('robot_pose_x', float())
        self.robot_pose_y = kwargs.get('robot_pose_y', float())
        self.radius = kwargs.get('radius', float())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.robot_pose_x != other.robot_pose_x:
            return False
        if self.robot_pose_y != other.robot_pose_y:
            return False
        if self.radius != other.radius:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @property
    def robot_pose_x(self):
        """Message field 'robot_pose_x'."""
        return self._robot_pose_x

    @robot_pose_x.setter
    def robot_pose_x(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'robot_pose_x' field must be of type 'float'"
        self._robot_pose_x = value

    @property
    def robot_pose_y(self):
        """Message field 'robot_pose_y'."""
        return self._robot_pose_y

    @robot_pose_y.setter
    def robot_pose_y(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'robot_pose_y' field must be of type 'float'"
        self._robot_pose_y = value

    @property
    def radius(self):
        """Message field 'radius'."""
        return self._radius

    @radius.setter
    def radius(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'radius' field must be of type 'float'"
        self._radius = value


# Import statements for member types

# already imported above
# import rosidl_parser.definition


class Metaclass_RingGoal_Response(type):
    """Metaclass of message 'RingGoal_Response'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('turtlebot3_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'turtlebot3_msgs.srv.RingGoal_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__ring_goal__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__ring_goal__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__ring_goal__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__ring_goal__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__ring_goal__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class RingGoal_Response(metaclass=Metaclass_RingGoal_Response):
    """Message class 'RingGoal_Response'."""

    __slots__ = [
    ]

    _fields_and_field_types = {
    }

    SLOT_TYPES = (
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)


class Metaclass_RingGoal(type):
    """Metaclass of service 'RingGoal'."""

    _TYPE_SUPPORT = None

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('turtlebot3_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'turtlebot3_msgs.srv.RingGoal')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__ring_goal

            from turtlebot3_msgs.srv import _ring_goal
            if _ring_goal.Metaclass_RingGoal_Request._TYPE_SUPPORT is None:
                _ring_goal.Metaclass_RingGoal_Request.__import_type_support__()
            if _ring_goal.Metaclass_RingGoal_Response._TYPE_SUPPORT is None:
                _ring_goal.Metaclass_RingGoal_Response.__import_type_support__()


class RingGoal(metaclass=Metaclass_RingGoal):
    from turtlebot3_msgs.srv._ring_goal import RingGoal_Request as Request
    from turtlebot3_msgs.srv._ring_goal import RingGoal_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')

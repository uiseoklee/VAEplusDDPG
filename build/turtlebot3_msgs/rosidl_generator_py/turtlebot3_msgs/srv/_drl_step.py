# generated from rosidl_generator_py/resource/_idl.py.em
# with input from turtlebot3_msgs:srv/DrlStep.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'action'
# Member 'previous_action'
import array  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_DrlStep_Request(type):
    """Metaclass of message 'DrlStep_Request'."""

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
                'turtlebot3_msgs.srv.DrlStep_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__drl_step__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__drl_step__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__drl_step__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__drl_step__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__drl_step__request

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class DrlStep_Request(metaclass=Metaclass_DrlStep_Request):
    """Message class 'DrlStep_Request'."""

    __slots__ = [
        '_action',
        '_previous_action',
    ]

    _fields_and_field_types = {
        'action': 'sequence<float>',
        'previous_action': 'sequence<float>',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('float')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('float')),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.action = array.array('f', kwargs.get('action', []))
        self.previous_action = array.array('f', kwargs.get('previous_action', []))

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
        if self.action != other.action:
            return False
        if self.previous_action != other.previous_action:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @property
    def action(self):
        """Message field 'action'."""
        return self._action

    @action.setter
    def action(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'f', \
                "The 'action' array.array() must have the type code of 'f'"
            self._action = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, float) for v in value) and
                 True), \
                "The 'action' field must be a set or sequence and each value of type 'float'"
        self._action = array.array('f', value)

    @property
    def previous_action(self):
        """Message field 'previous_action'."""
        return self._previous_action

    @previous_action.setter
    def previous_action(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'f', \
                "The 'previous_action' array.array() must have the type code of 'f'"
            self._previous_action = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, float) for v in value) and
                 True), \
                "The 'previous_action' field must be a set or sequence and each value of type 'float'"
        self._previous_action = array.array('f', value)


# Import statements for member types

# Member 'state'
# already imported above
# import array

# already imported above
# import rosidl_parser.definition


class Metaclass_DrlStep_Response(type):
    """Metaclass of message 'DrlStep_Response'."""

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
                'turtlebot3_msgs.srv.DrlStep_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__drl_step__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__drl_step__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__drl_step__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__drl_step__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__drl_step__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class DrlStep_Response(metaclass=Metaclass_DrlStep_Response):
    """Message class 'DrlStep_Response'."""

    __slots__ = [
        '_state',
        '_reward',
        '_done',
        '_success',
        '_distance_traveled',
    ]

    _fields_and_field_types = {
        'state': 'sequence<float>',
        'reward': 'float',
        'done': 'boolean',
        'success': 'uint32',
        'distance_traveled': 'float',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('float')),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.state = array.array('f', kwargs.get('state', []))
        self.reward = kwargs.get('reward', float())
        self.done = kwargs.get('done', bool())
        self.success = kwargs.get('success', int())
        self.distance_traveled = kwargs.get('distance_traveled', float())

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
        if self.state != other.state:
            return False
        if self.reward != other.reward:
            return False
        if self.done != other.done:
            return False
        if self.success != other.success:
            return False
        if self.distance_traveled != other.distance_traveled:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @property
    def state(self):
        """Message field 'state'."""
        return self._state

    @state.setter
    def state(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'f', \
                "The 'state' array.array() must have the type code of 'f'"
            self._state = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, float) for v in value) and
                 True), \
                "The 'state' field must be a set or sequence and each value of type 'float'"
        self._state = array.array('f', value)

    @property
    def reward(self):
        """Message field 'reward'."""
        return self._reward

    @reward.setter
    def reward(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'reward' field must be of type 'float'"
        self._reward = value

    @property
    def done(self):
        """Message field 'done'."""
        return self._done

    @done.setter
    def done(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'done' field must be of type 'bool'"
        self._done = value

    @property
    def success(self):
        """Message field 'success'."""
        return self._success

    @success.setter
    def success(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'success' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'success' field must be an unsigned integer in [0, 4294967295]"
        self._success = value

    @property
    def distance_traveled(self):
        """Message field 'distance_traveled'."""
        return self._distance_traveled

    @distance_traveled.setter
    def distance_traveled(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'distance_traveled' field must be of type 'float'"
        self._distance_traveled = value


class Metaclass_DrlStep(type):
    """Metaclass of service 'DrlStep'."""

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
                'turtlebot3_msgs.srv.DrlStep')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__drl_step

            from turtlebot3_msgs.srv import _drl_step
            if _drl_step.Metaclass_DrlStep_Request._TYPE_SUPPORT is None:
                _drl_step.Metaclass_DrlStep_Request.__import_type_support__()
            if _drl_step.Metaclass_DrlStep_Response._TYPE_SUPPORT is None:
                _drl_step.Metaclass_DrlStep_Response.__import_type_support__()


class DrlStep(metaclass=Metaclass_DrlStep):
    from turtlebot3_msgs.srv._drl_step import DrlStep_Request as Request
    from turtlebot3_msgs.srv._drl_step import DrlStep_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')

import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/dmsgv1/VAEplusDDPG/install/turtlebot3_drl'

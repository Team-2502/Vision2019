# This is a NetworkTables client (eg, the DriverStation/coprocessor side).
# You need to tell it the IP address of the NetworkTables server (the
# robot or simulator).
#
# When running, this will continue incrementing the value 'dsTime', and the
# value should be visible to other networktables clients and the robot.
#

import sys
import time
from networktables import NetworkTables
from networktables.util import ntproperty

# To see messages from networktables, you must setup logging
import logging

logging.basicConfig(level=logging.DEBUG)

ip = "10.25.2.2"

NetworkTables.initialize(server=ip)

class SomeClient(object):
    """Demonstrates an object with magic networktables properties"""

    robotTime = ntproperty("/SmartDashboard/robotTime", 0, writeDefault=False)

    dsTime = ntproperty("/SmartDashboard/tvecs1", 0)


c = SomeClient()


i = 0
while True:

    # equivalent to wpilib.SmartDashboard.getNumber('robotTime', None)
    print("robotTime:", c.dsTime)

    # equivalent to wpilib.SmartDashboard.putNumber('dsTime', i)
    c.dsTime = i

    time.sleep(1)
    i += 1

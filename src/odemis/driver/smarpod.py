# -*- coding: utf-8 -*-
'''
Created on 17 April 2019

@author: Anders Muskens

Copyright © 2012-2019 Anders Muskens, Delmic

This file is part of Odemis.

Odemis is free software: you can redistribute it and/or modify it under the terms
of the GNU General Public License version 2 as published by the Free Software
Foundation.

Odemis is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
Odemis. If not, see http://www.gnu.org/licenses/.

This driver supports SmarAct and SmarPod actuators, which are accessed via a C DLL library
provided by SmarAct. This must be installed on the system for this actuator to run. Please
refer to the SmarAct readme for Linux installation instructions.
'''
from __future__ import division
from concurrent.futures import CancelledError, TimeoutError

import os
import logging
import time
import math
from ctypes import *
import copy
import threading

from odemis import model
from odemis.util import driver
from odemis.model import HwError, CancellableFuture, CancellableThreadPoolExecutor, isasync
from operator import pos


def add_coord(pos1, pos2):
    """
    Adds two coordinate dictionaries together and returns a new coordinate dictionary.

    NOTE: pos1 and pos2 MUST contain the same axes values

    pos1: dict (axis name str) -> (float)
    pos2: dict (axis name str) -> (float)
    Returns ret
        dict (axis name str) -> (float)
    """
    ret = pos1.copy()
    for an, v in pos2.items():
        ret[an] += v

    return ret


class Pose(Structure):
    """
    SmarPod Pose Structure (C Struct used by DLL)

    Note: internally, the system uses metres and degrees for rotation
    """
    _fields_ = [
        ("positionX", c_double),
        ("positionY", c_double),
        ("positionZ", c_double),
        ("rotationX", c_double),
        ("rotationY", c_double),
        ("rotationZ", c_double),
        ]
    
    def __add__(self, o):
        pose = Pose()
        pose.positionX = self.positionX + o.positionX
        pose.positionY = self.positionY + o.positionY
        pose.positionZ = self.positionZ + o.positionZ
        pose.rotationX = self.rotationX + o.rotationX
        pose.rotationY = self.rotationY + o.rotationY
        pose.rotationZ = self.rotationZ + o.rotationZ
        return pose

    def __sub__(self, o):
        pose = Pose()
        pose.positionX = self.positionX - o.positionX
        pose.positionY = self.positionY - o.positionY
        pose.positionZ = self.positionZ - o.positionZ
        pose.rotationX = self.rotationX - o.rotationX
        pose.rotationY = self.rotationY - o.rotationY
        pose.rotationZ = self.rotationZ - o.rotationZ
        return pose


def pose_to_dict(pose):
    """
    Convert a Pose (C structure) to a coordinate dictionary (str) -> (double)
        pose: a Pose (C struct)
    returns: Coordinate dictionary (str) -> (double) of axis name to value
    """
    pos = {}
    pos['x'] = pose.positionX
    pos['y'] = pose.positionY
    pos['z'] = pose.positionZ

    # Note: internally, the system uses metres and degrees for rotation
    pos['rx'] = pose.rotationX * (math.pi / 180.0)
    pos['ry'] = pose.rotationY * (math.pi / 180.0)
    pos['rz'] = pose.rotationZ * (math.pi / 180.0)
    return pos

def dict_to_pose(pos):
    """
    Convert a coordinate dictionary (str) -> (double) to a Pose C struct
        pos: Coordinate dictionary (str) -> (double) of axis name to value
    returns: a Pose (C struct)
    raises ValueError if an unsupported axis name is input
    """
    pose = Pose()

    # Note: internally, the system uses metres and degrees for rotation
    for an, v in pos.items():
        if an == "x":
            pose.positionX = v
        elif an == "y":
            pose.positionY = v
        elif an == "z":
            pose.positionZ = v
        elif an == "rx":
            pose.rotationX = v * (180.0 / math.pi)
        elif an == "ry":
            pose.rotationY = v * (180.0 / math.pi)
        elif an == "rz":
            pose.rotationZ = v * (180.0 / math.pi)
        else:
            raise ValueError("Invalid axis")
    return pose


class SmarPodDLL(CDLL):
    """
    Subclass of CDLL specific to SmarPod library, which handles error codes for
    all the functions automatically.
    """
    
    hwModel = c_long(10001)  # specifies the SmarPod 110.45 S (nano)

    # Status
    SMARPOD_OK = 0
    SMARPOD_OTHER_ERROR = 1
    SMARPOD_SYSTEM_NOT_INITIALIZED_ERROR = 2
    SMARPOD_NO_SYSTEMS_FOUND_ERROR = 3
    SMARPOD_INVALID_PARAMETER_ERROR = 4
    SMARPOD_COMMUNICATION_ERROR = 5
    SMARPOD_UNKNOWN_PROPERTY_ERROR = 6
    SMARPOD_RESOURCE_TOO_OLD_ERROR = 7
    SMARPOD_FEATURE_UNAVAILABLE_ERROR = 8
    SMARPOD_INVALID_SYSTEM_LOCATOR_ERROR = 9
    SMARPOD_QUERYBUFFER_SIZE_ERROR = 10
    SMARPOD_COMMUNICATION_TIMEOUT_ERROR = 11
    SMARPOD_DRIVER_ERROR = 12
    SMARPOD_STATUS_CODE_UNKNOWN_ERROR = 500
    SMARPOD_INVALID_ID_ERROR = 501
    SMARPOD_INITIALIZED_ERROR = 502
    SMARPOD_HARDWARE_MODEL_UNKNOWN_ERROR = 503
    SMARPOD_WRONG_COMM_MODE_ERROR = 504
    SMARPOD_NOT_INITIALIZED_ERROR = 505
    SMARPOD_INVALID_SYSTEM_ID_ERROR = 506
    SMARPOD_NOT_ENOUGH_CHANNELS_ERROR = 507
    SMARPOD_INVALID_CHANNEL_ERROR = 508
    SMARPOD_CHANNEL_USED_ERROR = 509
    SMARPOD_SENSORS_DISABLED_ERROR = 510
    SMARPOD_WRONG_SENSOR_TYPE_ERROR = 511
    SMARPOD_SYSTEM_CONFIGURATION_ERROR = 512
    SMARPOD_SENSOR_NOT_FOUND_ERROR = 513
    SMARPOD_STOPPED_ERROR = 514
    SMARPOD_BUSY_ERROR = 515
    SMARPOD_NOT_REFERENCED_ERROR = 550
    SMARPOD_POSE_UNREACHABLE_ERROR = 551

    # Defines
    SMARPOD_SENSORS_DISABLED = c_uint(0)
    SMARPOD_SENSORS_ENABLED = c_uint(1)
    SMARPOD_SENSORS_POWERSAVE = c_uint(2)

    # property symbols
    SMARPOD_FREF_METHOD = c_uint(1000)
    SMARPOD_FREF_ZDIRECTION = c_uint(1002)
    SMARPOD_FREF_XDIRECTION = c_uint(1003)
    SMARPOD_FREF_YDIRECTION = c_uint(1004)
    SMARPOD_PIVOT_MODE = c_uint(1010)
    SMARPOD_FREF_AND_CAL_FREQUENCY = c_uint(1020)
    SMARPOD_POSITIONERS_MIN_SPEED = c_uint(1100)

    # move-status constants
    SMARPOD_STOPPED = c_uint(0)
    SMARPOD_HOLDING = c_uint(1)
    SMARPOD_MOVING = c_uint(2)
    SMARPOD_CALIBRATING = c_uint(3)
    SMARPOD_REFERENCING = c_uint(4)
    SMARPOD_STANDBY = c_uint(5)
    
    SMARPOD_HOLDTIME_INFINITE = c_uint(60000)

    err_code = {
0: "OK",
1: "OTHER_ERROR",
2: "SYSTEM_NOT_INITIALIZED_ERROR",
3: "NO_SYSTEMS_FOUND_ERROR",
4: "INVALID_PARAMETER_ERROR",
5: "COMMUNICATION_ERROR",
6: "UNKNOWN_PROPERTY_ERROR",
7: "RESOURCE_TOO_OLD_ERROR",
8: "FEATURE_UNAVAILABLE_ERROR",
9: "INVALID_SYSTEM_LOCATOR_ERROR",
10: "QUERYBUFFER_SIZE_ERROR",
11: "COMMUNICATION_TIMEOUT_ERROR",
12: "DRIVER_ERROR",
500: "STATUS_CODE_UNKNOWN_ERROR",
501: "INVALID_ID_ERROR",
503: "HARDWARE_MODEL_UNKNOWN_ERROR",
504: "WRONG_COMM_MODE_ERROR",
505: "NOT_INITIALIZED_ERROR",
506: "INVALID_SYSTEM_ID_ERROR",
507: "NOT_ENOUGH_CHANNELS_ERROR",
510: "SENSORS_DISABLED_ERROR",
511: "WRONG_SENSOR_TYPE_ERROR",
512: "SYSTEM_CONFIGURATION_ERROR",
513: "SENSOR_NOT_FOUND_ERROR",
514: "STOPPED_ERROR",
515: "BUSY_ERROR",
550: "NOT_REFERENCED_ERROR",
551: "POSE_UNREACHABLE_ERROR",
552: "COMMAND_OVERRIDDEN_ERROR",
553: "ENDSTOP_REACHED_ERROR",
554: "NOT_STOPPED_ERROR",
555: "COULD_NOT_REFERENCE_ERROR",
556: "COULD_NOT_CALIBRATE_ERROR",
        }

    def __init__(self):
        if os.name == "nt":
            raise NotImplemented("Windows not yet supported")
            # WinDLL.__init__(self, "libsmarpod.dll")  # TODO check it works
            # atmcd64d.dll on 64 bits
        else:
            # Global so that its sub-libraries can access it
            CDLL.__init__(self, "libsmarpod.so", RTLD_GLOBAL)

    def __getitem__(self, name):
        try:
            func = super(SmarPodDLL, self).__getitem__(name)
        except Exception:
            raise AttributeError("Failed to find %s" % (name,))
        func.__name__ = name
        func.errcheck = self.sp_errcheck
        return func

    @staticmethod
    def sp_errcheck(result, func, args):
        """
        Analyse the retuhwModelrn value of a call and raise an exception in case of
        error.
        Follows the ctypes.errcheck callback convention
        """
        # everything returns DRV_SUCCESS on correct usage, _except_ GetTemperature()
        if result != SmarPodDLL.SMARPOD_OK:
            raise SmarPodError(result)

        return result


class SmarPodError(Exception):
    """
    SmarPod Exception
    """
    def __init__(self, error_code):
        self.errno = error_code
        super(SmarPodError, self).__init__("Error %d. %s" % (error_code, SmarPodDLL.err_code.get(error_code, "")))


class SmarPod(model.Actuator):
    
    def __init__(self, name, role, locator, ref_on_init=False, actuator_speed=0.1,
                 axes=None, **kwargs):
        """
        A driver for a SmarAct SmarPod Actuator.
        This driver uses a DLL provided by SmarAct which connects via
        USB or TCP/IP using a locator string.

        name: (str)
        role: (str)
        locator: (str) Use "fake" for a simulator.
            For a real device, MCS controllers with USB interface can be addressed with the
            following locator syntax:
                usb:id:<id>
            where <id> is the first part of a USB devices serial number which
            is printed on the MCS controller.
            If the controller has a TCP/IP connection, use:
                network:<ip>:<port>
        ref_on_init: (bool) determines if the controller should automatically reference
            on initialization
        actuator_speed: (double) the default speed (in m/s) of the actuators
        axes: dict str (axis name) -> dict (axis parameters)
            axis parameters: {
                range: [float, float], default is -1 -> 1
                unit: (str) default will be set to 'm'
            }
        """
        if not axes:
            raise ValueError("Needs at least 1 axis.")

        if locator != "fake":
            self.core = SmarPodDLL()
        else:
            self.core = FakeSmarPodDLL()
            
        # Not to be mistaken with axes which is a simple public view
        self._axis_map = {}  # axis name -> axis number used by controller
        axes_def = {}  # axis name -> Axis object
        self._locator = c_char_p(locator.encode("ascii"))
        self._options = c_char_p("".encode("ascii"))  # In the current version, this must be an empty string.

        for axis_name, axis_par in axes.items():
            try:
                axis_range = axis_par['range']
            except KeyError:
                logging.info("Axis %s has no range. Assuming (-1, 1)", axis_name)
                axis_range = (-1, 1)

            try:
                axis_unit = axis_par['unit']
            except KeyError:
                logging.info("Axis %s has no unit. Assuming m", axis_name)
                axis_unit = "m"

            ad = model.Axis(canAbs=True, unit=axis_unit, range=axis_range)
            axes_def[axis_name] = ad
            
            
        # Connect to the device
        self._id = c_uint()
        self.core.Smarpod_Open(byref(self._id), SmarPodDLL.hwModel, self._locator, self._options)
        logging.debug("Successfully connected to SmarPod Controller ID %d", self._id.value)
        self.core.Smarpod_SetSensorMode(self._id, SmarPodDLL.SMARPOD_SENSORS_ENABLED)

        model.Actuator.__init__(self, name, role, axes=axes_def, **kwargs)

        # Add metadata
        self._swVersion = self.GetSwVersion()
        self._metadata[model.MD_SW_VERSION] = self._swVersion
        logging.debug("Using SmarPod library version %s", self._swVersion)

        self.position = model.VigilantAttribute({}, readonly=True)

        # will take care of executing axis move asynchronously
        self._executor = CancellableThreadPoolExecutor(1)  # one task at a time

        axes_ref = {a: False for a, i in self.axes.items()}
        # VA dict str(axis) -> bool
        self.referenced = model.VigilantAttribute(axes_ref, readonly=True)
        referenced = c_int()
        self.core.Smarpod_IsReferenced(self._id, byref(referenced))
        if not referenced.value:
            logging.debug("SmarPod is not referenced. The device will not function until referencing occurs.")
            if ref_on_init:
                self.reference().result()
        else:
            logging.debug("SmarPod is referenced")

        # Use a default actuator speed
        self.SetSpeed(actuator_speed)
        self._speed = self.GetSpeed()
        self._accel = self.GetAcceleration()

        try:
            self._updatePosition()    # will fail if not referenced.
        except SmarPodError as ex:
            if ex.errno == SmarPodDLL.SMARPOD_NOT_REFERENCED_ERROR:
                logging.info("Must reference SmarPod controller before use.")
            else:
                raise

    def terminate(self):
        # should be safe to close the device multiple times if terminate is called more than once.
        self.core.Smarpod_Close(self._id)
        super(SmarPod, self).terminate()
        
    def GetSwVersion(self):
        """
        Request the software version from the DLL file
        """
        major = c_uint()
        minor = c_uint()
        update = c_uint()
        self.core.Smarpod_GetDLLVersion(byref(major), byref(minor), byref(update))
        ver = "%u.%u.%u" % (major.value, minor.value, update.value)
        return ver

    def IsReferenced(self):
        """
        Ask the controller if it is referenced
        """
        referenced = c_int()
        self.core.Smarpod_IsReferenced(self._id, byref(referenced))
        return bool(referenced.value)
        
    def GetMoveStatus(self):
        """
        Gets the move status from the controller.
        Returns:
            SmarPodDLL.SMARPOD_MOVING is returned if moving
            SmarPodDLL.SMARPOD_STOPPED when stopped
            SmarPodDLL.SMARPOD_HOLDING when holding between moves
            SmarPodDLL.SMARPOD_CALIBRATING when calibrating
            SmarPodDLL.SMARPOD_REFERENCING when referencing
            SmarPodDLL.SMARPOD_STANDBY
        """
        status = c_uint()
        self.core.Smarpod_GetMoveStatus(self._id, byref(status))
        return status

    def Move(self, pos, hold_time=0, block=False):
        """
        Move to pose command. Non-blocking
        pos: (dict str -> float): axis name -> position
            This is converted to the pose C-struct which is sent to the SmarPod DLL
        hold_time: (int) specify in milliseconds how long to hold after the move.
        block: (bool): Set to True if the function should block until the move completes
        Raises: SmarPodError if a problem occurs
        """
        # convert into a smartpad pose
        newPose = dict_to_pose(pos)

        # Use an infiinite holdtime and non-blocking (final argument)
        self.core.Smarpod_Move(self._id, byref(newPose), c_uint(hold_time), c_int(block))

    def GetPose(self):
        """
        Get the current pose of the SmarPod

        returns: (dict str -> float): axis name -> position
        """
        pose = Pose()
        self.core.Smarpod_GetPose(self._id, byref(pose))
        position = pose_to_dict(pose)
        logging.info("Current position: %s", position)
        return position

    def Stop(self):
        """
        Stop command sent to the SmarPod
        """
        logging.debug("Stopping...")
        self.core.Smarpod_Stop(self._id)

    def SetSpeed(self, value):
        """
        Set the speed of the SmarPod motion
        value: (double) indicating speed for all axes
        """
        logging.debug("Setting speed to %f", value)
        # the second argument (1) turns on speed control.
        self.core.Smarpod_SetSpeed(self._id, c_int(1), c_double(value))

    def GetSpeed(self):
        """
        Returns (double) the speed of the SmarPod motion
        """
        speed_control = c_int()
        speed = c_double()
        self.core.Smarpod_GetSpeed(self._id, byref(speed_control), byref(speed))
        return speed.value

    def SetAcceleration(self, value):
        """
        Set the acceleration of the SmarPod motion
        value: (double) indicating acceleration for all axes
        """
        logging.debug("Setting acceleration to %f", value)
        # Passing 1 enables acceleration control.
        self.core.Smarpod_SetAcceleration(self._id, c_int(1), c_double(value))

    def GetAcceleration(self):
        """
        Returns (double) the acceleration of the SmarPod motion
        """
        acceleration_control = c_int()
        acceleration = c_double()
        self.core.Smarpod_GetAcceleration(self._id, byref(acceleration_control), byref(acceleration))
        return acceleration.value
    
    def IsPoseReachable(self, pos):
        """
        Ask the controller if a pose is reachable
        pos: (dict of str -> float): a coordinate dictionary of axis name to value
        returns: true if the pose is reachable - false otherwise.
        """
        reachable = c_int()
        self.core.Smarpod_IsPoseReachable(self._id, byref(dict_to_pose(pos)), byref(reachable))
        return bool(reachable.value)
    
    def stop(self, axes=None):
        """
        Stop the SmarPod controller and update position
        """
        self.Stop()
        self._updatePosition()

    def _updatePosition(self):
        """
        update the position VA
        """
        pos = self.GetPose()
        pos = self._applyInversion(pos)
        logging.debug("Updated position to %s", pos)
        self.position._set_value(pos, force_write=True)

    def _createMoveFuture(self, ref=False):
        """
        ref: if true, will use a different canceller
        Return (CancellableFuture): a future that can be used to manage a move
        """
        f = CancellableFuture()
        f._moving_lock = threading.Lock()  # taken while moving
        f._must_stop = threading.Event()  # cancel of the current future requested
        f._was_stopped = False  # if cancel was successful
        if not ref:
            f.task_canceller = self._cancelCurrentMove
        else:
            f.task_canceller = self._cancelReference
        return f

    @isasync
    def reference(self, _=None):
        """
        reference usually takes axes as an argument. However, the SmarPod references all
        axes together so this argument is extraneous.
        """
        f = self._createMoveFuture(ref=True)
        self._executor.submitf(f, self._doReference, f)
        return f

    def _doReference(self, future):
        """
        Actually runs the referencing code
        future (Future): the future it handles
        raise:
            IOError: if referencing failed due to hardware
            CancelledError if was cancelled
        """
        # Reset reference so that if it fails, it states the axes are not
        # referenced (anymore)
        with future._moving_lock:
            try:
                # set the referencing for all axes to fals
                self.referenced._value = {a: False for a in self.axes.keys()}

                # The SmarPod references all axes at once. This function blocks
                self.core.Smarpod_FindReferenceMarks(self._id)

                if self.IsReferenced():
                    self.referenced._value = {a: True for a in self.axes.keys()}
                    self._updatePosition()

            except SmarPodError as ex:
                future._was_stopped = True
                # This occurs if a stop command interrupts referencing
                if ex.errno == SmarPodDLL.SMARPOD_STOPPED_ERROR:
                    logging.info("Referencing stopped: %s", ex)
                    raise CancelledError()
                else:
                    raise
            except Exception:
                logging.exception("Referencing failure")
                raise
            finally:
                # We only notify after updating the position so that when a listener
                # receives updates both values are already updated.
                # read-only so manually notify
                self.referenced.notify(self.referenced.value)

    @isasync
    def moveAbs(self, pos):
        """
        API call to absolute move
        """
        if not pos:
            return model.InstantaneousFuture()
        
        self._checkMoveAbs(pos)
        if not self.IsPoseReachable(pos):
            raise ValueError("Pose %s is not reachable by the SmarPod controller" % (pos,))

        f = self._createMoveFuture()
        f = self._executor.submitf(f, self._doMoveAbs, f, pos)
        return f

    def _doMoveAbs(self, future, pos):
        """
        Blocking and cancellable absolute move
        future (Future): the future it handles
        _pos (dict str -> float): axis name -> absolute target position
        raise:
            SmarPodError: if the controller reported an error
            CancelledError: if cancelled before the end of the move
        """
        last_upd = time.time()
        dur = 30  # TODO: Calculate an estimated move duration
        end = time.time() + dur
        max_dur = dur * 2 + 1
        logging.debug("Expecting a move of %g s, will wait up to %g s", dur, max_dur)
        timeout = last_upd + max_dur

        with future._moving_lock:
            self.Move(pos)
            while not future._must_stop.is_set():
                status = self.GetMoveStatus()
                # check if move is done
                if status.value == SmarPodDLL.SMARPOD_STOPPED.value:
                    break

                now = time.time()
                if now > timeout:
                    logging.warning("Stopping move due to timeout after %g s.", max_dur)
                    self.stop()
                    raise TimeoutError("Move is not over after %g s, while "
                                       "expected it takes only %g s" %
                                       (max_dur, dur))

                # Update the position from time to time (10 Hz)
                if now - last_upd > 0.1:
                    self._updatePosition()
                    last_upd = time.time()

                # Wait half of the time left (maximum 0.1 s)
                left = end - time.time()
                sleept = max(0.001, min(left / 2, 0.1))
                future._must_stop.wait(sleept)
            else:
                self.stop()
                future._was_stopped = True
                raise CancelledError()

        self._updatePosition()

        logging.debug("move successfully completed")

    def _cancelCurrentMove(self, future):
        """
        Cancels the current move (both absolute or relative). Non-blocking.
        future (Future): the future to stop. Unused, only one future must be
         running at a time.
        return (bool): True if it successfully cancelled (stopped) the move.
        """
        # The difficulty is to synchronise correctly when:
        #  * the task is just starting (not finished requesting axes to move)
        #  * the task is finishing (about to say that it finished successfully)
        logging.debug("Canceling current move")

        future._must_stop.set()  # tell the thread taking care of the move it's over
        with future._moving_lock:
            if not future._was_stopped:
                logging.debug("Canceling failed")
            self._updatePosition()
            return future._was_stopped

    def _cancelReference(self, future):
        # The difficulty is to synchronize correctly when:
        #  * the task is just starting (about to request axes to move)
        #  * the task is finishing (about to say that it finished successfully)
        logging.debug("Canceling current referencing")

        self.Stop()
        future._must_stop.set()  # tell the thread taking care of the referencing it's over

        # Synchronise with the ending of the future
        with future._moving_lock:

            if not future._was_stopped:
                logging.debug("Cancelling failed")
            return future._was_stopped

    @isasync
    def moveRel(self, shift):
        """
        API call for relative move
        """
        if not shift:
            return model.InstantaneousFuture()

        self._checkMoveRel(shift)

        f = self._createMoveFuture()
        f = self._executor.submitf(f, self._doMoveRel, f, shift)
        return f
    
    def _doMoveRel(self, future, shift):
        """
        Do a relative move by converting it into an absolute move
        """
        pos = add_coord(self.position.value, shift)
        self._doMoveAbs(future, pos)

# Only for testing/simulation purpose
# Very rough version that is just enough so that if the wrapper behaves correctly,
# it returns the expected values.


def _deref(p, typep):
    """
    p (byref object)
    typep (c_type): type of pointer
    Use .value to change the value of the object
    """
    # This is using internal ctypes attributes, that might change in later
    # versions. Ugly!
    # Another possibility would be to redefine byref by identity function:
    # byref= lambda x: x
    # and then dereferencing would be also identity function.
    return typep.from_address(addressof(p._obj))


class FakeSmarPodDLL(object):
    """
    Fake SmarPod DLL for simulator
    """

    def __init__(self):
        self.pose = Pose()
        self.target = Pose()
        self.properties = {}
        self._speed = c_double(0)
        self._speed_control = c_int()
        self._accel_control = c_int()
        self._accel = c_double(0)
        self.referenced = False

        # Specify ranges
        self._range = {}
        self._range['x'] = (-1, 1)
        self._range['y'] = (-1, 1)
        self._range['z'] = (-1, 1)
        self._range['rx'] = (-45, 45)
        self._range['ry'] = (-45, 45)
        self._range['rz'] = (-45, 45)

        self.stopping = threading.Event()

        self._current_move_start = time.time()
        self._current_move_finish = time.time()

    def _pose_in_range(self, pose):
        if self._range['x'][0] <= pose.positionX <= self._range['x'][1] and \
            self._range['y'][0] <= pose.positionY <= self._range['y'][1] and \
            self._range['z'][0] <= pose.positionZ <= self._range['z'][1] and \
            self._range['rx'][0] <= pose.rotationX <= self._range['rx'][1] and \
            self._range['ry'][0] <= pose.rotationY <= self._range['ry'][1] and \
            self._range['rz'][0] <= pose.rotationZ <= self._range['rz'][1]:
            return True
        else:
            return False

    """
    DLL functions (fake)
    These functions are provided by the real SmarPod DLL
    """
    def Smarpod_Open(self, id, timeout, locator, options):
        pass

    def Smarpod_Close(self, id):
        pass

    def Smarpod_SetSensorMode(self, id, mode):
        pass

    def Smarpod_FindReferenceMarks(self, id):
        self.stopping.clear()
        time.sleep(0.5)
        if self.stopping.is_set():
            self.referenced = False
            raise SmarPodError(SmarPodDLL.SMARPOD_STOPPED_ERROR)
        else:
            self.referenced = True

    def Smarpod_IsPoseReachable(self, id, p_pos, p_reachable):
        reachable = _deref(p_reachable, c_int)
        pos = _deref(p_pos, Pose)
        if self._pose_in_range(pos):
            reachable.value = 1
        else:
            reachable.value = 0

    def Smarpod_IsReferenced(self, id, p_referenced):
        referenced = _deref(p_referenced, c_int)
        referenced.value = 1 if self.referenced else 0

    def Smarpod_Move(self, id, p_pose, hold_time, block):
        self.stopping.clear()
        pose = _deref(p_pose, Pose)
        if self._pose_in_range(pose):
            self._current_move_finish = time.time() + 1.0
            self.target.positionX = pose.positionX
            self.target.positionY = pose.positionY
            self.target.positionZ = pose.positionZ
            self.target.rotationX = pose.rotationX
            self.target.rotationY = pose.rotationY
            self.target.rotationZ = pose.rotationZ
        else:
            raise SmarPodError(SmarPodDLL.SMARPOD_POSE_UNREACHABLE_ERROR)

    def Smarpod_GetPose(self, id, p_pose):
        pose = _deref(p_pose, Pose)
        pose.positionX = self.pose.positionX
        pose.positionY = self.pose.positionY
        pose.positionZ = self.pose.positionZ
        pose.rotationX = self.pose.rotationX
        pose.rotationY = self.pose.rotationY
        pose.rotationZ = self.pose.rotationZ
        return SmarPodDLL.SMARPOD_OK

    def Smarpod_GetMoveStatus(self, id, p_status):
        status = _deref(p_status, c_int)

        if time.time() > self._current_move_finish:
            self.pose = copy.copy(self.target)
            status.value = SmarPodDLL.SMARPOD_STOPPED.value
        else:
            status.value = SmarPodDLL.SMARPOD_MOVING.value

    def Smarpod_Stop(self, id):
        self.stopping.set()

    def Smarpod_SetSpeed(self, id, speed_control, speed):
        self._speed = speed
        self._speed_control = speed_control

    def Smarpod_GetSpeed(self, id, p_speed_control, p_speed):
        speed = _deref(p_speed, c_double)
        speed.value = self._speed.value
        speed_control = _deref(p_speed_control, c_int)
        speed_control.value = self._speed_control.value

    def Smarpod_SetAcceleration(self, id, accel_control, accel):
        self._accel = accel
        self._accel_control = accel_control

    def Smarpod_GetAcceleration(self, id, p_accel_control, p_accel):
        accel = _deref(p_accel, c_double)
        accel.value = self._accel.value
        accel_control = _deref(p_accel_control, c_int)
        accel_control.value = self._accel_control.value

    def Smarpod_GetDLLVersion(self, p_major, p_minor, p_update):
        major = _deref(p_major, c_uint)
        major.value = 1
        minor = _deref(p_minor, c_uint)
        minor.value = 2
        update = _deref(p_update, c_uint)
        update.value = 3

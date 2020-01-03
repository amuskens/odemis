# -*- coding: utf-8 -*-
'''
Created on 6 Mar 2013

@author: Anders Muskens

Copyright © 2019 Anders Musknes, Delmic

This file is part of Odemis.

Odemis is free software: you can redistribute it and/or modify it under the terms
of the GNU General Public License version 2 as published by the Free Software
Foundation.

Odemis is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
Odemis. If not, see http://www.gnu.org/licenses/.
'''
from __future__ import division

import logging
import math
import numpy
from odemis import model
from odemis.model import isasync, ComponentBase, DataFlowBase
import odemis
import os

from ctypes import *


class AvantesDLL(CDLL):
    # Error
    ERR_SUCCESS = 0
    ERR_INVALID_PARAMETER = -1
    ERR_OPERATION_NOT_SUPPORTED = -2
    ERR_DEVICE_NOT_FOUND = -3
    ERR_INVALID_DEVICE_ID = -4
    ERR_OPERATION_PENDING = -5
    ERR_TIMEOUT = -6
    ERR_INVALID_PASSWORD = -7
    ERR_INVALID_MEAS_DATA = -8
    ERR_INVALID_SIZE = -9
    ERR_INVALID_PIXEL_RANGE = -10
    ERR_INVALID_INT_TIME = -11
    ERR_INVALID_COMBINATION = -12
    ERR_INVALID_CONFIGURATION = -13
    ERR_NO_MEAS_BUFFER_AVAIL = -14
    ERR_UNKNOWN = -15
    ERR_COMMUNICATION = -16
    ERR_NO_SPECTRA_IN_RAM = -17
    ERR_INVALID_DLL_VERSION = -18
    ERR_NO_MEMORY = -19
    ERR_DLL_INITIALISATION = -20
    ERR_INVALID_STATE = -21
    ERR_INVALID_REPLY = -22
    ERR_CONNECTION_FAILURE = ERR_COMMUNICATION
    ERR_ACCESS = -24

    # Error lookup
    err_code = {
        0: "ERR_SUCCESS",
        - 1: "ERR_INVALID_PARAMETER",
        - 2: "ERR_OPERATION_NOT_SUPPORTED",
        - 3: "ERR_DEVICE_NOT_FOUND",
        - 4: "ERR_INVALID_DEVICE_ID",
        - 5: "ERR_OPERATION_PENDING",
        - 6: "ERR_TIMEOUT",
        - 7: "ERR_INVALID_PASSWORD",
        - 8: "ERR_INVALID_MEAS_DATA",
        - 9: "ERR_INVALID_SIZE",
        - 10: "ERR_INVALID_PIXEL_RANGE",
        - 11: "ERR_INVALID_INT_TIME",
        - 12: "ERR_INVALID_COMBINATION",
        - 13: "ERR_INVALID_CONFIGURATION",
        - 14: "ERR_NO_MEAS_BUFFER_AVAIL",
        - 15: "ERR_UNKNOWN",
        - 16: "ERR_COMMUNICATION",
        - 17: "ERR_NO_SPECTRA_IN_RAM",
        - 18: "ERR_INVALID_DLL_VERSION",
        - 19: "ERR_NO_MEMORY",
        - 20: "ERR_DLL_INITIALISATION",
        - 21: "ERR_INVALID_STATE",
        - 22: "ERR_INVALID_REPLY",
        ERR_CONNECTION_FAILURE: "ERR_COMMUNICATION",
        - 24: "ERR_ACCESS"
        }

    MEASUREMENT_RESULT_FUNC = CFUNCTYPE(c_int, POINTER(c_int))

    def __init__(self):
        if os.name == "nt":
            raise NotImplemented("Windows not yet supported")
            # WinDLL.__init__(self, "lib.dll")  # TODO check it works
            # atmcd64d.dll on 64 bits
        else:
            # Global so that its sub-libraries can access it
            CDLL.__init__(self, "libname", RTLD_GLOBAL)

    def __getitem__(self, name):
        try:
            func = super(AvantesDLL, self).__getitem__(name)
        except Exception:
            raise AttributeError("Failed to find %s" % (name,))
        func.__name__ = name
        func.errcheck = self.av_errcheck
        return func

    @staticmethod
    def av_errcheck(result, func, args):
        """
        Analyse the retuhwModelrn value of a call and raise an exception in case of
        error.
        Follows the ctypes.errcheck callback convention
        """
        if result < AvantesDLL.SUCCESS:
            raise AvantesError(result)

        return result


class AvantesError(Exception):
    """
    Avantes Exception
    """

    def __init__(self, error_code):
        self.errno = error_code
        super(AvantesError, self).__init__("Error %d. %s" % (error_code, AvantesDLL.err_code.get(error_code, "")))


class Spectrometer(model.Detector):

    def __init__(self, name, role, dependencies, **kwargs):

        self.core = AvantesDLL()

        try:
            sp = dependencies["spectrograph"]
            if not isinstance(sp, ComponentBase):
                raise ValueError("Dependency spectrograph is not a component.")
            try:
                if "wavelength" not in sp.axes:
                    raise ValueError("Dependency spectrograph has no 'wavelength' axis.")
            except Exception:
                raise ValueError("Dependency spectrograph is not an Actuator.")
            self._spectrograph = sp
            self.dependencies.value.add(sp)
        except KeyError:
            self._spectrograph = None

        # Init the spectrometer
        port = c_short(0)  # 0 for USB port
        self.core.AVS_Init(port)

    def terminate(self):
        self.core.AVS_Done()


class Spectrograph(model.Actuator):
    """
    AviSpec Spectrograph
    """

    def __init__(self, name, role, wlp, children=None, **kwargs):
        """
        wlp (list of floats): polynomial for conversion from distance from the
          center of the CCD to wavelength (in m):
          w = wlp[0] + wlp[1] * x + wlp[2] * x²...
          where w is the wavelength (in m), x is the position from the center
          (in m, negative are to the left), and p is the polynomial
          (in m, m^0, m^-1...). So, typically, a first order
          polynomial contains as first element the center wavelength, and as
          second element the light dispersion (in m/m)
        """
        if kwargs.get("inverted", None):
            raise ValueError("Axis of spectrograph cannot be inverted")

        if not isinstance(wlp, list) or len(wlp) < 1:
            raise ValueError("wlp need to be a list of at least one float")

        # Note: it used to need a "ccd" child, but not anymore
        self._swVersion = "N/A (Odemis %s)" % odemis.__version__
        self._hwVersion = name

        self._wlp = wlp
        pos = {"wavelength": self._wlp[0]}
        wla = model.Axis(range=(0, 2400e-9), unit="m")
        model.Actuator.__init__(self, name, role, axes={"wavelength": wla},
                                **kwargs)
        self.position = model.VigilantAttribute(pos, unit="m", readonly=True)

    # We simulate the axis, to give the same interface as a fully controllable
    # spectrograph, but it has to actually reflect the state of the hardware.
    @isasync
    def moveRel(self, shift):
        # convert to a call to moveAbs
        new_pos = {}
        for axis, value in shift.items():
            new_pos[axis] = self.position.value[axis] + value
        return self.moveAbs(new_pos)

    @isasync
    def moveAbs(self, pos):
        for axis, value in pos.items():
            if axis == "wavelength":
                # it's read-only, so we change it via _value
                self.position._value[axis] = value
                self.position.notify(self.position.value)
            else:
                raise LookupError("Axis '%s' doesn't exist" % axis)

        return model.InstantaneousFuture()

    def stop(self, axes=None):
        # nothing to do
        pass


class SpecDataFlow(model.DataFlow):

    def __init__(self, comp, ccddf):
        """
        comp: the spectrometer instance
        ccddf (DataFlow): the dataflow of the real CCD
        """
        model.DataFlow.__init__(self)
        self.component = comp
        self._ccddf = ccddf
        self.active = False
        # Metadata is a little tricky because it must be synchronised with the
        # actual acquisition, but it's difficult to know with which settings
        # the acquisition was taken (when the settings are changing while
        # generating).
        self._beg_metadata = {}  # Metadata (more or less) at the beginning of the acquisition

    def start_generate(self):
        logging.debug("Activating Spectrometer acquisition")
        self.active = True
        self.component._applyCCDSettings()
        self._beg_metadata = self.component._metadata.copy()
        self._ccddf.subscribe(self._newFrame)

    def stop_generate(self):
        self._ccddf.unsubscribe(self._newFrame)
        self.active = False
        logging.debug("Spectrometer acquisition finished")
        # TODO: tell the component that it's over?

    def synchronizedOn(self, event):
        self._ccddf.synchronizedOn(event)

    def _newFrame(self, df, data):
        """
        Get the new frame from the detector
        """
        if data.shape[0] != 1:  # Shape is YX, so shape[0] is *vertical*
            logging.debug("Shape of spectrometer data is %s, binning vertical dim", data.shape)
            orig_dtype = data.dtype
            orig_shape = data.shape
            data = numpy.sum(data, axis=0)  # uint64 (if data.dtype is int)
            data.shape = (1,) + data.shape
            orig_bin = data.metadata.get(model.MD_BINNING, (1, 1))
            data.metadata[model.MD_BINNING] = orig_bin[0], orig_bin[1] * orig_shape[0]

            # Subtract baseline (aka black level) to avoid it from being multiplied,
            # so instead of having "Sum(data) + Sum(bl)", we have "Sum(data) + bl".
            try:
                baseline = data.metadata[model.MD_BASELINE]
                baseline_sum = orig_shape[0] * baseline
                # If the baseline is too high compared to the actual black, we
                # could end up subtracting too much, and values would underflow
                # => be extra careful and never subtract more than min value.
                minv = float(data.min())
                extra_bl = baseline_sum - baseline
                if extra_bl > minv:
                    extra_bl = minv
                    logging.info("Baseline reported at %d * %d, but lower values found, so only subtracting %d",
                                 baseline, orig_shape[0], extra_bl)

                # Same as "data -= extra_bl", but also works if extra_bl < 0
                numpy.subtract(data, extra_bl, out=data, casting="unsafe")
                data.metadata[model.MD_BASELINE] = baseline_sum - extra_bl
            except KeyError:
                pass

            # If int, revert to original type, with data clipped (not overflowing)
            if orig_dtype.kind in "biu":
                idtype = numpy.iinfo(orig_dtype)
                data = data.clip(idtype.min, idtype.max).astype(orig_dtype)

        # Check the metadata seems correct, and if not, recompute it on-the-fly
        md = self._beg_metadata
        # WL_LIST should be the same length as the data (excepted if the
        # spectrograph is in 0th order, then it should be empty or not present)
        wll = md.get(model.MD_WL_LIST)
        if wll and len(wll) != data.shape[1]:
            dmd = data.metadata
            logging.debug("WL_LIST len = %d vs %d", len(wll), data.shape[1])
            try:
                npixels = data.shape[1]
                pxs = dmd[model.MD_SENSOR_PIXEL_SIZE][0] * dmd[model.MD_BINNING][0]
                logging.info("Recomputing correct WL_LIST metadata")
                wll = self.component._spectrograph.getPixelToWavelength(npixels, pxs)
                if len(wll) == 0 and model.MD_WL_LIST in md:
                    del md[model.MD_WL_LIST]  # remove WL list from MD if empty
                else:
                    md[model.MD_WL_LIST] = wll
            except KeyError as ex:
                logging.warning("Failed to compute correct WL_LIST metadata: %s", ex)
            except Exception:
                logging.exception("Failed to compute WL_LIST metadata")

        # Remove non useful metadata
        for k in NON_SPEC_MD:
            data.metadata.pop(k, None)

        data.metadata.update(md)
        udata = self.component._transposeDAToUser(data)
        model.DataFlow.notify(self, udata)

        # If the acquisition continues, it will likely be using the current settings
        self._beg_metadata = self.component._metadata.copy()



'''
Created on Aug 24, 2018

@author: Anders Muskens

Copyright © 2018 Anders Muskens, Delmic

This file is part of Odemis.

Odemis is free software: you can redistribute it and/or modify it under the
terms  of the GNU General Public License version 2 as published by the Free
Software  Foundation.

Odemis is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY;  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR  PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
Odemis. If not, see http://www.gnu.org/licenses/.
'''

from __future__ import division

from concurrent.futures._base import CancelledError, CANCELLED, FINISHED, \
    RUNNING
import cv2
import logging
import math
from numpy import array, linalg
import numpy
from odemis import model
from odemis.acq.align import transform, spot, autofocus, FindOverlay
from odemis.acq.align.autofocus import AcquireNoBackground, MTD_EXHAUSTIVE
from odemis.acq.drift import MeasureShift
from odemis.dataio import tiff
from odemis.util import img, executeAsyncTask
import os
from scipy.ndimage import zoom
import threading
import time

logger = logging.getLogger(__name__)


def estimateResolutionShiftFactorTime(et):
    """
    Estimates Resolution-related shift calculation procedure duration
    returns (float):  process estimated time #s
    """
    # Approximately 28 acquisitions
    dur = 28 * et + 1
    return dur  # s

def _CancelFuture(future):
    """
    Canceller of task running in a future
    """
    logger.debug("Cancelling calculation...")

    with future._task_lock:
        if future._task_state == FINISHED:
            return False
        future._task_state = CANCELLED
        logger.debug("Calculation cancelled.")

    return True


def _discard_data(df, data):
    """
    Does nothing, just discard the SEM data received (for spot mode)
    """
    pass

def ResolutionShiftFactor(detector, escan, logpath=None):
    """
    Wrapper for DoResolutionShiftFactor. It provides the ability to check the
    progress of the procedure.
    detector (model.Detector): The se-detector
    escan (model.Emitter): The e-beam scanner
    logpath (string or None): if not None, will store the acquired SEM images
      in the directory.
    returns (ProgressiveFuture): Progress DoResolutionShiftFactor
    """
    # Create ProgressiveFuture and update its state to RUNNING
    est_start = time.time() + 0.1
    et = 7.5e-07 * numpy.prod(escan.resolution.range[1])
    f = model.ProgressiveFuture(start=est_start,
                                end=est_start + estimateResolutionShiftFactorTime(et))
    f._task_state = RUNNING

    # Task to run
    f.task_canceller = _CancelFuture
    f._task_lock = threading.Lock()

    # Run in separate thread
    executeAsyncTask(f, _DoResolutionShiftFactor,
                     args=(f, detector, escan, logpath))
    return f

def _DoResolutionShiftFactor(future, detector, escan, logpath):
    """
    Acquires SEM images of several resolution values (from largest to smallest)
    and detects the shift between each image and the largest one using phase
    correlation. To this end, it has to resample the smaller resolution image to
    larger’s image resolution in order to feed it to the phase correlation. Then
    it does linear fit for tangent of these shift values.
    future (model.ProgressiveFuture): Progressive future provided by the wrapper
    detector (model.Detector): The se-detector
    escan (model.Emitter): The e-beam scanner
    logpath (string or None): if not None, will store the acquired SEM images
      in the directory.
    returns (tuple of floats): slope of linear fit
            (tuple of floats): intercept of linear fit
    raises:
        CancelledError() if cancelled
        IOError if shift cannot be estimated
    """
    logger.info("Starting Resolution-related shift calculation...")
    try:
        escan.scale.value = (1, 1)
        escan.horizontalFoV.value = 1200e-06  # m
        escan.translation.value = (0, 0)
        if not escan.rotation.readonly:
            escan.rotation.value = 0
        escan.shift.value = (0, 0)
        escan.accelVoltage.value = 5.3e3  # to ensure that features are visible
        escan.spotSize.value = 2.7  # smaller values seem to give a better contrast
        et = escan.dwellTime.clip(7.5e-07) * numpy.prod(escan.resolution.range[1])

        # Start with largest resolution
        max_resolution = escan.resolution.range[1][0]  # pixels
        min_resolution = 256  # pixels
        cur_resolution = max_resolution
        shift_values = []
        resolution_values = []

        detector.data.subscribe(_discard_data)  # unblank the beam
        f = detector.applyAutoContrast()
        f.result()
        detector.data.unsubscribe(_discard_data)

        largest_image = None  # reference image
        smaller_image = None

        images = []
        while cur_resolution >= min_resolution:
            if future._task_state == CANCELLED:
                raise CancelledError()

            # SEM image of current resolution
            scale = max_resolution / cur_resolution
            escan.scale.value = (scale, scale)
            escan.resolution.value = (cur_resolution, cur_resolution)
            # Retain the same overall exposure time
            escan.dwellTime.value = et / numpy.prod(escan.resolution.value)  # s

            smaller_image = detector.data.get(asap=False)
            images.append(smaller_image)

            # First iteration is special
            if largest_image is None:
                largest_image = smaller_image
                # Ignore value between 2048 and 1024
                cur_resolution -= 1024
                continue

            # Resample the smaller image to fit the resolution of the larger image
            resampled_image = zoom(smaller_image, max_resolution / smaller_image.shape[0])
            # Apply phase correlation
            shift_px = MeasureShift(largest_image, resampled_image, 10)
            logger.debug("Computed resolution shift of %s px @ res=%d", shift_px, cur_resolution)

            if abs(shift_px[0]) > 400 or abs(shift_px[1]) > 100:
                logger.warning("Skipping extreme shift of %s px", shift_px)
            else:
                # Fit the 1st order RC circuit model, to be linear
                if shift_px[0] != 0:
                    smx = 1 / math.tan(2 * math.pi * shift_px[0] / max_resolution)
                else:
                    smx = None  # shift
                if shift_px[1] != 0:
                    smy = 1 / math.tan(2 * math.pi * shift_px[1] / max_resolution)
                else:
                    smy = None  # shift

                shift_values.append((smx, smy))
                resolution_values.append(cur_resolution)

            cur_resolution -= 64

        logger.debug("Computed shift of %s for resolutions %s", shift_values, resolution_values)
        if logpath:
            tiff.export(os.path.join(logpath, "res_shift.tiff"), images)

        # Linear fit
        smxs, smys, rx, ry = [], [], [], []
        for r, (smx, smy) in zip(resolution_values, shift_values):
            if smx is not None:
                smxs.append(smx)
                rx.append(r)
            if smy is not None:
                smys.append(smy)
                ry.append(r)

        a_x, b_x = 0, 0
        if smxs:
            coef_x = array([rx, [1] * len(rx)])
            a_nx, b_nx = linalg.lstsq(coef_x.T, smxs)[0]
            logger.debug("Computed linear reg NX as %s, %s", a_nx, b_nx)
            if a_nx != 0:
                a_x = -1 / a_nx
                b_x = b_nx / a_nx

        a_y, b_y = 0, 0
        if smys:
            coef_y = array([ry, [1] * len(ry)])
            a_ny, b_ny = linalg.lstsq(coef_y.T, smys)[0]
            logger.debug("Computed linear reg NY as %s, %s", a_ny, b_ny)
            if a_ny != 0:
                a_y = -1 / a_ny
                b_y = b_ny / a_ny

        return (a_x, a_y), (b_x, b_y)

    finally:
        with future._task_lock:
            if future._task_state == CANCELLED:
                raise CancelledError()
            future._task_state = FINISHED

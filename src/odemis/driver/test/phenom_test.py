#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 7 Feb 2014

Copyright © 2014 Kimon Tsitsikas, Delmic

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

import Pyro4
import copy
import logging
from odemis import model
from odemis.dataio import hdf5
import os
import pickle
import threading
import time
import unittest
from unittest.case import skip

from odemis.driver import phenom

# logging.getLogger().setLevel(logging.DEBUG)

# arguments used for the creation of basic components
CONFIG_SED = {"name": "sed", "role": "sed"}
CONFIG_BSD = {"name": "bsd", "role": "bsd"}
CONFIG_SCANNER = {"name": "scanner", "role": "ebeam"}
CONFIG_FOCUS = {"name": "focus", "role": "ebeam-focus", "axes": ["z"]}
CONFIG_NC_FOCUS = {"name": "navcam_focus", "role": "overview-focus", "axes": ["z"]}
CONFIG_STAGE = {"name": "stage", "role": "stage"}
CONFIG_NAVCAM = {"name": "camera", "role": "overview-ccd"}
CONFIG_PRESSURE = {"name": "pressure", "role": "chamber"}
CONFIG_SEM = {"name": "sem", "role": "sem", "host": "http://Phenom-MVE0206151080.local:8888",
              "username": "delmic", "password" : "6526AM9688B1",
              "children": {"detector": CONFIG_SED, "scanner": CONFIG_SCANNER,
                           "stage": CONFIG_STAGE, "focus": CONFIG_FOCUS,
                           "camera": CONFIG_NAVCAM, "navcam_focus": CONFIG_NC_FOCUS,
                           "pressure": CONFIG_PRESSURE}
              }
@skip("skip")
class TestSEMStatic(unittest.TestCase):
    """
    Tests which don't need a SEM component ready
    """
    def test_creation(self):
        """
        Doesn't even try to acquire an image, just create and delete components
        """
        sem = phenom.SEM(**CONFIG_SEM)
        self.assertEqual(len(sem.children), 7)

        for child in sem.children:
            if child.name == CONFIG_SED["name"]:
                sed = child
            elif child.name == CONFIG_SCANNER["name"]:
                scanner = child

        self.assertEqual(len(scanner.resolution.value), 2)
        self.assertIsInstance(sed.data, model.DataFlow)

        self.assertTrue(sem.selfTest(), "SEM self test failed.")
        sem.terminate()

    def test_error(self):
        wrong_config = copy.deepcopy(CONFIG_SEM)
        wrong_config["device"] = "/dev/comdeeeee"
        self.assertRaises(Exception, phenom.SEM, **wrong_config)


        wrong_config = copy.deepcopy(CONFIG_SEM)
        wrong_config["children"]["scanner"]["channels"] = [1, 1]
        self.assertRaises(Exception, phenom.SEM, **wrong_config)

    def test_pickle(self):
        try:
            os.remove("test")
        except OSError:
            pass
        daemon = Pyro4.Daemon(unixsocket="test")

        sem = phenom.SEM(daemon=daemon, **CONFIG_SEM)

        dump = pickle.dumps(sem, pickle.HIGHEST_PROTOCOL)
#        print "dump size is", len(dump)
        sem_unpickled = pickle.loads(dump)
        self.assertEqual(len(sem_unpickled.children), 7)
        sem.terminate()

class TestSEM(unittest.TestCase):
    """
    Tests which can share one SEM device
    """
    @classmethod
    def setUpClass(cls):
        cls.sem = phenom.SEM(**CONFIG_SEM)

        for child in cls.sem.children:
            if child.name == CONFIG_SED["name"]:
                cls.sed = child
            elif child.name == CONFIG_SCANNER["name"]:
                cls.scanner = child
            elif child.name == CONFIG_FOCUS["name"]:
                cls.focus = child
            elif child.name == CONFIG_STAGE["name"]:
                cls.stage = child
            elif child.name == CONFIG_NAVCAM["name"]:
                cls.camera = child
            elif child.name == CONFIG_NC_FOCUS["name"]:
                cls.navcam_focus = child
            elif child.name == CONFIG_PRESSURE["name"]:
                cls.pressure = child

    @classmethod
    def tearUpClass(cls):
        cls.sem.terminate()
        time.sleep(3)

    def setUp(self):
        # reset resolution and dwellTime
        self.scanner.scale.value = (1, 1)
        self.scanner.resolution.value = (512, 256)
        self.size = self.scanner.resolution.value
        self.scanner.dwellTime.value = self.scanner.dwellTime.range[0]
        self.acq_dates = (set(), set())  # 2 sets of dates, one for each receiver
        self.acq_done = threading.Event()

    def tearUp(self):
#        print gc.get_referrers(self.camera)
#        gc.collect()
        pass

    def assertTupleAlmostEqual(self, first, second, places=None, msg=None, delta=None):
        """
        check two tuples are almost equal (value by value)
        """
        for f, s in zip(first, second):
            self.assertAlmostEqual(f, s, places=places, msg=msg, delta=delta)

    def compute_expected_duration(self):
        dwell = self.scanner.dwellTime.value
        settle = 5.e-6
        size = self.scanner.resolution.value
        return size[0] * size[1] * dwell + size[1] * settle
    @skip("skip")
    def test_acquire(self):
        self.scanner.dwellTime.value = 10e-6  # s
        expected_duration = self.compute_expected_duration()

        start = time.time()
        im = self.sed.data.get()
        duration = time.time() - start
        print im.metadata
        hdf5.export("PhenomTest", im)
        self.assertEqual(im.shape, self.size[::-1])
        self.assertGreaterEqual(duration, expected_duration, "Error execution took %f s, less than exposure time %d." % (duration, expected_duration))
        self.assertIn(model.MD_DWELL_TIME, im.metadata)
    @skip("skip")
    def test_hfv(self):
        orig_pxs = self.scanner.pixelSize.value
        orig_hfv = self.scanner.horizontalFoV.value
        self.scanner.horizontalFoV.value = orig_hfv / 2

        self.assertAlmostEqual(orig_pxs[0] / 2, self.scanner.pixelSize.value[0])
    @skip("skip")
    def test_roi(self):
        """
        check that .translation and .scale work
        """

        # First, test simple behaviour on the VA
        # max resolution
        max_res = self.scanner.resolution.range[1]
        self.scanner.scale.value = (1, 1)
        self.scanner.resolution.value = max_res
        self.scanner.translation.value = (-1, 1)  # will be set back to 0,0 as it cannot move
        self.assertEqual(self.scanner.translation.value, (0, 0))

        # scale up
        self.scanner.scale.value = (16, 16)
        exp_res = (max_res[0] // 16, max_res[1] // 16)
        self.assertTupleAlmostEqual(self.scanner.resolution.value, exp_res)
        self.scanner.translation.value = (-1, 1)
        self.assertEqual(self.scanner.translation.value, (0, 0))

        # shift
        exp_res = (max_res[0] // 32, max_res[1] // 32)
        self.scanner.resolution.value = exp_res
        self.scanner.translation.value = (-1, 1)
        self.assertTupleAlmostEqual(self.scanner.resolution.value, exp_res)
        self.assertEqual(self.scanner.translation.value, (-1, 1))

        # change scale to some float
        self.scanner.resolution.value = (max_res[0] // 16, max_res[1] // 16)
        self.scanner.scale.value = (1.5, 2.3)
        exp_res = (max_res[0] // 1.5, max_res[1] // 2.3)
        self.assertTupleAlmostEqual(self.scanner.resolution.value, exp_res)
        self.assertEqual(self.scanner.translation.value, (0, 0))

        self.scanner.scale.value = (1, 1)
        self.assertTupleAlmostEqual(self.scanner.resolution.value, max_res, delta=1.1)
        self.assertEqual(self.scanner.translation.value, (0, 0))

        # Then, check metadata fits with the expectations
        center = (1e3, -2e3)  # m
        # simulate the information on the position (normally from the mdupdater)
        self.scanner.updateMetadata({model.MD_POS: center})

        self.scanner.resolution.value = max_res
        self.scanner.scale.value = (16, 16)
        self.scanner.dwellTime.value = self.scanner.dwellTime.range[0]

        # normal acquisition
        im = self.sed.data.get()
        self.assertEqual(im.shape, self.scanner.resolution.value[-1::-1])
        self.assertTupleAlmostEqual(im.metadata[model.MD_POS], center)

        # shift a bit
        # reduce the size of the image so that we can have translation
        self.scanner.resolution.value = (max_res[0] // 32, max_res[1] // 32)
        self.scanner.translation.value = (-1.26, 10)  # px
        pxs = self.scanner.pixelSize.value
        exp_pos = (center[0] + (-1.26 * pxs[0]),
                   center[1] - (10 * pxs[1]))  # because translation Y is opposite from physical one
        im = self.sed.data.get()
        self.assertEqual(im.shape, self.scanner.resolution.value[-1::-1])
        self.assertTupleAlmostEqual(im.metadata[model.MD_POS], exp_pos)

        # only one point
#         self.scanner.resolution.value = (1, 1)
#         print self.scanner.resolution.value, self.scanner.scale.value, self.scanner.dwellTime.value
#         im = self.sed.data.get()
#         hdf5.export("test3.h5", model.DataArray(im))
#         self.assertEqual(im.shape, self.scanner.resolution.value[-1::-1])
#         self.assertTupleAlmostEqual(im.metadata[model.MD_POS], exp_pos)

    @skip("faster")
    def test_acquire_high_osr(self):
        """
        small resolution, but large osr, to force acquisition not by whole array
        """
        self.scanner.resolution.value = (256, 200)
        self.size = self.scanner.resolution.value
        self.scanner.dwellTime.value = self.scanner.dwellTime.range[0] * 100
        expected_duration = self.compute_expected_duration()  # about 1 min

        start = time.time()
        im = self.sed.data.get()
        duration = time.time() - start

        self.assertEqual(im.shape, self.size[-1:-3:-1])
        self.assertGreaterEqual(duration, expected_duration, "Error execution took %f s, less than exposure time %d." % (duration, expected_duration))
        self.assertIn(model.MD_DWELL_TIME, im.metadata)
    @skip("skip")
    def test_long_dwell_time(self):
        """
        one pixel only, but long dwell time (> 4s), which means it uses 
        duplication rate.
        """
        self.scanner.resolution.value = self.scanner.resolution.range[0]
        self.size = self.scanner.resolution.value
        self.scanner.dwellTime.value = self.scanner.dwellTime.range[1]  # DPR should be 3
        expected_duration = self.compute_expected_duration()  # same as dwell time

        start = time.time()
        im = self.sed.data.get()
        duration = time.time() - start

        self.assertEqual(im.shape, self.size[::-1])
        self.assertGreaterEqual(duration, expected_duration, "Error execution took %f s, less than exposure time %d." % (duration, expected_duration))
        self.assertIn(model.MD_DWELL_TIME, im.metadata)
    @skip("skip")
    def test_acquire_long_short(self):
        """
        test being able to cancel image acquisition if dwell time is too long
        """
        self.scanner.resolution.value = (256, 200)
        self.size = self.scanner.resolution.value
        self.scanner.dwellTime.value = self.scanner.dwellTime.range[0] * 100
        expected_duration_l = self.compute_expected_duration()  # about 5 s

        self.left = 1
        start = time.time()

        # acquire one long, and change to a short time
        self.sed.data.subscribe(self.receive_image)
        # time.sleep(0.1) # make sure it has started
        self.scanner.dwellTime.value = self.scanner.dwellTime.range[0]  # shorten
        expected_duration_s = self.compute_expected_duration()
        # unsub/sub should always work, as long as there is only one subscriber
        self.sed.data.unsubscribe(self.receive_image)
        self.sed.data.subscribe(self.receive_image)

        self.acq_done.wait(2 + expected_duration_l * 1.1)
        duration = time.time() - start

        self.assertTrue(self.acq_done.is_set())
        self.assertGreaterEqual(duration, expected_duration_s, "Error execution took %f s, less than exposure time %f." % (duration, expected_duration_s))
        self.assertLess(duration, expected_duration_l, "Execution took %f s, as much as the long exposure time %f." % (duration, expected_duration_l))
    @skip("skip")
    def test_acquire_flow(self):
        expected_duration = self.compute_expected_duration()

        number = 5
        self.left = number
        self.sed.data.subscribe(self.receive_image)

        self.acq_done.wait(number * (2 + expected_duration * 1.1))  # 2s per image should be more than enough in any case

        self.assertEqual(self.left, 0)
    @skip("skip")
    def test_acquire_with_va(self):
        """
        Change some settings before and while acquiring
        """
        dwell = self.scanner.dwellTime.range[0] * 2
        self.scanner.dwellTime.value = dwell
        self.scanner.resolution.value = self.scanner.resolution.range[1]  # test big image
        self.size = self.scanner.resolution.value
        expected_duration = self.compute_expected_duration()

        number = 3
        self.left = number
        self.sed.data.subscribe(self.receive_image)

        # change the attribute
        time.sleep(expected_duration)
        dwell = self.scanner.dwellTime.range[0]
        self.scanner.dwellTime.value = dwell
        expected_duration = self.compute_expected_duration()

        self.acq_done.wait(number * (2 + expected_duration * 1.1))  # 2s per image should be more than enough in any case

        self.sed.data.unsubscribe(self.receive_image)  # just in case it failed
        self.assertEqual(self.left, 0)
    @skip("skip")
    def test_df_fast_sub_unsub(self):
        """
        Test the dataflow on a very fast cycle subscribing/unsubscribing
        SEMComedi had a bug causing the threads not to start again
        """
        self.scanner.dwellTime.value = self.scanner.dwellTime.range[0]
        number = 10
        expected_duration = self.compute_expected_duration()

        self.left = 10000  # don't unsubscribe automatically

        for i in range(number):
            self.sed.data.subscribe(self.receive_image)
            time.sleep(0.001 * i)
            self.sed.data.unsubscribe(self.receive_image)

        # now this one should work
        self.sed.data.subscribe(self.receive_image)
        time.sleep(expected_duration * 2)  # make sure we received at least one image
        self.sed.data.unsubscribe(self.receive_image)

        self.assertLessEqual(self.left, 10000 - 1)
    @skip("skip")
    def test_df_alternate_sub_unsub(self):
        """
        Test the dataflow on a quick cycle subscribing/unsubscribing
        Andorcam3 had a real bug causing deadlock in this scenario
        """
        self.scanner.dwellTime.value = 10e-6
        number = 5
        expected_duration = self.compute_expected_duration()

        self.left = 10000 + number  # don't unsubscribe automatically

        for i in range(number):
            self.sed.data.subscribe(self.receive_image)
            time.sleep(expected_duration * 1.2)  # make sure we received at least one image
            self.sed.data.unsubscribe(self.receive_image)

        # if it has acquired a least 5 pictures we are already happy
        self.assertLessEqual(self.left, 10000)

    def onEvent(self):
        self.events += 1

    def receive_image(self, dataflow, image):
        """
        callback for df of test_acquire_flow()
        """
        self.assertEqual(image.shape, self.size[-1:-3:-1])
        self.assertIn(model.MD_DWELL_TIME, image.metadata)
        self.acq_dates[0].add(image.metadata[model.MD_ACQ_DATE])
#        print "Received an image"
        self.left -= 1
        if self.left <= 0:
            dataflow.unsubscribe(self.receive_image)
            self.acq_done.set()
    @skip("skip")
    def test_focus(self):
        """
        Check it's possible to change the focus
        """
        pos = self.focus.position.value
        f = self.focus.moveRel({"z":0.1e-3})  # 1 mm
        f.result()
        self.assertNotEqual(self.focus.position.value, pos)
#         self.sed.data.get()

        f = self.focus.moveRel({"z":-0.3e-3})  # 10 mm
        f.result()
        self.assertNotEqual(self.focus.position.value, pos)
#         self.sed.data.get()

        # restore original position
        f = self.focus.moveAbs(pos)
        f.result()
        self.assertAlmostEqual(self.focus.position.value, pos, 5)

    # @skip("skip")
    def test_move(self):
        """
        Check it's possible to move the stage
        """
        pos = self.stage.position.value
        f = self.stage.moveRel({"x":-100e-6, "y":-100e-6})  # 1 mm
        f.result()
        self.assertTupleAlmostEqual(self.stage.position.value, pos)
        time.sleep(1)
        f = self.stage.moveRel({"x":100e-6, "y":100e-6})  # 1 mm
        f.result()
        self.assertTupleAlmostEqual(self.stage.position.value, pos)

    @skip("skip")
    def test_navcam(self):
        """
        Check it's possible to move the stage
        """
        # Exposure time is fixed, time is mainly spent on the image transfer
        expected_duration = 0.5  # s
        start = time.time()
        img = self.camera.data.get()
        duration = time.time() - start
        self.assertGreaterEqual(duration, expected_duration, "Error execution took %f s, less than exposure time %d." % (duration, expected_duration))

    @skip("skip")
    def test_navcam_focus(self):
        """
        Check it's possible to change the overview focus
        """
        pos = self.navcam_focus.position.value
        f = self.navcam_focus.moveRel({"z":0.1e-3})  # 1 mm
        f.result()
        self.assertNotEqual(self.navcam_focus.position.value, pos)
        time.sleep(1)

        # restore original position
        f = self.navcam_focus.moveAbs(pos)
        f.result()
        self.assertAlmostEqual(self.navcam_focus.position.value, pos)

    @skip("skip")
    def test_pressure(self):
        """
        Check it's possible to change the pressure state
        """
        f = self.pressure.moveAbs({"pressure":1e04})  # move to NavCam
        f.result()
        new_pos = self.pressure.position.value["pressure"]
        self.assertEqual({'pressure': 10000.0}, new_pos)


if __name__ == "__main__":
    unittest.main()

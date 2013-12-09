# -*- coding: utf-8 -*-
"""
Created on 28 Nov 2013

@author: Kimon Tsitsikas

Copyright © 2012-2013 Kimon Tsitsikas, Delmic

This file is part of Odemis.

Odemis is free software: you can redistribute it and/or modify it under the
terms  of the GNU General Public License version 2 as published by the Free
Software  Foundation.

Odemis is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY;  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR  PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
Odemis. If not, see http://www.gnu.org/licenses/.
"""

from __future__ import division

import numpy
import math
import scipy
import operator
import scipy.signal
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from odemis import model
from odemis import dataio
from numpy import unravel_index
from numpy import argsort
from numpy import histogram
from scipy.spatial import cKDTree
from operator import itemgetter


def FindCenterCoordinates(subimages):
    """
    For each subimage generated by DivideInNeighborhoods, detects the center 
    of the contained spot. Finally produces a list with the center coordinates 
    corresponding to each subimage.
    subimages (List of model.DataArray): List of 2D arrays containing pixel intensity
    returns (List of tuples): Coordinates of spot centers
    """
    number_of_subimages = subimages.__len__()
    spot_coordinates = []

    # Pop each subimage from the list
    for i in xrange(number_of_subimages):
        subimage = subimages[i]
        subimage_x, subimage_y = subimage.shape

        # See Parthasarathy's paper for details
        xk_onerow = numpy.arange(-(subimage_y - 1) / 2 + 0.5, (subimage_y - 1) / 2, 1)
        (xk_onerow_x,) = xk_onerow.shape
        xk = numpy.tile(xk_onerow, subimage_x - 1)
        xk = xk.reshape((subimage_x - 1, xk_onerow_x))
        yk_onecol = numpy.arange((subimage_x - 1) / 2 - 0.5, -(subimage_x - 1) / 2, -1)
        (yk_onecol_x,) = yk_onecol.shape
        yk_onecol = yk_onecol.reshape((yk_onecol_x, 1))
        yk = numpy.tile(yk_onecol, subimage_y - 1)

        dIdu = subimage[0:subimage_x - 1, 1:subimage_y] - subimage[1:subimage_x, 0:subimage_y - 1]
        dIdv = subimage[0:subimage_x - 1, 0:subimage_y - 1] - subimage[1:subimage_x, 1:subimage_y]

        # Smoothing
        h = numpy.tile(numpy.ones(3) / 9, 3).reshape(3, 3)  # simple 3x3 averaging filter
        dIdu = scipy.signal.convolve2d(dIdu, h, mode='same', fillvalue=0)
        dIdv = scipy.signal.convolve2d(dIdv, h, mode='same', fillvalue=0)

        # Calculate intensity gradient in xy coordinate system
        dIdx = dIdu - dIdv
        dIdy = dIdu + dIdv

        # Assign a,b
        a = -dIdy
        b = dIdx

        # Normalize such that a^2 + b^2 = 1
        I2 = numpy.hypot(a, b)
        s = (I2 != 0)
        a[s] = a[s] / I2[s]
        b[s] = b[s] / I2[s]

        # Solve for c
        c = -a * xk - b * yk

        # Weighting: weight by square of gradient magnitude and inverse distance to gradient intensity centroid.
        dI2 = dIdu * dIdu + dIdv * dIdv
        sdI2 = numpy.sum(dI2[:])
        x0 = numpy.sum(dI2[:] * xk[:]) / sdI2
        y0 = numpy.sum(dI2[:] * yk[:]) / sdI2
        w = dI2 / (0.05 + numpy.sqrt((xk - x0) * (xk - x0) + (yk - y0) * (yk - y0)))

        # Make the edges zero, because of the filter
        w[0, :] = 0
        w[w.shape[0] - 1, :] = 0
        w[:, 0] = 0
        w[:, w.shape[1] - 1] = 0

        # Find radial center
        swa2 = numpy.sum(w[:] * a[:] * a[:])
        swab = numpy.sum(w[:] * a[:] * b[:])
        swb2 = numpy.sum(w[:] * b[:] * b[:])
        swac = numpy.sum(w[:] * a[:] * c[:])
        swbc = numpy.sum(w[:] * b[:] * c[:])
        det = swa2 * swb2 - swab * swab
        xc = (swab * swbc - swb2 * swac) / det
        yc = (swab * swac - swa2 * swbc) / det

        # Output relative to upper left coordinate
        xc = xc + (subimage_y + 1) / 2
        yc = -yc + (subimage_x + 1) / 2
        spot_coordinates.append((xc, yc))

    return spot_coordinates

def DivideInNeighborhoods(image, number_of_spots):
    """
    Given an image that includes N spots, divides it in N subimages with each of them 
    to include one spot. Briefly, it filters the image, finds the N “brightest” spots 
    and crops the region around them generating the subimages. This process is repeated 
    until image division is feasible.
    image (model.DataArray): 2D array containing the intensity of each pixel
    number_of_spots (int,int): The number of CL spots
    returns subimages (List of DataArrays): One subimage per spot
            subimage_coordinates (List of tuples): The coordinates of the center of each 
                                                subimage with respect to the overall image
            subimage_size (int): One dimension because it is square
    """
    subimage_coordinates = []
    subimages = []

    # Determine size of filter window
    filter_window_size = int(image.size / (3 * ((number_of_spots[0] * number_of_spots[1]) ** 2)))

    # Determine threshold
    i_max, j_max = unravel_index(image.argmax(), image.shape)
    i_min, j_min = unravel_index(image.argmin(), image.shape)
    max_diff = image[i_max, j_max] - image[i_min, j_min]
    threshold = max_diff / 3

    # Filter the parts of the image with variance in intensity greater
    # than the threshold
    data_max = filters.maximum_filter(image, filter_window_size)
    maxima = (image == data_max)
    data_min = filters.minimum_filter(image, filter_window_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)

    # Go through these parts and crop the subimages based on the neighborhood_size value
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1) / 2
        y_center = (dy.start + dy.stop - 1) / 2

        subimage_coordinates.append((x_center, y_center))
        # TODO: change +10 and -10 to number relative to spot size
        subimage = image[(dy.start - filter_window_size):(dy.stop + 1 + filter_window_size), (dx.start - filter_window_size):(dx.stop + 1 + filter_window_size)]
        subimages.append(subimage)

    #Take care of fault spots (e.g. cosmic ray)
    clean_subimages, clean_subimage_coordinates = FilterCosmicRay(image, subimages, subimage_coordinates)
    subimage_size = subimage.shape[0]

    return clean_subimages, clean_subimage_coordinates, subimage_size

def ReconstructImage(subimage_coordinates, spot_coordinates, subimage_size):
    """
    Given the coordinates of each subimage as also the coordinates of the spot into it, 
    generates the coordinates of the spots with respect to the overall image.
    subimage_coordinates (List of tuples): The coordinates of the 
                                        center of each subimage with 
                                        respect to the overall image
    spot_coordinates (List of tuples): Coordinates of spot centers
    subimage_size(int): One dimension because it is square
    returns (List of tuples): Coordinates of spots in optical image
    """
    optical_coordinates = []
    center_position = (subimage_size / 2) - 1
    for ta, tb in zip(subimage_coordinates, spot_coordinates):
        t = tuple(a + (b - center_position) for a, b in zip(ta, tb))
        optical_coordinates.append(t)

    return optical_coordinates

def FilterCosmicRay(image, subimages, subimage_coordinates):
    """
    It removes subimages that contain cosmic rays.
    image (model.DataArray): 2D array containing the intensity of each pixel
    subimages (List of model.DataArray): List of 2D arrays containing pixel intensity
    returns (List of model.DataArray): List of subimages without the ones containing
                                       cosmic ray
            (List of tuples): The coordinates of the center of each subimage with respect 
                            to the overall image
    """
    number_of_subimages = subimages.__len__()
    clean_subimages = []
    clean_subimage_coordinates = []
    for i in xrange(number_of_subimages):
        hist, bin_edges = histogram(subimages[i], bins=10)
        # Remove subimage if its istogram implies a cosmic ray
        if ~((hist[3:7] == numpy.zeros(4)).all()):
            clean_subimages.append(subimages[i])
            clean_subimage_coordinates.append(subimage_coordinates[i])
            
    # If we removed more than 3 subimages give up and return the initial list
    # This is based on the assumption that each image would contain at maximum
    # 3 cosmic rays.
    if (((subimages.__len__()-clean_subimages.__len__())>3) or (clean_subimages.__len__()==0)):
        clean_subimages = subimages
        clean_subimage_coordinates = subimage_coordinates

    return clean_subimages, clean_subimage_coordinates

def MatchCoordinates(optical_coordinates, electron_coordinates):
    """
    Orders the list of spot coordinates generated by FindCenterCoordinates in order to 
    match the corresponding spot coordinates of the grid in the electron image.
    optical_coordinates (List of tuples): Coordinates of spots in optical image
    electron_coordinates (List of tuples): Coordinates of spots in electron image
    returns (List of tuples): Ordered list of coordinates in optical image with respect 
                                to the order in the electron image
    """
    index = KNNsearch(electron_coordinates, optical_coordinates)

    # Sort optical coordinates based on the KNNsearch output index
    sorted_optical = [y for (x, y) in sorted(zip(index, optical_coordinates))]
    """
    index_optical = zip(index, optical_coordinates)
    index_optical.sort()
    optical_sorted = [x for y, x in yx]
    """
    return sorted_optical

def KNNsearch(x_coordinates, y_coordinates):
    """
    Applies K-nearest neighbors search to the lists x_coordinates and y_coordinates.
    x_coordinates (List of tuples): List of coordinates
    y_coordinates (List of tuples): List of coordinates
    returns (List of integers): Contains the index of nearest neighbor in x_coordinates 
                                for the corresponding element in y_coordinates
    """
    points = numpy.array(x_coordinates)
    tree = cKDTree(points)
    distance, index = tree.query(y_coordinates)

    return index

def TransfromCoordinates(x_coordinates, translation, rotation, scale):
    """
    Transforms the x_coordinates according to the parameters.
    x_coordinates (List of tuples): List of coordinates
    translation (Tuple of floats): Translation
    rotation (float): Rotation #degrees
    scale (float): Scaling
    returns (List of tuples): Transformed coordinates
    """
    transformed_coordinates = []
    for ta in x_coordinates:
        # scaling-rotation-translation
        tuple_scale = (scale, scale)
        scaled = tuple(map(operator.mul, ta, tuple_scale))

        x, y = scaled
        rad_rotation = rotation * (math.pi / 180)  # rotation in radians, counterclockwise
        x_rotated = x * math.cos(rad_rotation) - y * math.sin(rad_rotation)
        y_rotated = x * math.sin(rad_rotation) + y * math.cos(rad_rotation)
        rotated = (x_rotated, y_rotated)
        translated = tuple(map(operator.add, rotated, translation))
        transformed_coordinates.append(translated)

    return transformed_coordinates

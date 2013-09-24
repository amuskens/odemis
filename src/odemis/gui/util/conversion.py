#-*- coding: utf-8 -*-
'''
@author: Rinze de Laat

Copyright © 2012 Rinze de Laat, Éric Piel, Delmic

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

# Inspired by code from:
# http://codingmess.blogspot.nl/2009/05/conversion-of-wavelength-in-nanometers.html
# based on:
# http://www.physics.sfasu.edu/astro/color/spectra.html
def wave2rgb(wavelength):
    """
    Convert a wavelength into a (r,g,b) value
    wavelength (0<float): wavelength in m
    return (3-tupe int in 0..255): RGB value
    """
    w = wavelength * 1e9
    # outside of the visible spectrum, use fixed colour
    w = min(max(w, 350), 780)

    # colour
    if 350 <= w < 440:
        r = -(w - 440) / (440 - 350)
        g = 0
        b = 1
    elif 440 <= w < 490:
        r = 0
        g = (w - 440) / (490 - 440)
        b = 1
    elif 490 <= w < 510:
        r = 0
        g = 1
        b = -(w - 510) / (510 - 490)
    elif 510 <= w < 580:
        r = (w - 510) / (580 - 510)
        g = 1
        b = 0
    elif 580 <= w < 645:
        r = 1
        g = -(w - 645) / (645 - 580)
        b = 0
    elif 645 <= w <= 780:
        r = 1
        g = 0
        b = 0
    else:
        logging.warning("Unable to compute RGB for wavelength %d", w)

    return int(round(255 * r)), int(round(255 * g)), int(round(255 * b))

def hex_to_rgb(hex_str):
    """
    Convert a Hexadecimal color representation into an 3-tuple of ints
    return (tuple of 3 (0<int<255): R, G, and B
    """
    hex_str = hex_str[-6:]
    return tuple(int(hex_str[i:i + 2], 16) for i in [0, 2, 4])

def hex_to_rgba(hex_str, af=255):
    """ Convert a Hexadecimal color representation into an 4-tuple of ints """
    return hex_to_rgb(hex_str) + (af,)

def wxcol_to_rgb(wxcol):
    return (wxcol.Red(), wxcol.Green(), wxcol.Blue())

# To handle RGB as floats (for Cairo, etc.)
def hex_to_frgb(hex_str):
    """
    Convert a Hexadecimal color representation into an 3-tuple of floats
    return (tuple of 3 (0<float<1): R, G, and B
    """
    hex_str = hex_str[-6:]
    return tuple(int(hex_str[i:i + 2], 16) / 255 for i in [0, 2, 4])

def hex_to_frgba(hex_str, af=1.0):
    """ Convert a Hexadecimal color representation into an 4-tuple of floats """
    return hex_to_frgb(hex_str) + (af,)

def wxcol_to_frgb(wxcol):
    return (wxcol.Red() / 255, wxcol.Green() / 255, wxcol.Blue() / 255)

def change_brightness(colf, weight):
    """
    Brighten (or darken) a given colour
    See also wx.lib.agw.aui.aui_utilities.StepColour() and Colour.ChangeLightness() from 3.0 
    colf (tuple of 3+ 0<float<1): RGB colour (and alpha)
    weight (-1<float<1): how much to brighten (>0) or darken (<0) 
    return (tuple of 3+ 0<float<1): new RGB colour
    """
    if weight > 0:
        # blend towards white
        f, lim = min, 1.0
    else:
        # blend towards black
        f, lim = max, 0.0
        weight = -weight

    new_col = tuple(f(c * (1 - weight) + lim * weight, lim) for c in colf[:3])

    return new_col + colf[3:]

TOP_LEFT = 0
TOP_RIGHT = 1
BOTTOM_LEFT = 2
BOTTOM_RIGHT = 3
def dichotomy_to_region(seq):
    """
    Converts a dichotomy sequence into a region
    See DichotomyOverlay for more information
    seq (list of 0<=int<4): list of sub part selected
    returns (tuple of 4 0<=float<=1): left, top, right, bottom (in ratio)
    """
    roi = [0, 0 , 1 , 1] # starts from the whole area
    for quad in seq:
        l, t, r, b = roi
        # divide the roi according to the quadrant
        if quad in [TOP_LEFT, BOTTOM_LEFT]:
            r = l + (r - l) / 2
        else:
            l = (r + l) / 2
        if quad in [TOP_LEFT, TOP_RIGHT]:
            b = t + (b - t) / 2
        else:
            t = (b + t) / 2
        assert(0 <= l <= r <= 1 and 0 <= t <= b <= 1)
        roi = [l, t, r, b]

    return roi


def normalize_rect(rect):
    """
    Ensure that a rectangle has a the left less than right, and top less than
    bottom.
    rect (iterable of 4 floats): left, top, right, bottom
    return (iterable of 4 floats): left, top, right, bottom
    """
    l, t, r, b = rect
    if l > r:
        l, r = r, l
    if t > b:
        t , b = b, t

    nrect = type(rect)((l, t, r, b))
    return nrect
